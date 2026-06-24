#!/bin/bash
set -euo pipefail

# 분산처리 테스트 모드: DIST_TEST=1 이면 startup_dist.sh 로 위임
[ "${DIST_TEST:-0}" = "1" ] && exec /bin/bash /startup_dist.sh

REPO=https://github.com/seongwonM/SH-ARD-embedding-develop.git
if [ -n "${GH_TOKEN:-}" ]; then
    TARGET="https://${GH_TOKEN}@github.com/seongwonM/SH-ARD-embedding-develop.git"
else
    TARGET="$REPO"
fi

# git clone 한 번만 시도 — 실패해도 빌드된 코드로 계속 진행
if rm -rf /tmp/latest && git clone --depth=1 "$TARGET" /tmp/latest 2>&1; then
    cp -rf /tmp/latest/bench /app/
    echo "[startup] 코드 업데이트 완료"
else
    echo "[startup] git clone 실패 — 빌드된 코드로 실행"
    mkdir -p /tmp/latest
fi

MODEL_SAFE=$(echo "${MODEL_ID:-}" | tr '/' '_')
MODE_SUFFIX=${VECTOR_MODE:+_${VECTOR_MODE}}
VECTOR_DB="${VECTOR_DB:-qdrant}"
DB_SUFFIX="_${VECTOR_DB}"

# Instant Cluster에서 NODE_RANK가 주입되면 각 pod이 독립 경로 사용 (NFS 충돌 방지)
# DIST_TEST 없이도 multi-pod 병렬 벤치마크 가능
_RANK_SUFFIX="${NODE_RANK:+_rank${NODE_RANK}}"
_GPU_RAW=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
_GPU_SAFE=$(echo "$_GPU_RAW" | sed 's/NVIDIA //g; s/Tesla //g' | tr ' ' '_' | tr -cd 'a-zA-Z0-9_' | cut -c1-20)
_GPU_SUFFIX="${_GPU_SAFE:+_${_GPU_SAFE}}"
REPORTS_PATH="/workspace/reports/${MODEL_SAFE:-all}${MODE_SUFFIX:-}${DB_SUFFIX}${_RANK_SUFFIX}"

if [ "$VECTOR_DB" = "milvus" ]; then
    # Milvus Standalone (DEB 패키지 바이너리) — embedded etcd + local storage + woodpecker
    MILVUS_DATA="/workspace/milvus_data/${MODEL_SAFE:-default}${MODE_SUFFIX:-}${_RANK_SUFFIX}"
    # 디렉터리 보장 (기존 데이터 보존 — runner.py가 불완전 컬렉션 감지 시 drop/재색인)
    mkdir -p "${MILVUS_DATA}/etcd" "${MILVUS_DATA}/woodpecker" "${MILVUS_DATA}/local"

    # DEB 기본 milvus.yaml을 베이스로 유지하고, 우리가 바꿀 값만 Python deep-merge로 패치.
    # 통째로 덮어쓰면 DEB 기본값(minSizeFromIdleToSealed 등)이 Go 기본값 0으로 떨어져 패닉 발생.
    python3 - "${MILVUS_DATA}" << 'PYEOF'
import yaml, copy, sys

milvus_data = sys.argv[1]

def deep_merge(base, override):
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result

with open('/etc/milvus/configs/milvus.yaml') as f:
    base = yaml.safe_load(f)

overrides = {
    'mq': {'type': 'woodpecker'},
    'woodpecker': {
        'storage': {'type': 'local', 'rootPath': f'{milvus_data}/woodpecker'},
        'logstore': {'segmentSyncPolicy': {'maxIntervalForLocalStorage': '10ms'}},
    },
    'etcd': {
        'endpoints': 'localhost:2379',
        'use': {'embed': True},
        'data': {'dir': f'{milvus_data}/etcd'},
        'config': {'path': '/etc/milvus/configs/embedEtcd.yaml'},
    },
    'localStorage': {'path': f'{milvus_data}/local'},
    'common': {'storageType': 'local'},
    'streaming': {
        'flush': {
            'memoryThreshold': 0.4,
            'growingSegmentBytesHwmThreshold': 0.08,
            'growingSegmentBytesLwmThreshold': 0.04,
        },
    },
    'rootCoord': {'port': 53100},
    'dataCoord': {'port': 13333},
    'queryCoord': {'port': 19531},
    'queryNode': {'port': 21123},
    'indexNode': {'port': 21121},
    'dataNode': {'port': 21124},
    'proxy': {
        'port': 19530,
        'internalPort': 19529,
        'grpc': {
            'serverMaxRecvSize': 268435456,
            'serverMaxSendSize': 268435456,
            'clientMaxRecvSize': 268435456,
            'clientMaxSendSize': 268435456,
        },
    },
}

merged = deep_merge(base, overrides)
with open('/etc/milvus/configs/milvus.yaml', 'w') as f:
    yaml.dump(merged, f, default_flow_style=False, allow_unicode=True)
print('[milvus] config deep-merge 완료')
PYEOF

    echo "[milvus] config 완료: $(ls /etc/milvus/configs/)"
    # MILVUSCONF: initConfPath()가 CWD+/configs를 찾는데 /app/configs 없음 → yaml 미로드
    #   → MILVUSCONF로 명시해야 /etc/milvus/configs/milvus.yaml을 읽음
    # ETCD_DATA_DIR: DEB yaml의 etcd.data.data.dir 키 버그 우회 (코드는 etcd.data.dir 읽음)
    # 재시도 최대 3회: etcd WAL 부패로 "leader changed" 패닉 발생 시 etcd 초기화 후 재시도
    _milvus_started=0
    for _milvus_try in 1 2 3; do
        MILVUSCONF=/etc/milvus/configs \
        ETCD_DATA_DIR="${MILVUS_DATA}/etcd" \
        DEPLOY_MODE=STANDALONE \
        milvus run standalone >"${MILVUS_DATA}/milvus.log" 2>&1 &
        MILVUS_PID=$!
        echo "[milvus] 서버 시작 (시도 $_milvus_try/3, PID=$MILVUS_PID, log=${MILVUS_DATA}/milvus.log)..."

        _milvus_ok=0
        _milvus_died=0
        for _i in $(seq 1 90); do
            if ! kill -0 "$MILVUS_PID" 2>/dev/null; then
                _milvus_died=1
                break
            fi
            if curl -sf http://localhost:9091/healthz >/dev/null 2>&1; then
                _milvus_ok=1
                break
            fi
            sleep 2
        done

        if [ "$_milvus_ok" -eq 1 ]; then
            _milvus_started=1
            break
        fi

        echo "[milvus] 시도 $_milvus_try 실패 (died=$_milvus_died)"
        echo "=== milvus.log 마지막 50줄 ==="
        tail -50 "${MILVUS_DATA}/milvus.log" || true
        if [ "$_milvus_try" -lt 3 ]; then
            echo "[milvus] etcd WAL 초기화 후 재시도 (재기동 시 컬렉션 재색인 필요)..."
            rm -rf "${MILVUS_DATA}/etcd"
            mkdir -p "${MILVUS_DATA}/etcd"
            sleep 3
        fi
    done

    if [ "$_milvus_started" -eq 0 ]; then
        echo "[milvus] 최종 실패 — 전체 로그:"
        cat "${MILVUS_DATA}/milvus.log" || true
        sleep infinity
    fi
    echo "[milvus] 서버 준비 완료"
    MILVUS_URI="http://localhost:19530"
else
    # Qdrant 서버 시작 (pod별 독립 스토리지 — RocksDB lock 충돌 방지)
    QDRANT_STORAGE="/workspace/qdrant_storage/${MODEL_SAFE:-default}${MODE_SUFFIX:-}${_RANK_SUFFIX}"
    mkdir -p "$QDRANT_STORAGE"
    QDRANT__STORAGE__STORAGE_PATH="$QDRANT_STORAGE" \
    QDRANT__LOG_LEVEL=WARN \
    QDRANT__SERVICE__MAX_REQUEST_SIZE_MB=2048 \
    QDRANT__SERVICE__REQUEST_TIMEOUT=600 \
    qdrant --disable-telemetry &
    QDRANT_PID=$!
    echo "[qdrant] 서버 시작 (PID=$QDRANT_PID), 준비 대기 중..."
    until curl -sf http://localhost:6333/healthz >/dev/null 2>&1; do sleep 1; done
    echo "[qdrant] 서버 준비 완료"
fi

MODEL_ARG=""
[ -n "${MODEL_ID:-}" ] && MODEL_ARG="--model ${MODEL_ID}"

MODE_ARG=""
[ -n "${VECTOR_MODE:-}" ] && MODE_ARG="--vector-mode ${VECTOR_MODE}"

CONC_ARG=""
[ -n "${SEARCH_CONCURRENCY:-}" ] && CONC_ARG="--search-concurrency ${SEARCH_CONCURRENCY}"

# 데이터 경로 결정: git clone 성공 시 /tmp/latest/data, 실패 시 /workspace/datasets fallback
DATA_DIR="/tmp/latest/data"
if [ ! -f "${DATA_DIR}/corpus_all.parquet" ]; then
    if [ -f "/workspace/datasets/corpus_all.parquet" ]; then
        DATA_DIR="/workspace/datasets"
        echo "[startup] git clone 내 data 없음 → fallback: ${DATA_DIR}"
    else
        echo "[startup] ERROR: 데이터 없음 (${DATA_DIR}, /workspace/datasets 모두 확인 실패)"
        exit 1
    fi
fi
echo "[startup] 데이터 경로: ${DATA_DIR}"

python -m bench.runner \
    ${MODEL_ARG} \
    ${MODE_ARG} \
    ${CONC_ARG} \
    --model-dtype auto \
    --batch-size "${BATCH_SIZE:-16}" \
    --out "${REPORTS_PATH}" \
    --vectordb "${VECTOR_DB:-qdrant}" \
    --qdrant-url http://localhost:6333 \
    --milvus-uri "${MILVUS_URI:-http://localhost:19530}" \
    --data-root "${DATA_DIR}"

if [ -n "${GH_TOKEN:-}" ]; then
    echo "[git] GH_TOKEN 감지됨, 결과 push 시작..."
    RESULT_FILE="${REPORTS_PATH}/summary.json"
    if [ ! -f "$RESULT_FILE" ]; then
        echo "[git] 결과 파일 없음: $RESULT_FILE — push 스킵"
    else
        cd /tmp/latest
        git config user.email 'pod@runpod.io'
        git config user.name 'RunPod'
        mkdir -p results
        cp "$RESULT_FILE" "results/${MODEL_SAFE:-all}${MODE_SUFFIX:-}${DB_SUFFIX}${_RANK_SUFFIX}${_GPU_SUFFIX}_$(date +%Y%m%d_%H%M%S).json"
        echo "[git] 결과 파일 복사 완료"
        git add results/
        if git diff --cached --quiet; then
            echo "[git] 변경 없음 (이미 push됨)"
        else
            git commit -m "result: ${MODEL_SAFE:-all} 벤치마크 완료"
            # 다른 pod가 먼저 push한 경우 충돌 방지
            git pull --rebase origin main || echo "[git] rebase 실패 — 그냥 push 시도"
            git push && echo "[git] push 완료" || echo "[git] push 실패 — 수동 확인 필요"
        fi
    fi
fi

echo "[완료] 대기 중 (재시작 방지)"
sleep infinity
