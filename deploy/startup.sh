#!/bin/bash
set -euo pipefail

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
REPORTS_PATH="/workspace/reports/${MODEL_SAFE:-all}${MODE_SUFFIX:-}${DB_SUFFIX}"

if [ "$VECTOR_DB" = "milvus" ]; then
    # ── 데이터 경로 (Network Volume /workspace 공유) ──────────────────────────
    MILVUS_DATA="/workspace/milvus_data/${MODEL_SAFE:-default}${MODE_SUFFIX:-}"

    # ── RunPod Instant Cluster 환경변수 기반 모드 결정 ────────────────────────
    # NODE_RANK, PRIMARY_ADDR, NODE_ADDR 는 RunPod이 자동 설정
    _NODE_RANK="${NODE_RANK:-}"
    _PRIMARY_ADDR="${PRIMARY_ADDR:-}"
    _NODE_ADDR="${NODE_ADDR:-localhost}"

    if [ -z "$_NODE_RANK" ]; then
        _MODE="standalone"
    elif [ "$_NODE_RANK" = "0" ]; then
        _MODE="coordinator"
    else
        _MODE="worker"
    fi
    echo "[milvus] 모드: $_MODE (NODE_RANK=${_NODE_RANK:-unset}, PRIMARY_ADDR=${_PRIMARY_ADDR:-N/A})"

    # 디렉터리 보장 (기존 데이터 보존 — runner.py가 불완전 컬렉션 감지 시 drop/재색인)
    mkdir -p "${MILVUS_DATA}/etcd" "${MILVUS_DATA}/woodpecker" "${MILVUS_DATA}/local"
    mkdir -p /tmp/milvuscfg

    # ── 컴포넌트별 milvus.yaml 생성 (DEB 기본값 deep-merge) ──────────────────
    # standalone : 하나의 yaml (etcd embed)
    # coordinator: mixcoord(etcd embed) / streamingnode(localhost) / proxy(localhost)
    # worker     : datanode(PRIMARY_ADDR) / querynode(PRIMARY_ADDR)
    python3 - "${MILVUS_DATA}" "${_MODE}" "${_PRIMARY_ADDR}" << 'PYEOF'
import yaml, copy, sys, os

milvus_data, mode, primary_addr = sys.argv[1], sys.argv[2], sys.argv[3]

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

common = {
    'mq': {'type': 'woodpecker'},
    'woodpecker': {
        'storage': {'type': 'local', 'rootPath': f'{milvus_data}/woodpecker'},
        'logstore': {'segmentSyncPolicy': {'maxIntervalForLocalStorage': '10ms'}},
    },
    'localStorage': {'path': f'{milvus_data}/local'},
    'common': {'storageType': 'local'},
    'streaming': {'flush': {
        'memoryThreshold': 0.4,
        'growingSegmentBytesHwmThreshold': 0.08,
        'growingSegmentBytesLwmThreshold': 0.04,
    }},
    'rootCoord':  {'port': 53100},
    'dataCoord':  {'port': 13333},
    'queryCoord': {'port': 19531},
    'queryNode':  {'port': 21123},
    'indexNode':  {'port': 21121},
    'dataNode':   {'port': 21124},
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

# etcd 설정 변형
etcd_embed  = {
    'endpoints': 'localhost:2379',
    'use': {'embed': True},
    'data': {'dir': f'{milvus_data}/etcd'},
    'config': {'path': '/etc/milvus/configs/embedEtcd.yaml'},
}
etcd_local  = {'endpoints': 'localhost:2379',      'use': {'embed': False}}
etcd_remote = {'endpoints': f'{primary_addr}:2379', 'use': {'embed': False}}

if mode == 'standalone':
    components = {'standalone': etcd_embed}
elif mode == 'coordinator':
    # mixcoord만 embedded etcd 시작, 나머지는 localhost:2379에 연결
    components = {
        'mixcoord':      etcd_embed,
        'streamingnode': etcd_local,
        'proxy':         etcd_local,
    }
else:  # worker
    components = {
        'datanode':  etcd_remote,
        'querynode': etcd_remote,
    }

for comp, etcd_cfg in components.items():
    cfg = deep_merge(base, {**common, 'etcd': etcd_cfg})
    os.makedirs(f'/tmp/milvuscfg/{comp}', exist_ok=True)
    with open(f'/tmp/milvuscfg/{comp}/milvus.yaml', 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    print(f'[milvus] {comp} config 생성')
PYEOF

    # coordinator: embedded etcd를 외부 노드(worker)에서도 접근 가능하도록 설정
    if [ "$_MODE" = "coordinator" ]; then
        python3 - "${_NODE_ADDR}" << 'PYEOF2'
import yaml, sys
node_addr = sys.argv[1]
try:
    with open('/etc/milvus/configs/embedEtcd.yaml') as f:
        cfg = yaml.safe_load(f) or {}
except FileNotFoundError:
    cfg = {}
cfg['listen-client-urls']    = 'http://0.0.0.0:2379'
cfg['advertise-client-urls'] = f'http://{node_addr}:2379'
cfg['listen-peer-urls']      = 'http://0.0.0.0:2380'
with open('/etc/milvus/configs/embedEtcd.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
print(f'[milvus] embedEtcd 외부 접근 활성화 (advertise={node_addr}:2379)')
PYEOF2
    fi

    # ── Standalone ────────────────────────────────────────────────────────────
    if [ "$_MODE" = "standalone" ]; then
        MILVUSCONF=/tmp/milvuscfg/standalone \
        ETCD_DATA_DIR="${MILVUS_DATA}/etcd" \
        DEPLOY_MODE=STANDALONE \
            milvus run standalone >"${MILVUS_DATA}/milvus.log" 2>&1 &
        MILVUS_PID=$!
        echo "[milvus] standalone 시작 (PID=$MILVUS_PID), 준비 대기 중..."

        _milvus_ok=0
        for _i in $(seq 1 90); do
            if ! kill -0 "$MILVUS_PID" 2>/dev/null; then
                echo "[milvus] 프로세스 비정상 종료!"
                echo "=== milvus.log ===" && cat "${MILVUS_DATA}/milvus.log" || true
                sleep infinity
            fi
            curl -sf http://localhost:9091/healthz >/dev/null 2>&1 && { _milvus_ok=1; break; }
            sleep 2
        done
        if [ "$_milvus_ok" -eq 0 ]; then
            echo "[milvus] 헬스체크 타임아웃!"
            cat "${MILVUS_DATA}/milvus.log" || true
            sleep infinity
        fi
        echo "[milvus] standalone 준비 완료"

    # ── Coordinator (Instant Cluster NODE_RANK=0) ──────────────────────────────
    elif [ "$_MODE" = "coordinator" ]; then
        # 1) mixcoord: embedded etcd + rootCoord/dataCoord/queryCoord/indexCoord
        MILVUSCONF=/tmp/milvuscfg/mixcoord \
        ETCD_DATA_DIR="${MILVUS_DATA}/etcd" \
            milvus run mixcoord >"${MILVUS_DATA}/mixcoord.log" 2>&1 &

        # 2) embedded etcd 포트(2379) 열릴 때까지 대기
        echo "[milvus] embedded etcd 대기 중..."
        for _i in $(seq 1 60); do
            nc -z localhost 2379 2>/dev/null && break || sleep 2
        done

        # 3) streamingnode + proxy (localhost etcd에 연결)
        MILVUSCONF=/tmp/milvuscfg/streamingnode \
            milvus run streamingnode >"${MILVUS_DATA}/streamingnode.log" 2>&1 &
        MILVUSCONF=/tmp/milvuscfg/proxy \
            milvus run proxy >"${MILVUS_DATA}/proxy.log" 2>&1 &

        # 4) proxy 헬스체크
        _milvus_ok=0
        for _i in $(seq 1 120); do
            curl -sf http://localhost:9091/healthz >/dev/null 2>&1 && { _milvus_ok=1; break; }
            sleep 3
        done
        if [ "$_milvus_ok" -eq 0 ]; then
            echo "[milvus] coordinator 헬스체크 타임아웃!"
            echo "=== mixcoord.log ===" && cat "${MILVUS_DATA}/mixcoord.log" || true
            echo "=== proxy.log ===" && cat "${MILVUS_DATA}/proxy.log" || true
            sleep infinity
        fi
        echo "[milvus] coordinator 준비 완료 (mixcoord + streamingnode + proxy)"

    # ── Worker (Instant Cluster NODE_RANK>0) ──────────────────────────────────
    else
        # coordinator의 etcd(2379)가 열릴 때까지 대기
        echo "[milvus-worker] coordinator etcd 대기 중 (${_PRIMARY_ADDR}:2379)..."
        until nc -z "${_PRIMARY_ADDR}" 2379 2>/dev/null; do sleep 5; done
        # mixcoord가 etcd에 완전히 등록될 때까지 추가 대기
        sleep 15
        echo "[milvus-worker] coordinator 연결 확인"

        MILVUSCONF=/tmp/milvuscfg/datanode \
            milvus run datanode >"${MILVUS_DATA}/datanode.log" 2>&1 &
        MILVUSCONF=/tmp/milvuscfg/querynode \
            milvus run querynode >"${MILVUS_DATA}/querynode.log" 2>&1 &

        echo "[milvus-worker] datanode + querynode 시작 완료"
        echo "[milvus-worker] coordinator 벤치마크 완료 시 클러스터 종료됨"
        # worker는 벤치마크를 실행하지 않음 — cluster 종료까지 대기
        sleep infinity
    fi

    MILVUS_URI="http://localhost:19530"

else
    # Qdrant 서버 시작 (pod별 독립 스토리지 — RocksDB lock 충돌 방지)
    QDRANT_STORAGE="/workspace/qdrant_storage/${MODEL_SAFE:-default}${MODE_SUFFIX:-}"
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
        cp "$RESULT_FILE" "results/${MODEL_SAFE:-all}${MODE_SUFFIX:-}${DB_SUFFIX}_$(date +%Y%m%d_%H%M%S).json"
        echo "[git] 결과 파일 복사 완료"
        git add results/
        if git diff --cached --quiet; then
            echo "[git] 변경 없음 (이미 push됨)"
        else
            git commit -m "result: ${MODEL_SAFE:-all} 벤치마크 완료"
            git pull --rebase origin main || echo "[git] rebase 실패 — 그냥 push 시도"
            git push && echo "[git] push 완료" || echo "[git] push 실패 — 수동 확인 필요"
        fi
    fi
fi

echo "[완료] 대기 중 (재시작 방지)"
sleep infinity
