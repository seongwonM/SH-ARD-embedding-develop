#!/bin/bash
# RunPod 분산처리 테스트 전용 startup.
# startup.sh(standalone 기준값)와 독립적으로 동작.
#
# 환경변수 (RunPod pod 설정에서 지정):
#   NODE_RANK        — 미설정=standalone, 0=coordinator, 1+=worker (RunPod Instant Cluster 자동)
#   PRIMARY_ADDR     — coordinator pod IP (RunPod Instant Cluster 자동)
#   NODE_ADDR        — 현재 pod IP     (RunPod Instant Cluster 자동)
#   MODEL_ID         — BAAI/bge-m3 등
#   VECTOR_MODE      — dense | sparse | colbert (default: dense)
#   BATCH_SIZE       — 인코딩 배치 크기 (default: 16)
#   NUM_SHARDS       — 테스트할 num_shards 목록, 공백 구분 (default: "1")
#   REPLICA_NUMBER   — 테스트할 replica_number 목록 (default: "1")
#   SEARCH_WORKERS   — 동시 검색 스레드 수 목록 (default: "1")
#   WORKER_WAIT_SEC  — coordinator가 worker 준비를 기다리는 초 (default: 30)
#   GH_TOKEN         — GitHub push용 토큰 (선택)
set -euo pipefail

REPO=https://github.com/seongwonM/SH-ARD-embedding-develop.git
if [ -n "${GH_TOKEN:-}" ]; then
    TARGET="https://${GH_TOKEN}@github.com/seongwonM/SH-ARD-embedding-develop.git"
else
    TARGET="$REPO"
fi

if rm -rf /tmp/latest && git clone --depth=1 "$TARGET" /tmp/latest 2>&1; then
    cp -rf /tmp/latest/bench /app/
    echo "[startup] 코드 업데이트 완료"
else
    echo "[startup] git clone 실패 — 빌드된 코드로 실행"
    mkdir -p /tmp/latest
fi

# ── 환경변수 ──────────────────────────────────────────────────────────────────
MODEL_SAFE=$(echo "${MODEL_ID:-}" | tr '/' '_')
MODE_SUFFIX=${VECTOR_MODE:+_${VECTOR_MODE}}
MILVUS_DATA="/workspace/milvus_data/dist_${MODEL_SAFE:-default}${MODE_SUFFIX:-}"

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

mkdir -p "${MILVUS_DATA}/etcd" "${MILVUS_DATA}/woodpecker" "${MILVUS_DATA}/local"
mkdir -p /tmp/milvuscfg_dist

# ── 컴포넌트별 milvus.yaml 생성 ───────────────────────────────────────────────
# standalone / coordinator: woodpecker local (NFS /workspace 공유)
# worker                  : woodpecker local (동일 NFS 경로)
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

etcd_embed  = {'endpoints': 'localhost:2379', 'use': {'embed': True},
               'data': {'dir': f'{milvus_data}/etcd'},
               'config': {'path': '/etc/milvus/configs/embedEtcd.yaml'}}
etcd_local  = {'endpoints': 'localhost:2379',       'use': {'embed': False}}
etcd_remote = {'endpoints': f'{primary_addr}:2379', 'use': {'embed': False}}

if mode == 'standalone':
    components = {'standalone': etcd_embed}
elif mode == 'coordinator':
    components = {
        'mixcoord':      etcd_embed,
        'streamingnode': etcd_local,
        'proxy':         etcd_local,
    }
else:
    components = {
        'datanode':  etcd_remote,
        'querynode': etcd_remote,
    }

for comp, etcd_cfg in components.items():
    cfg = deep_merge(base, {**common, 'etcd': etcd_cfg})
    os.makedirs(f'/tmp/milvuscfg_dist/{comp}', exist_ok=True)
    with open(f'/tmp/milvuscfg_dist/{comp}/milvus.yaml', 'w') as f:
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

# ── Standalone ────────────────────────────────────────────────────────────────
if [ "$_MODE" = "standalone" ]; then
    MILVUSCONF=/tmp/milvuscfg_dist/standalone \
    ETCD_DATA_DIR="${MILVUS_DATA}/etcd" \
    DEPLOY_MODE=STANDALONE \
        milvus run standalone >"${MILVUS_DATA}/milvus.log" 2>&1 &
    MILVUS_PID=$!
    echo "[milvus] standalone 시작 (PID=$MILVUS_PID)..."

    _ok=0
    for _i in $(seq 1 90); do
        if ! kill -0 "$MILVUS_PID" 2>/dev/null; then
            echo "[milvus] 비정상 종료!" && cat "${MILVUS_DATA}/milvus.log" || true
            sleep infinity
        fi
        curl -sf http://localhost:9091/healthz >/dev/null 2>&1 && { _ok=1; break; }
        sleep 2
    done
    [ "$_ok" -eq 0 ] && echo "[milvus] 헬스체크 타임아웃!" && cat "${MILVUS_DATA}/milvus.log" && sleep infinity
    echo "[milvus] standalone 준비 완료"

# ── Coordinator (NODE_RANK=0) ─────────────────────────────────────────────────
elif [ "$_MODE" = "coordinator" ]; then
    MILVUSCONF=/tmp/milvuscfg_dist/mixcoord \
    ETCD_DATA_DIR="${MILVUS_DATA}/etcd" \
        milvus run mixcoord >"${MILVUS_DATA}/mixcoord.log" 2>&1 &

    echo "[milvus] embedded etcd 대기 중..."
    for _i in $(seq 1 60); do nc -z localhost 2379 2>/dev/null && break || sleep 2; done

    MILVUSCONF=/tmp/milvuscfg_dist/streamingnode \
        milvus run streamingnode >"${MILVUS_DATA}/streamingnode.log" 2>&1 &
    MILVUSCONF=/tmp/milvuscfg_dist/proxy \
        milvus run proxy >"${MILVUS_DATA}/proxy.log" 2>&1 &

    _ok=0
    for _i in $(seq 1 120); do
        curl -sf http://localhost:9091/healthz >/dev/null 2>&1 && { _ok=1; break; }
        sleep 3
    done
    if [ "$_ok" -eq 0 ]; then
        echo "[milvus] coordinator 타임아웃!"
        cat "${MILVUS_DATA}/mixcoord.log" || true
        cat "${MILVUS_DATA}/proxy.log"    || true
        sleep infinity
    fi
    echo "[milvus] coordinator 준비 완료"

    # worker 노드가 etcd에 등록될 때까지 대기
    _WORKER_WAIT="${WORKER_WAIT_SEC:-30}"
    echo "[milvus] worker 등록 대기 ${_WORKER_WAIT}초..."
    sleep "${_WORKER_WAIT}"

# ── Worker (NODE_RANK>=1) ──────────────────────────────────────────────────────
else
    echo "[milvus-worker] coordinator etcd 대기 중 (${_PRIMARY_ADDR}:2379)..."
    until nc -z "${_PRIMARY_ADDR}" 2379 2>/dev/null; do sleep 5; done
    sleep 15

    MILVUSCONF=/tmp/milvuscfg_dist/datanode \
        milvus run datanode >"${MILVUS_DATA}/datanode_rank${_NODE_RANK}.log" 2>&1 &
    MILVUSCONF=/tmp/milvuscfg_dist/querynode \
        milvus run querynode >"${MILVUS_DATA}/querynode_rank${_NODE_RANK}.log" 2>&1 &

    echo "[milvus-worker] datanode + querynode 시작 (rank=${_NODE_RANK})"
    echo "[milvus-worker] coordinator 벤치마크 완료 시 종료됨"
    sleep infinity
fi

# ── 데이터 경로 ───────────────────────────────────────────────────────────────
DATA_DIR="/tmp/latest/data"
if [ ! -f "${DATA_DIR}/corpus_all.parquet" ]; then
    if [ -f "/workspace/datasets/corpus_all.parquet" ]; then
        DATA_DIR="/workspace/datasets"
    else
        echo "[startup] ERROR: 데이터 없음"
        exit 1
    fi
fi
echo "[startup] 데이터 경로: ${DATA_DIR}"

# ── 분산 벤치마크 실행 (coordinator / standalone만 도달) ────────────────────────
REPORTS_PATH="/workspace/reports/dist_${MODEL_SAFE:-all}${MODE_SUFFIX:-}"
mkdir -p "${REPORTS_PATH}"

# NUM_SHARDS / REPLICA_NUMBER / SEARCH_WORKERS 는 공백 구분 목록으로 전달
# 예: NUM_SHARDS="1 2" REPLICA_NUMBER="1 2" SEARCH_WORKERS="1 2 4"
_NUM_SHARDS="${NUM_SHARDS:-1}"
_REPLICA_NUMBER="${REPLICA_NUMBER:-1}"
_SEARCH_WORKERS="${SEARCH_WORKERS:-1}"

python -m bench.dist_bench \
    --milvus-uri "http://localhost:19530" \
    --data-root "${DATA_DIR}" \
    --model "${MODEL_ID:-BAAI/bge-m3}" \
    --vector-mode "${VECTOR_MODE:-dense}" \
    --model-dtype auto \
    --batch-size "${BATCH_SIZE:-16}" \
    --num-shards ${_NUM_SHARDS} \
    --replica-number ${_REPLICA_NUMBER} \
    --search-workers ${_SEARCH_WORKERS} \
    --out "${REPORTS_PATH}"

# ── 결과 push ─────────────────────────────────────────────────────────────────
if [ -n "${GH_TOKEN:-}" ]; then
    echo "[git] 결과 push 시작..."
    SUMMARY="${REPORTS_PATH}/dist_summary.json"
    if [ -f "$SUMMARY" ]; then
        cd /tmp/latest
        git config user.email 'pod@runpod.io'
        git config user.name 'RunPod'
        mkdir -p results
        cp "$SUMMARY" "results/dist_${MODEL_SAFE:-all}${MODE_SUFFIX:-}_$(date +%Y%m%d_%H%M%S).json"
        git add results/
        if ! git diff --cached --quiet; then
            git commit -m "result: dist ${MODEL_SAFE:-all} 벤치마크 완료"
            git pull --rebase origin main || true
            git push && echo "[git] push 완료" || echo "[git] push 실패"
        fi
    fi
fi

echo "[완료] 대기 중"
sleep infinity
