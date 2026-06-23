#!/bin/bash
# RunPod 분산처리 테스트 startup.
# startup.sh(standalone baseline)와 독립적으로 동작.
#
# 환경변수:
#   NODE_RANK        — 미설정=standalone, 0=coordinator, 1+=worker (RunPod Instant Cluster 자동)
#   PRIMARY_ADDR     — coordinator pod IP (RunPod Instant Cluster 자동)
#   NODE_ADDR        — 현재 pod IP (RunPod Instant Cluster 자동)
#   MODEL_ID         — BAAI/bge-m3 등
#   VECTOR_MODE      — dense|sparse|colbert (default: dense)
#   BATCH_SIZE       — 인코딩 배치 크기 (default: 16)
#   REPLICAS         — 테스트할 replica_number 목록, 공백 구분 (예: "1 2")
#   WORKERS          — 동시 검색 스레드 수 목록 (예: "1 2 4")
#   WORKER_WAIT_SEC  — coordinator가 worker 등록을 기다리는 초 (default: 30)
#   BASELINE_JSON    — baseline summary.json 경로 (비교 출력용, 선택)
#   GH_TOKEN         — GitHub push 토큰 (선택)
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
_PRIMARY_ADDR="${PRIMARY_ADDR%%/*}"    # CIDR 제거 (10.x.x.x/24 → 10.x.x.x)
_NODE_ADDR="${NODE_ADDR%%/*}"          # CIDR 제거
_NODE_ADDR="${_NODE_ADDR:-localhost}"

if [ -z "$_NODE_RANK" ]; then
    _MODE="standalone"
elif [ "$_NODE_RANK" = "0" ]; then
    _MODE="coordinator"
else
    _MODE="worker"
fi
echo "[milvus] 모드: $_MODE (NODE_RANK=${_NODE_RANK:-unset})"

mkdir -p "${MILVUS_DATA}/etcd" "${MILVUS_DATA}/woodpecker" "${MILVUS_DATA}/local"
mkdir -p /tmp/milvuscfg_dist

# ── 컴포넌트별 config 생성 ─────────────────────────────────────────────────────
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
    'rootCoord': {'port': 53100}, 'dataCoord': {'port': 13333},
    'queryCoord': {'port': 19531}, 'queryNode': {'port': 21123},
    'dataNode': {'port': 21124},
    'proxy': {'port': 19530, 'internalPort': 19529,
              'grpc': {'serverMaxRecvSize': 268435456, 'serverMaxSendSize': 268435456,
                       'clientMaxRecvSize': 268435456, 'clientMaxSendSize': 268435456}},
}

etcd_embed  = {'endpoints': 'localhost:2379', 'use': {'embed': True},
               'data': {'dir': f'{milvus_data}/etcd'},
               'config': {'path': '/etc/milvus/configs/embedEtcd.yaml'}}
etcd_local  = {'endpoints': 'localhost:2379',       'use': {'embed': False}}
etcd_remote = {'endpoints': f'{primary_addr}:2379', 'use': {'embed': False}}

if mode in ('standalone', 'coordinator'):
    # coordinator도 milvus run standalone으로 실행 — embedded etcd는 standalone만 지원
    # 워커 pod들이 추가 querynode/datanode를 등록하면 QueryCoord가 자동으로 활용
    components = {'standalone': etcd_embed}
else:
    components = {'datanode': etcd_remote, 'querynode': etcd_remote}

for comp, etcd_cfg in components.items():
    cfg = deep_merge(base, {**common, 'etcd': etcd_cfg})
    os.makedirs(f'/tmp/milvuscfg_dist/{comp}', exist_ok=True)
    with open(f'/tmp/milvuscfg_dist/{comp}/milvus.yaml', 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    print(f'[milvus] {comp} config 생성')
PYEOF

if [ "$_MODE" = "coordinator" ]; then
    python3 - "${_NODE_ADDR}" << 'PYEOF2'
import yaml, sys
node_addr = sys.argv[1].split('/')[0]  # CIDR 표기 제거 (e.g. 10.65.0.2/24 → 10.65.0.2)
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
print(f'[milvus] embedEtcd 외부 접근 활성화 ({node_addr}:2379)')
PYEOF2
fi

# ── Milvus 실행 ───────────────────────────────────────────────────────────────
if [ "$_MODE" = "standalone" ] || [ "$_MODE" = "coordinator" ]; then
    MILVUSCONF=/tmp/milvuscfg_dist/standalone \
    ETCD_DATA_DIR="${MILVUS_DATA}/etcd" \
    DEPLOY_MODE=STANDALONE \
        milvus run standalone >"${MILVUS_DATA}/milvus.log" 2>&1 &
    MILVUS_PID=$!

    for _i in $(seq 1 120); do
        if ! kill -0 "$MILVUS_PID" 2>/dev/null; then
            echo "[ERROR] milvus 프로세스 종료됨. 로그:"
            cat "${MILVUS_DATA}/milvus.log"
            sleep infinity
        fi
        curl -sf http://localhost:9091/healthz >/dev/null 2>&1 && break || sleep 3
    done
    echo "[milvus] 준비 완료 (모드: ${_MODE})"

    if [ "$_MODE" = "coordinator" ]; then
        echo "[milvus] worker QueryNode 등록 대기 ${WORKER_WAIT_SEC:-30}초..."
        sleep "${WORKER_WAIT_SEC:-30}"
    fi

else
    # worker: coordinator healthz 대기 (nc 대신 curl 사용)
    echo "[worker] coordinator 대기 (http://${_PRIMARY_ADDR}:9091/healthz)..."
    until curl -sf "http://${_PRIMARY_ADDR}:9091/healthz" >/dev/null 2>&1; do sleep 5; done
    sleep 10

    MILVUSCONF=/tmp/milvuscfg_dist/querynode \
        milvus run querynode >"${MILVUS_DATA}/querynode_rank${_NODE_RANK}.log" 2>&1 &
    MILVUSCONF=/tmp/milvuscfg_dist/datanode \
        milvus run datanode >"${MILVUS_DATA}/datanode_rank${_NODE_RANK}.log" 2>&1 &

    echo "[worker] querynode + datanode 시작 (rank=${_NODE_RANK})"
    sleep infinity
fi

# ── 데이터 경로 ───────────────────────────────────────────────────────────────
DATA_DIR="/tmp/latest/data"
[ ! -f "${DATA_DIR}/corpus_all.parquet" ] && DATA_DIR="/workspace/datasets"

# ── 분산 벤치마크 (coordinator / standalone 만 도달) ──────────────────────────
REPORTS_PATH="/workspace/reports/dist_${MODEL_SAFE:-all}${MODE_SUFFIX:-}"
mkdir -p "${REPORTS_PATH}"

python -m bench.dist_bench \
    --milvus-uri "http://localhost:19530" \
    --data-root  "${DATA_DIR}" \
    --model      "${MODEL_ID:-BAAI/bge-m3}" \
    --vector-mode "${VECTOR_MODE:-dense}" \
    --batch-size  "${BATCH_SIZE:-16}" \
    --replicas    ${REPLICAS:-1 2} \
    --workers     ${WORKERS:-1 2 4} \
    ${BASELINE_JSON:+--baseline "${BASELINE_JSON}"} \
    --out "${REPORTS_PATH}"

# ── 결과 push ─────────────────────────────────────────────────────────────────
if [ -n "${GH_TOKEN:-}" ] && [ -f "${REPORTS_PATH}/dist_summary.json" ]; then
    cd /tmp/latest
    git config user.email 'pod@runpod.io'
    git config user.name 'RunPod'
    mkdir -p results
    cp "${REPORTS_PATH}/dist_summary.json" \
       "results/dist_${MODEL_SAFE:-all}${MODE_SUFFIX:-}_$(date +%Y%m%d_%H%M%S).json"
    git add results/
    if ! git diff --cached --quiet; then
        git commit -m "result: dist ${MODEL_SAFE:-all} 벤치마크 완료"
        git pull --rebase origin main || true
        git push && echo "[git] push 완료" || echo "[git] push 실패"
    fi
fi

echo "[완료]"
sleep infinity
