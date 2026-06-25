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

REPO=https://github.com/seongwonM/SH-ARD-embedding-develop.git
if [ -n "${GH_TOKEN:-}" ]; then
    TARGET="https://${GH_TOKEN}@github.com/seongwonM/SH-ARD-embedding-develop.git"
else
    TARGET="$REPO"
fi

# ── 자기 업데이트 ──────────────────────────────────────────────────────────────
# 이미지에 baked-in된 스크립트는 낡을 수 있다.
# git에서 최신 startup_dist.sh를 받아 exec 하면 이미지 캐시와 무관하게 항상 최신 버전 실행.
# _STARTUP_DIST_UPDATED=1 이면 이미 재실행된 것이므로 skip.
if [ "${_STARTUP_DIST_UPDATED:-0}" != "1" ]; then
    if rm -rf /tmp/latest && git clone --depth=1 "$TARGET" /tmp/latest 2>&1; then
        echo "[startup] 코드 수신 완료 — 최신 startup_dist.sh 로 재실행"
        export _STARTUP_DIST_UPDATED=1
        exec bash /tmp/latest/deploy/startup_dist.sh
    else
        echo "[startup] git clone 실패 — 빌드된 코드로 실행"
        mkdir -p /tmp/latest/bench
    fi
    export _STARTUP_DIST_UPDATED=1
fi

set -euo pipefail

# bench/ 업데이트 (자기 업데이트 경로에서는 클론이 이미 완료됨)
if [ -d /tmp/latest/bench ]; then
    cp -rf /tmp/latest/bench /app/
fi

# ── 환경변수 ──────────────────────────────────────────────────────────────────
MODEL_SAFE=$(echo "${MODEL_ID:-}" | tr '/' '_')
MODE_SUFFIX=${VECTOR_MODE:+_${VECTOR_MODE}}
MILVUS_DATA="/workspace/milvus_data/dist_${MODEL_SAFE:-default}${MODE_SUFFIX:-}"

_NODE_RANK="${NODE_RANK:-}"
_PRIMARY_ADDR="${PRIMARY_ADDR:-}"
_PRIMARY_ADDR="${_PRIMARY_ADDR%%/*}"   # CIDR 제거 (10.x.x.x/24 → 10.x.x.x)
_NODE_ADDR="${NODE_ADDR:-}"
_NODE_ADDR="${_NODE_ADDR%%/*}"         # CIDR 제거
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

# ── Milvus config 생성 ────────────────────────────────────────────────────────
python3 - "${MILVUS_DATA}" "${_MODE}" "${_PRIMARY_ADDR}" << 'PYEOF'
import yaml, copy, sys

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

if mode in ('standalone', 'coordinator'):
    etcd_cfg = {'endpoints': 'localhost:2379', 'use': {'embed': True},
                'data': {'dir': f'{milvus_data}/etcd'},
                'config': {'path': '/etc/milvus/configs/embedEtcd.yaml'}}
else:
    etcd_cfg = {'endpoints': f'{primary_addr}:2379', 'use': {'embed': False}}

cfg = deep_merge(base, {**common, 'etcd': etcd_cfg})
with open('/etc/milvus/configs/milvus.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
print(f'[milvus] config 생성 완료 (mode={mode})')
PYEOF

# coordinator: embedded etcd를 외부에서 접근 가능하도록 설정
if [ "$_MODE" = "coordinator" ]; then
    python3 - "${_NODE_ADDR}" << 'PYEOF2'
import yaml, sys
node_addr = sys.argv[1].split('/')[0]
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

# ── Milvus 실행 (최대 3회 재시도 — etcd WAL 부패 시 초기화 후 재시도) ─────────
if [ "$_MODE" = "standalone" ] || [ "$_MODE" = "coordinator" ]; then
    _milvus_started=0
    for _milvus_try in 1 2 3; do
        MILVUSCONF=/etc/milvus/configs \
        ETCD_DATA_DIR="${MILVUS_DATA}/etcd" \
        DEPLOY_MODE=STANDALONE \
            milvus run standalone >"${MILVUS_DATA}/milvus.log" 2>&1 &
        MILVUS_PID=$!
        echo "[milvus] 서버 시작 (시도 $_milvus_try/3, PID=$MILVUS_PID)..."

        _milvus_ok=0
        _milvus_died=0
        for _i in $(seq 1 120); do
            if ! kill -0 "$MILVUS_PID" 2>/dev/null; then
                _milvus_died=1
                break
            fi
            curl -sf http://localhost:9091/healthz >/dev/null 2>&1 && { _milvus_ok=1; break; } || sleep 3
        done

        if [ "$_milvus_ok" -eq 1 ]; then
            _milvus_started=1
            break
        fi

        echo "[milvus] 시도 $_milvus_try 실패 (died=$_milvus_died)"
        tail -50 "${MILVUS_DATA}/milvus.log" || true
        if [ "$_milvus_try" -lt 3 ]; then
            echo "[milvus] etcd WAL 초기화 후 재시도..."
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
    echo "[milvus] 준비 완료 (모드: ${_MODE})"

    if [ "$_MODE" = "coordinator" ]; then
        echo "[milvus] worker QueryNode 등록 대기 ${WORKER_WAIT_SEC:-30}초..."
        sleep "${WORKER_WAIT_SEC:-30}"
    fi

else
    # worker: coordinator healthz 대기
    echo "[worker] coordinator 대기 (http://${_PRIMARY_ADDR}:9091/healthz)..."
    until curl -sf "http://${_PRIMARY_ADDR}:9091/healthz" >/dev/null 2>&1; do sleep 5; done
    sleep 10

    MILVUSCONF=/etc/milvus/configs \
        milvus run querynode >"${MILVUS_DATA}/querynode_rank${_NODE_RANK}.log" 2>&1 &
    MILVUSCONF=/etc/milvus/configs \
        milvus run datanode >"${MILVUS_DATA}/datanode_rank${_NODE_RANK}.log" 2>&1 &

    echo "[worker] querynode + datanode 시작 (rank=${_NODE_RANK})"
    sleep infinity
fi

# ── CUDA 장치 검증 (coordinator / standalone 만 도달) ────────────────────────
# nvidia-smi 목록 중 실제 접근 가능한 GPU만 선별.
# 각 GPU를 격리된 서브프로세스로 검증해 broken GPU 제외.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    _GPU_INDICES=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' \n')
    if [ -n "$_GPU_INDICES" ]; then
        _VALID_GPUS=$(python3 - "${_GPU_INDICES}" << 'PYEOF3'
import subprocess, sys, concurrent.futures, os

def _test(idx):
    try:
        r = subprocess.run(
            ['python3', '-c',
             'import torch,sys; assert torch.cuda.is_available(); '
             'torch.zeros(1, device="cuda:0"); sys.exit(0)'],
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(idx)},
            capture_output=True, timeout=90,
        )
        return str(idx) if r.returncode == 0 else None
    except Exception:
        return None

indices = sys.argv[1].split(',') if sys.argv[1] else []
if not indices:
    print('')
    sys.exit(0)
with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(indices))) as ex:
    results = list(ex.map(_test, indices))
print(','.join(r for r in results if r is not None))
PYEOF3
        )
        if [ -n "$_VALID_GPUS" ]; then
            export CUDA_VISIBLE_DEVICES="$_VALID_GPUS"
            echo "[CUDA] 유효 GPU: ${CUDA_VISIBLE_DEVICES}"
        else
            export CUDA_VISIBLE_DEVICES=""
            echo "[CUDA] 접근 가능한 GPU 없음 — CPU 사용"
        fi
    fi
fi

# ── 데이터 경로 ───────────────────────────────────────────────────────────────
DATA_DIR="/tmp/latest/data"
[ ! -f "${DATA_DIR}/corpus_all.parquet" ] && DATA_DIR="/workspace/datasets"

# ── 분산 벤치마크 (coordinator / standalone 만 도달) ──────────────────────────
REPORTS_PATH="/workspace/reports/dist_${MODEL_SAFE:-all}${MODE_SUFFIX:-}"
mkdir -p "${REPORTS_PATH}"

python -m bench.dist_bench \
    --milvus-uri  "http://localhost:19530" \
    --data-root   "${DATA_DIR}" \
    --model       "${MODEL_ID:-BAAI/bge-m3}" \
    --vector-mode "${VECTOR_MODE:-dense}" \
    --batch-size  "${BATCH_SIZE:-16}" \
    --replicas    ${REPLICAS:-1 2} \
    --workers     ${WORKERS:-1 2 4} \
    --duration    "${SEARCH_DURATION:-30}" \
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
