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
    # Milvus Standalone (DEB 패키지 바이너리) — embedded etcd + local storage
    MILVUS_DATA="/workspace/milvus_data/${MODEL_SAFE:-default}${MODE_SUFFIX:-}"
    mkdir -p "${MILVUS_DATA}/etcd"
    ETCD_USE_EMBED=true \
    ETCD_DATA_DIR="${MILVUS_DATA}/etcd" \
    COMMON_STORAGETYPE=local \
    milvus run standalone >"${MILVUS_DATA}/milvus.log" 2>&1 &
    MILVUS_PID=$!
    echo "[milvus] 서버 시작 (PID=$MILVUS_PID, log=${MILVUS_DATA}/milvus.log), 준비 대기 중..."
    until curl -sf http://localhost:9091/healthz >/dev/null 2>&1; do sleep 2; done
    echo "[milvus] 서버 준비 완료"
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
        cp "$RESULT_FILE" "results/${MODEL_SAFE:-all}${MODE_SUFFIX:-}${DB_SUFFIX}.json"
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
