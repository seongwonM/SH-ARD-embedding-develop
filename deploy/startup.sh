#!/bin/bash
set -euo pipefail

REPO=https://github.com/seongwonM/SH-ARD-embedding-develop.git
if [ -n "${GH_TOKEN:-}" ]; then
    TARGET="https://${GH_TOKEN}@github.com/seongwonM/SH-ARD-embedding-develop.git"
else
    TARGET="$REPO"
fi

for i in 1 2 3 4 5; do
    rm -rf /tmp/latest && git clone --depth=1 "$TARGET" /tmp/latest 2>&1 && break
    echo "[startup] git clone 실패 (시도 $i/5) — 10초 후 재시도..."
    sleep 10
done

if [ -d /tmp/latest ]; then
    cp -rf /tmp/latest/bench /app/
    echo "[startup] 코드 업데이트 완료"
else
    echo "[startup] git clone 최종 실패 — 빌드된 코드로 실행"
fi

MODEL_SAFE=$(echo "${MODEL_ID:-}" | tr '/' '_')
MODE_SUFFIX=${VECTOR_MODE:+_${VECTOR_MODE}}
QDRANT_PATH="/workspace/qdrant_storage/${MODEL_SAFE:-all}${MODE_SUFFIX:-}"
REPORTS_PATH="/workspace/reports/${MODEL_SAFE:-all}${MODE_SUFFIX:-}"

MODEL_ARG=""
[ -n "${MODEL_ID:-}" ] && MODEL_ARG="--model ${MODEL_ID}"

MODE_ARG=""
[ -n "${VECTOR_MODE:-}" ] && MODE_ARG="--vector-mode ${VECTOR_MODE}"

python -m bench.runner \
    ${MODEL_ARG} \
    ${MODE_ARG} \
    --model-dtype auto \
    --batch-size "${BATCH_SIZE:-16}" \
    --out "${REPORTS_PATH}" \
    --qdrant-path "${QDRANT_PATH}" \
    --data-root /tmp/latest/data

if [ -n "${GH_TOKEN:-}" ]; then
    cd /tmp/latest
    git config user.email 'pod@runpod.io'
    git config user.name 'RunPod'
    mkdir -p results
    cp "${REPORTS_PATH}/summary.json" "results/${MODEL_SAFE:-all}${MODE_SUFFIX:-}.json"
    git add results/
    git diff --cached --quiet || git commit -m "result: ${MODEL_SAFE:-all} 벤치마크 완료"
    git push
    echo "[git] 결과 push 완료"
fi

echo "[완료] 대기 중 (재시작 방지)"
sleep infinity
