#!/bin/bash
set -euo pipefail

BRANCH=${BRANCH:-test}
REPO=https://github.com/seongwonM/SH-ARD-embedding-develop.git
if [ -n "${GH_TOKEN:-}" ]; then
    TARGET="https://${GH_TOKEN}@github.com/seongwonM/SH-ARD-embedding-develop.git"
else
    TARGET="$REPO"
fi

if rm -rf /tmp/latest && git clone --depth=1 --branch "$BRANCH" "$TARGET" /tmp/latest 2>&1; then
    cp -rf /tmp/latest/bench /app/
    echo "[startup] 코드 업데이트 완료 (branch=$BRANCH)"
else
    echo "[startup] git clone 실패 — 빌드된 코드로 실행"
    mkdir -p /tmp/latest
fi

MODEL_SAFE=$(echo "${MODEL_ID:-}" | tr '/' '_')
REPORTS_PATH="/workspace/reports/${MODEL_SAFE:-all}"

# Qdrant 서버 시작
mkdir -p /workspace/qdrant_storage
QDRANT__STORAGE__STORAGE_PATH=/workspace/qdrant_storage \
QDRANT__LOG_LEVEL=WARN \
qdrant --disable-telemetry &
QDRANT_PID=$!
echo "[qdrant] 서버 시작 (PID=$QDRANT_PID), 준비 대기 중..."
until curl -sf http://localhost:6333/healthz >/dev/null 2>&1; do sleep 1; done
echo "[qdrant] 서버 준비 완료"

MODEL_ARG=""
[ -n "${MODEL_ID:-}" ] && MODEL_ARG="--model ${MODEL_ID}"

python -m bench.runner \
    ${MODEL_ARG} \
    --model-dtype auto \
    --batch-size "${BATCH_SIZE:-16}" \
    --out "${REPORTS_PATH}" \
    --qdrant-url http://localhost:6333 \
    --data-root /tmp/latest/data

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
        cp "$RESULT_FILE" "results/${MODEL_SAFE:-all}.json"
        git add results/
        if git diff --cached --quiet; then
            echo "[git] 변경 없음 (이미 push됨)"
        else
            git commit -m "result: ${MODEL_SAFE:-all} 벤치마크 완료"
            git pull --rebase origin "$BRANCH" || echo "[git] rebase 실패 — 그냥 push 시도"
            git push && echo "[git] push 완료" || echo "[git] push 실패 — 수동 확인 필요"
        fi
    fi
fi

echo "[완료] 대기 중 (재시작 방지)"
sleep infinity
