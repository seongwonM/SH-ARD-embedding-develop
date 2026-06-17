FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/models \
    HF_DATASETS_CACHE=/workspace/datasets/.cache \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    DATA_ROOT=/workspace/datasets \
    MODEL_ID="" \
    BATCH_SIZE=16

WORKDIR /app

COPY deploy/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir flash-attn --no-build-isolation \
 && apt-get update && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

COPY bench ./bench

# 시작 순서:
#   1. GitHub에서 최신 bench/ 코드 pull (pod 재시작만으로 코드 업데이트)
#   2. PVC에 데이터 없으면 data_prep.py 실행 (있으면 스킵 — 멀티 pod 공유 가능)
#   3. MODEL_ID 환경변수가 있으면 단일 모델만 실행, 없으면 전체 실행
#      → 4-pod 병렬 시: 각 pod에 MODEL_ID 환경변수로 모델 지정
#         pod1: MODEL_ID=BAAI/bge-m3
#         pod2: MODEL_ID=Qwen/Qwen3-Embedding-0.6B
#         pod3: MODEL_ID=Qwen/Qwen3-Embedding-4B
#         pod4: MODEL_ID=Qwen/Qwen3-Embedding-8B
CMD ["bash", "-c", "\
  REPO=https://github.com/seongwonM/SH-ARD-embedding-develop.git; \
  TARGET=${GH_TOKEN:+https://${GH_TOKEN}@github.com/seongwonM/SH-ARD-embedding-develop.git}; \
  rm -rf /tmp/latest && git clone --depth=1 ${TARGET:-$REPO} /tmp/latest 2>&1 \
    && cp -rf /tmp/latest/bench /app/ \
    && echo '[startup] 코드 업데이트 완료' \
    || echo '[startup] git clone 실패 — 빌드된 코드로 실행'; \
  MODEL_SAFE=$(echo ${MODEL_ID} | tr '/' '_'); \
  QDRANT_PATH=/workspace/qdrant_storage/${MODEL_SAFE:-all}; \
  REPORTS_PATH=/workspace/reports/${MODEL_SAFE:-all}; \
  MODEL_ARG=${MODEL_ID:+--model ${MODEL_ID}}; \
  python -m bench.runner \
    ${MODEL_ARG} \
    --model-dtype auto \
    --batch-size ${BATCH_SIZE} \
    --out ${REPORTS_PATH} \
    --qdrant-path ${QDRANT_PATH} \
    --data-root /tmp/latest/data \
  && if [ -n \"${GH_TOKEN}\" ]; then \
       cd /tmp/latest \
    && git config user.email 'pod@runpod.io' \
    && git config user.name 'RunPod' \
    && mkdir -p results \
    && cp ${REPORTS_PATH}/summary.json results/${MODEL_SAFE}.json \
    && git add results/ \
    && git diff --cached --quiet || git commit -m \"result: ${MODEL_SAFE} 벤치마크 완료\" \
    && git push \
    && echo '[git] 결과 push 완료'; \
  fi; \
  echo '[완료] 대기 중 (재시작 방지)'; \
  sleep infinity"]
# 커스텀 데이터셋 사용 예:
#   --datasets my_csv_dataset another_jsonl_dataset
# 각 데이터셋 디렉터리에 docs.*/queries.*/qrels.* 파일만 있으면 됨
