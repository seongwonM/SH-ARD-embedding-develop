FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/models \
    HF_DATASETS_CACHE=/workspace/datasets/.cache \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    DATA_ROOT=/workspace/datasets \
    MODEL_ID="" \
    BATCH_SIZE=16 \
    VECTOR_MODE=dense \
    BRANCH=test

WORKDIR /app

COPY deploy/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir "flash-attn==2.7.4.post1" --no-build-isolation \
 && apt-get update && apt-get install -y --no-install-recommends git curl \
 && curl -L https://github.com/qdrant/qdrant/releases/download/v1.13.6/qdrant-x86_64-unknown-linux-musl.tar.gz \
    | tar xz -C /usr/local/bin \
 && rm -rf /var/lib/apt/lists/*

COPY bench ./bench
COPY deploy/startup.sh ./startup.sh
RUN chmod +x ./startup.sh

# 시작 순서:
#   1. GitHub test 브랜치에서 최신 bench/ 코드 pull (pod 재시작만으로 코드 업데이트)
#   2. Qdrant Rust 서버 백그라운드 실행 + healthz 폴링
#   3. MODEL_ID 환경변수가 있으면 단일 모델만 실행, 없으면 전체 실행
#      → 4-pod 병렬 시: 각 pod에 MODEL_ID 환경변수로 모델 지정
#         pod1: MODEL_ID=BAAI/bge-m3
#         pod2: MODEL_ID=Qwen/Qwen3-Embedding-0.6B
#         pod3: MODEL_ID=Qwen/Qwen3-Embedding-4B
#         pod4: MODEL_ID=Qwen/Qwen3-Embedding-8B
CMD ["./startup.sh"]
