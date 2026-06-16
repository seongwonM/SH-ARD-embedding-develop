FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/models \
    HF_DATASETS_CACHE=/workspace/datasets/.cache \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR /app

COPY deploy/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
 && apt-get update && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

COPY bench ./bench

# 시작 시 GitHub에서 최신 코드 pull (pod 재시작만으로 코드 업데이트)
# private repo면 RunPod 환경변수에 GH_TOKEN 설정 필요
CMD ["bash", "-c", \
     "REPO=https://github.com/seongwonM/SH-ARD-embedding-develop.git; \
      TARGET=${GH_TOKEN:+https://${GH_TOKEN}@github.com/seongwonM/SH-ARD-embedding-develop.git}; \
      rm -rf /tmp/latest && git clone --depth=1 ${TARGET:-$REPO} /tmp/latest 2>&1 \
        && cp -rf /tmp/latest/bench /app/ \
        && echo '[startup] 코드 업데이트 완료' \
        || echo '[startup] git clone 실패 — 빌드된 코드로 실행'; \
      python -m bench.runner \
        --model-dtype auto \
        --batch-size 256 \
        --out /workspace/reports \
        --qdrant-path /workspace/qdrant_storage"]
