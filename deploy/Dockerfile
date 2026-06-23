FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/models \
    HF_DATASETS_CACHE=/workspace/datasets/.cache \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    DATA_ROOT=/workspace/datasets \
    MODEL_ID="" \
    VECTOR_MODE="" \
    VECTOR_DB="qdrant" \
    BATCH_SIZE=16

WORKDIR /app

COPY deploy/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir "flash-attn==2.7.4.post1" --no-build-isolation \
 && apt-get update \
 && apt-get install -y --no-install-recommends git curl \
 && curl -L https://github.com/qdrant/qdrant/releases/download/v1.18.0/qdrant-x86_64-unknown-linux-musl.tar.gz \
    | tar xz -C /usr/local/bin \
 && curl -L https://github.com/milvus-io/milvus/releases/download/v2.6.9/milvus_2.6.9-1_amd64.deb \
    -o /tmp/milvus.deb \
 && apt-get install -y /tmp/milvus.deb \
 && rm /tmp/milvus.deb \
 && rm -rf /var/lib/apt/lists/*

COPY bench ./bench
COPY deploy/startup.sh /startup.sh
COPY deploy/startup_dist.sh /startup_dist.sh
RUN chmod +x /startup.sh /startup_dist.sh

# RunPod 환경변수로 pod별 설정 지정:
#   MODEL_ID, VECTOR_MODE, VECTOR_DB, BATCH_SIZE
#   분산 테스트용 (startup_dist.sh): REPLICAS, WORKERS, WORKER_WAIT_SEC, BASELINE_JSON
#   Instant Cluster: NODE_RANK, PRIMARY_ADDR, NODE_ADDR (RunPod 자동 주입)
CMD ["/bin/bash", "/startup.sh"]
