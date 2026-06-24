"""Milvus VectorStore 구현 (dense / sparse / colbert).

Milvus Standalone 설치: DEB 패키지 (embedded etcd + local storage)
  설치: milvus_2.6.9-1_amd64.deb (v2.6.4+: Array of Structs + MAX_SIM 지원)
  시작: MILVUSCONF=/etc/milvus/configs milvus run standalone
  URI : "http://localhost:19530"

colbert: Array of Structs + MAX_SIM_COSINE (pymilvus >= 2.6.0 필요)
  - 문서당 토큰 임베딩 리스트를 colbert_embs[emb] 필드에 저장
  - 검색: EmbeddingList로 쿼리 토큰 임베딩 전달 → 서버 사이드 MaxSim
"""
from __future__ import annotations

import gc
import math

_ENCODE_CHUNK = 2_000
_INSERT_BATCH = 64


# ── MilvusStore ───────────────────────────────────────────────────────────────

class MilvusStore:

    def __init__(self, uri: str, token: str = "") -> None:
        from pymilvus import MilvusClient
        self._uri = uri
        self._token = token
        kwargs = {"uri": uri, "timeout": 300}
        if token:
            kwargs["token"] = token
        self._client = MilvusClient(**kwargs)
        print(f"[Milvus] 서버: {uri}", flush=True)

    def has_collection(self, name: str) -> bool:
        return self._client.has_collection(name)

    def collection_size(self, name: str) -> int:
        stats = self._client.get_collection_stats(name)
        return int(stats.get("row_count", 0))

    def create_collection(self, name: str, dim: int, vector_mode: str = "dense") -> None:
        from pymilvus import MilvusClient, DataType

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id",     DataType.INT64,  is_primary=True)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=512)

        index_params = MilvusClient.prepare_index_params()

        if vector_mode == "sparse":
            schema.add_field("vector", DataType.SPARSE_FLOAT_VECTOR)
            index_params.add_index(
                field_name="vector",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
            )
        elif vector_mode == "colbert":
            # Array of Structs: 문서당 토큰 임베딩 리스트 → MAX_SIM_COSINE
            struct_schema = self._client.create_struct_field_schema()
            struct_schema.add_field("emb", DataType.FLOAT_VECTOR, dim=dim)
            schema.add_field(
                "colbert_embs",
                DataType.ARRAY,
                element_type=DataType.STRUCT,
                struct_schema=struct_schema,
                max_capacity=512,
            )
            index_params.add_index(
                field_name="colbert_embs[emb]",
                index_type="AUTOINDEX",
                metric_type="MAX_SIM_COSINE",
            )
        else:  # dense
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
            index_params.add_index(
                field_name="vector",
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": 16, "efConstruction": 100},
            )

        self._client.create_collection(
            collection_name=name,
            schema=schema,
            index_params=index_params,
        )
        print(f"  컬렉션 생성: {name}  mode={vector_mode}  dim={dim}", flush=True)

    def upload_stream(self, name: str, data_iter, vector_mode: str = "dense") -> None:
        import time
        # ColBERT: 토큰 임베딩이 크므로 배치 작게, 2000건마다 강제 flush로 서버 OOM 방지
        insert_batch = 16 if vector_mode == "colbert" else _INSERT_BATCH
        flush_every = _ENCODE_CHUNK  # 2000건마다

        print("  [upload] insert 시작...", flush=True)
        batch: list[dict] = []
        total_inserted = 0
        next_flush_at = flush_every

        for pid, doc_id, vec in data_iter:
            if vector_mode == "colbert":
                row = {
                    "id": pid,
                    "doc_id": doc_id,
                    "colbert_embs": [{"emb": v.tolist()} for v in vec],
                }
            elif vector_mode == "sparse":
                row = {
                    "id": pid,
                    "doc_id": doc_id,
                    "vector": {int(k): float(v) for k, v in vec.items()},
                }
            else:
                row = {"id": pid, "doc_id": doc_id, "vector": vec.tolist()}
            batch.append(row)
            if len(batch) >= insert_batch:
                for attempt in range(3):
                    try:
                        self._client.insert(collection_name=name, data=batch)
                        break
                    except Exception:
                        if attempt == 2:
                            raise
                        time.sleep(2 ** attempt)
                total_inserted += len(batch)
                batch.clear()
                # ColBERT는 벡터가 커서 Milvus growing segment가 RAM을 빠르게 소진 → 강제 flush
                if vector_mode == "colbert" and total_inserted >= next_flush_at:
                    print(f"  [upload] flush ({total_inserted:,}건)...", flush=True)
                    self._client.flush(collection_name=name)
                    next_flush_at += flush_every

        if batch:
            self._client.insert(collection_name=name, data=batch)
        print(f"  [upload] insert 완료 ({total_inserted + len(batch):,}건)", flush=True)

    def finalize_index(self, name: str, vector_mode: str = "dense") -> None:
        self._client.load_collection(name)
        print(f"  인덱스 로드 완료 ({vector_mode})", flush=True)

    def search_batch(
        self,
        name:        str,
        vectors,
        top_k:       int,
        vector_mode: str = "dense",
    ) -> list[list[tuple[str, float]]]:
        import time
        CHUNK = 16 if vector_mode == "colbert" else 256
        n = len(vectors)
        all_results: list[list[tuple[str, float]]] = []
        n_batches = (n + CHUNK - 1) // CHUNK
        # colbert는 브루트포스라 배치당 시간이 길어서 매 배치 로깅
        log_every = 1 if vector_mode == "colbert" else max(1, n_batches // 10)
        t0 = time.time()

        for start in range(0, n, CHUNK):
            end = min(start + CHUNK, n)
            batch_idx = start // CHUNK

            if vector_mode == "colbert":
                from pymilvus.client.embedding_list import EmbeddingList
                data = []
                for i in range(start, end):
                    emb_list = EmbeddingList()
                    for token_vec in vectors[i]:
                        emb_list.add(token_vec.tolist())
                    data.append(emb_list)
                print(f"  검색 배치 {batch_idx+1}/{n_batches} gRPC 전송 중...", flush=True)
                results = self._client.search(
                    collection_name=name,
                    data=data,
                    anns_field="colbert_embs[emb]",
                    search_params={"metric_type": "MAX_SIM_COSINE"},
                    limit=top_k,
                    output_fields=["doc_id"],
                    timeout=120,
                )
            elif vector_mode == "sparse":
                query_data = [
                    {int(k): float(v) for k, v in vectors[i].items()}
                    for i in range(start, end)
                ]
                results = self._client.search(
                    collection_name=name,
                    data=query_data,
                    anns_field="vector",
                    search_params={"metric_type": "IP", "params": {}},
                    limit=top_k,
                    output_fields=["doc_id"],
                )
            else:  # dense
                query_data = [vectors[i].tolist() for i in range(start, end)]
                results = self._client.search(
                    collection_name=name,
                    data=query_data,
                    anns_field="vector",
                    search_params={"metric_type": "COSINE", "params": {"ef": 100}},
                    limit=top_k,
                    output_fields=["doc_id"],
                )

            for hits in results:
                all_results.append([(h["entity"]["doc_id"], h["distance"]) for h in hits])

            if (batch_idx + 1) % log_every == 0 or end == n:
                elapsed = time.time() - t0
                qps = end / elapsed if elapsed > 0 else 0
                print(f"  검색 진행: {end:,}/{n:,}  ({elapsed:.0f}s, {qps:.1f} q/s)", flush=True)

        return all_results

    def search_concurrent(
        self,
        name: str,
        vectors,
        top_k: int,
        vector_mode: str = "dense",
        concurrency: int = 1,
        duration_sec: int = 30,
    ) -> dict:
        """VDBBench 방식 concurrent QPS 측정. 각 스레드가 독립 커넥션으로 랜덤 쿼리 전송."""
        import threading, time, random
        from pymilvus import MilvusClient

        if vector_mode != "dense":
            return {"qps": None, "p50_ms": None, "p95_ms": None, "p99_ms": None, "total": 0,
                    "note": f"{vector_mode} concurrent 미지원"}

        latencies: list[float] = []
        lock = threading.Lock()
        stop_event = threading.Event()
        n = len(vectors)

        def _worker():
            kwargs = {"uri": self._uri, "timeout": 60}
            if self._token:
                kwargs["token"] = self._token
            client = MilvusClient(**kwargs)
            while not stop_event.is_set():
                idx = random.randint(0, n - 1)
                t0 = time.perf_counter()
                client.search(
                    collection_name=name,
                    data=[vectors[idx].tolist()],
                    anns_field="vector",
                    search_params={"metric_type": "COSINE", "params": {"ef": 100}},
                    limit=top_k,
                    output_fields=["doc_id"],
                )
                lat = time.perf_counter() - t0
                with lock:
                    latencies.append(lat)

        threads = [threading.Thread(target=_worker, daemon=True) for _ in range(concurrency)]
        for t in threads:
            t.start()
        time.sleep(duration_sec)
        stop_event.set()
        for t in threads:
            t.join(timeout=10)

        if not latencies:
            return {"qps": 0, "p50_ms": 0, "p95_ms": 0, "p99_ms": 0, "total": 0}

        latencies.sort()
        m = len(latencies)
        return {
            "qps":    round(m / duration_sec, 1),
            "p50_ms": round(latencies[int(m * 0.50)] * 1000, 1),
            "p95_ms": round(latencies[int(m * 0.95)] * 1000, 1),
            "p99_ms": round(latencies[int(m * 0.99)] * 1000, 1),
            "total":  m,
        }

    def drop_collection(self, name: str) -> None:
        self._client.drop_collection(name)


# ── 인덱싱 오케스트레이터 (Milvus 전용) ──────────────────────────────────────

def index_docs(
    store:      MilvusStore,
    name:       str,
    model,
    docs:       dict,
    batch_size: int,
) -> None:
    try:
        import torch
        _has_cuda = torch.cuda.is_available()
    except ImportError:
        _has_cuda = False

    vector_mode = getattr(model, "vector_mode", "dense")
    doc_ids     = list(docs.keys())
    n_total     = len(doc_ids)
    n_chunks    = math.ceil(n_total / _ENCODE_CHUNK)
    _log_every  = max(1, n_chunks // 10)

    store.create_collection(name, model.dim, vector_mode=vector_mode)

    def _data_generator():
        point_id = 0
        for ci in range(n_chunks):
            s = ci * _ENCODE_CHUNK
            e = min(s + _ENCODE_CHUNK, n_total)
            chunk_ids = doc_ids[s:e]
            texts = [
                f"{docs[d].get('title', '')} {docs[d]['chunk']}".strip()
                for d in chunk_ids
            ]

            embs = model.encode_docs(texts, batch_size)
            del texts

            for doc_id, vec in zip(chunk_ids, embs):
                yield point_id, doc_id, vec
                point_id += 1

            del embs
            if _has_cuda:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            gc.collect()

            if (ci + 1) % _log_every == 0 or ci + 1 == n_chunks:
                print(f"  인코딩 {e:,}/{n_total:,}", flush=True)

    store.upload_stream(name, _data_generator(), vector_mode=vector_mode)
    store.finalize_index(name, vector_mode=vector_mode)
    print(f"  인덱싱 완료: {n_total:,}건", flush=True)
