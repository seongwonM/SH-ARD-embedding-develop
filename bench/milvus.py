"""Milvus VectorStore 구현 (dense / sparse, colbert 미지원).

MilvusClient URI 형식:
  Standalone : "http://localhost:19530"
  Zilliz Cloud: "https://xxx.api.gcp-us-west1.zillizcloud.com" (token 별도)
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
        kwargs = {"uri": uri}
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
        """data_iter: (point_id, doc_id, vec) 튜플 스트림."""
        print("  [upload] insert 시작...", flush=True)
        batch: list[dict] = []
        for pid, doc_id, vec in data_iter:
            if vector_mode == "sparse":
                vector = {int(k): float(v) for k, v in vec.items()}
            else:
                vector = vec.tolist()
            batch.append({"id": pid, "doc_id": doc_id, "vector": vector})
            if len(batch) >= _INSERT_BATCH:
                self._client.insert(collection_name=name, data=batch)
                batch.clear()
        if batch:
            self._client.insert(collection_name=name, data=batch)
        print("  [upload] insert 완료", flush=True)

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
        CHUNK = 256
        n = len(vectors)
        all_results: list[list[tuple[str, float]]] = []

        metric_type   = "IP" if vector_mode == "sparse" else "COSINE"
        search_params = {} if vector_mode == "sparse" else {"ef": 100}

        for start in range(0, n, CHUNK):
            end = min(start + CHUNK, n)
            if vector_mode == "sparse":
                query_data = [
                    {int(k): float(v) for k, v in vectors[i].items()}
                    for i in range(start, end)
                ]
            else:
                query_data = [vectors[i].tolist() for i in range(start, end)]

            results = self._client.search(
                collection_name=name,
                data=query_data,
                anns_field="vector",
                search_params={"metric_type": metric_type, "params": search_params},
                limit=top_k,
                output_fields=["doc_id"],
            )
            for hits in results:
                all_results.append([(h["entity"]["doc_id"], h["distance"]) for h in hits])

        return all_results

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
