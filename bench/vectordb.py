"""
VectorDB 추상화 레이어 (Qdrant Rust 서버 전용).

vector_mode별 구성:
  dense  : VectorParams cosine, flat exact (m=0)
  sparse : SparseVectorParams — Rust 네이티브 inverted index
  colbert: MultiVectorConfig(MAX_SIM) — Rust 네이티브 multivector
"""
from __future__ import annotations

import gc
import math

import numpy as np

_ENCODE_CHUNK = 2_000   # 인코딩 단위 (메모리 제어)
_UPLOAD_BATCH = 256     # Qdrant upload_points 내부 배치 크기


# ── 인터페이스 ────────────────────────────────────────────────────────────────

class VectorStore:
    def has_collection(self, name: str) -> bool:                    raise NotImplementedError
    def collection_size(self, name: str) -> int:                    raise NotImplementedError
    def create_collection(self, name: str, dim: int, vector_mode: str = "dense") -> None: raise NotImplementedError
    def upload_stream(self, name: str, points, vector_mode: str = "dense") -> None: raise NotImplementedError
    def finalize_index(self, name: str, vector_mode: str = "dense") -> None:  raise NotImplementedError
    def search_batch(self, name: str, vectors, top_k: int, vector_mode: str = "dense") -> list[list[tuple[str, float]]]: raise NotImplementedError
    def drop_collection(self, name: str) -> None:                   raise NotImplementedError


# ── Qdrant 구현 ───────────────────────────────────────────────────────────────

class QdrantStore(VectorStore):
    """Qdrant Rust 서버 기반. dense/sparse/colbert 모두 서버 네이티브 API."""

    def __init__(self, url: str) -> None:
        from qdrant_client import QdrantClient
        self._client = QdrantClient(url=url, timeout=300)
        print(f"[Qdrant] 서버: {url}", flush=True)

    # ── 컬렉션 존재 / 크기 ────────────────────────────────────────────────────

    def has_collection(self, name: str) -> bool:
        return name in {c.name for c in self._client.get_collections().collections}

    def collection_size(self, name: str) -> int:
        return self._client.get_collection(name).points_count

    # ── 컬렉션 생성 ───────────────────────────────────────────────────────────

    def create_collection(self, name: str, dim: int, vector_mode: str = "dense") -> None:
        from qdrant_client.models import (
            Distance, VectorParams, HnswConfigDiff,
            SparseVectorParams, SparseIndexParams,
            MultiVectorConfig, MultiVectorComparator,
        )

        if vector_mode == "sparse":
            self._client.create_collection(
                collection_name=name,
                vectors_config={},
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(on_disk=False, full_scan_threshold=20_000),
                    )
                },
                on_disk_payload=False,
            )
        elif vector_mode == "colbert":
            self._client.create_collection(
                collection_name=name,
                vectors_config={
                    "colbert": VectorParams(
                        size=dim,
                        distance=Distance.COSINE,
                        multivector_config=MultiVectorConfig(
                            comparator=MultiVectorComparator.MAX_SIM,
                        ),
                        on_disk=False,
                    )
                },
                hnsw_config=HnswConfigDiff(m=0),
                on_disk_payload=False,
            )
        else:  # dense
            self._client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=dim, distance=Distance.COSINE, on_disk=False,
                ),
                hnsw_config=HnswConfigDiff(m=0),
                on_disk_payload=False,
            )
        print(f"  Qdrant 컬렉션 생성: {name}  mode={vector_mode}  dim={dim}", flush=True)

    # ── 업로드 ────────────────────────────────────────────────────────────────

    def upload_stream(self, name: str, points, vector_mode: str = "dense") -> None:
        print("  [upload] upload_points 시작...", flush=True)
        self._client.upload_points(
            collection_name=name,
            points=points,
            batch_size=_UPLOAD_BATCH,
            parallel=1,
            max_retries=3,
        )
        print("  [upload] upload_points 완료", flush=True)

    # ── 인덱스 마무리 ─────────────────────────────────────────────────────────

    def finalize_index(self, name: str, vector_mode: str = "dense") -> None:
        labels = {"dense": "flat cosine", "sparse": "sparse (native)", "colbert": "MaxSim (native)"}
        print(f"  인덱스 완료 ({labels.get(vector_mode, vector_mode)})", flush=True)

    # ── 검색 ─────────────────────────────────────────────────────────────────

    def search_batch(
        self,
        name:        str,
        vectors,
        top_k:       int,
        vector_mode: str = "dense",
    ) -> list[list[tuple[str, float]]]:
        from qdrant_client.models import QueryRequest, SparseVector

        CHUNK = 256
        n = len(vectors)
        all_results: list[list[tuple[str, float]]] = []

        for start in range(0, n, CHUNK):
            end = min(start + CHUNK, n)

            if vector_mode == "sparse":
                # vectors: list[dict[str|int, float]]  (lexical_weights)
                requests = [
                    QueryRequest(
                        query=SparseVector(
                            indices=[int(k) for k in vectors[i].keys()],
                            values=list(vectors[i].values()),
                        ),
                        limit=top_k,
                        with_payload=True,
                        using="sparse",
                    )
                    for i in range(start, end)
                ]
            elif vector_mode == "colbert":
                # vectors: list[np.ndarray [n_tokens, dim]]
                requests = [
                    QueryRequest(
                        query=vectors[i].tolist(),
                        limit=top_k,
                        with_payload=True,
                        using="colbert",
                    )
                    for i in range(start, end)
                ]
            else:  # dense
                requests = [
                    QueryRequest(
                        query=vectors[i].tolist(),
                        limit=top_k,
                        with_payload=True,
                    )
                    for i in range(start, end)
                ]

            batch = self._client.query_batch_points(
                collection_name=name,
                requests=requests,
            )
            for result in batch:
                all_results.append([(r.payload["doc_id"], r.score) for r in result.points])

        return all_results

    # ── 삭제 ─────────────────────────────────────────────────────────────────

    def drop_collection(self, name: str) -> None:
        self._client.delete_collection(name)


# ── PointStruct 팩토리 ────────────────────────────────────────────────────────

def _make_point(point_id: int, doc_id: str, vec, vector_mode: str):
    from qdrant_client.models import PointStruct, SparseVector

    if vector_mode == "sparse":
        return PointStruct(
            id=point_id,
            vector={"sparse": SparseVector(
                indices=[int(k) for k in vec],
                values=list(vec.values()),
            )},
            payload={"doc_id": doc_id},
        )
    if vector_mode == "colbert":
        return PointStruct(
            id=point_id,
            vector={"colbert": vec.astype(np.float32).tolist()},
            payload={"doc_id": doc_id},
        )
    return PointStruct(
        id=point_id,
        vector=vec.tolist(),
        payload={"doc_id": doc_id},
    )


# ── 인덱싱 오케스트레이터 ─────────────────────────────────────────────────────

def index_docs(
    store:      VectorStore,
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

    def _point_generator():
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
                yield _make_point(point_id, doc_id, vec, vector_mode)
                point_id += 1

            del embs
            if _has_cuda:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            gc.collect()

            if (ci + 1) % _log_every == 0 or ci + 1 == n_chunks:
                print(f"  인코딩 {e:,}/{n_total:,}", flush=True)

    store.upload_stream(name, _point_generator(), vector_mode=vector_mode)
    store.finalize_index(name, vector_mode=vector_mode)
    print(f"  인덱싱 완료: {n_total:,}건", flush=True)
