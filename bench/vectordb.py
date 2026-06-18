"""
VectorDB 추상화 레이어.

vector_mode별 Qdrant 컬렉션 구성:
  dense  : VectorParams (cosine) — HNSW
  sparse : SparseVectorParams    — inverted index
  colbert: VectorParams + MultiVectorConfig(MAX_SIM) — HNSW multi-vector

인덱싱 패턴 (Qdrant 공식 권장):
  - encode는 _ENCODE_CHUNK 단위로 chunking (OOM 방지)
  - upsert는 upload_points(generator) 로 streaming (자동 배치/재시도)
  - bulk insert 중 HNSW m=0, 완료 후 m=32/ef=256 (Qdrant 공식 권장)
"""
from __future__ import annotations

import gc
import math
from typing import Generator

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
    """Qdrant on-disk 또는 원격 서버 VectorDB."""

    def __init__(self, *, path: str | None = None, url: str | None = None) -> None:
        from qdrant_client import QdrantClient
        if url:
            self._client = QdrantClient(url=url)
            print(f"[Qdrant] 원격 서버: {url}", flush=True)
        else:
            self._client = QdrantClient(path=path)
            print(f"[Qdrant] on-disk: {path}", flush=True)

    def has_collection(self, name: str) -> bool:
        return name in {c.name for c in self._client.get_collections().collections}

    def collection_size(self, name: str) -> int:
        return self._client.get_collection(name).points_count

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
                        index=SparseIndexParams(on_disk=False, full_scan_threshold=5000),
                    )
                },
                on_disk_payload=False,
            )
        elif vector_mode == "colbert":
            self._client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=dim,
                    distance=Distance.COSINE,
                    multivector_config=MultiVectorConfig(
                        comparator=MultiVectorComparator.MAX_SIM,
                    ),
                    on_disk=True,
                ),
                hnsw_config=HnswConfigDiff(m=0),
                on_disk_payload=False,
            )
        else:  # dense
            self._client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE, on_disk=True),
                hnsw_config=HnswConfigDiff(m=0),
                on_disk_payload=False,
            )

        print(f"  Qdrant 컬렉션 생성: {name}  mode={vector_mode}  dim={dim}", flush=True)

    def upload_stream(self, name: str, points, vector_mode: str = "dense") -> None:
        """generator를 받아 upload_points로 streaming upsert (Qdrant 공식 권장 방식)."""
        self._client.upload_points(
            collection_name=name,
            points=points,
            batch_size=_UPLOAD_BATCH,
            parallel=1,
            max_retries=3,
        )

    def finalize_index(self, name: str, vector_mode: str = "dense") -> None:
        if vector_mode == "sparse":
            print(f"  sparse 인덱스 완료 (inverted index)", flush=True)
            return
        if vector_mode == "colbert":
            # colbert는 680만+ HNSW 노드 재구성 → 수 시간 + OOM
            # m=0(flat search)으로 정확한 MaxSim 계산 — 벤치마크에 더 적합
            print(f"  colbert 인덱스 완료 (flat/brute-force MaxSim)", flush=True)
            return
        from qdrant_client.models import HnswConfigDiff
        self._client.update_collection(
            collection_name=name,
            hnsw_config=HnswConfigDiff(m=32, ef_construct=256),
        )
        print(f"  HNSW 활성화: m=32, ef_construct=256", flush=True)

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

        if vector_mode == "sparse":
            from qdrant_client.models import SparseVector, QueryRequest
            for start in range(0, n, CHUNK):
                end = min(start + CHUNK, n)
                batch = self._client.query_batch_points(
                    collection_name=name,
                    requests=[
                        QueryRequest(
                            query=SparseVector(
                                indices=[int(k) for k in vectors[i]],
                                values=[float(v) for v in vectors[i].values()],
                            ),
                            using="sparse",
                            limit=top_k,
                            with_payload=True,
                        )
                        for i in range(start, end)
                    ],
                )
                for result in batch:
                    all_results.append([(r.payload["doc_id"], r.score) for r in result.points])

        elif vector_mode == "colbert":
            from qdrant_client.models import QueryRequest
            for start in range(0, n, CHUNK):
                end = min(start + CHUNK, n)
                batch = self._client.query_batch_points(
                    collection_name=name,
                    requests=[
                        QueryRequest(
                            query=vectors[i].tolist(),  # 2D: [[tok_dim...], ...]
                            limit=top_k,
                            with_payload=True,
                        )
                        for i in range(start, end)
                    ],
                )
                for result in batch:
                    all_results.append([(r.payload["doc_id"], r.score) for r in result.points])

        else:  # dense
            from qdrant_client.models import QueryRequest, SearchParams
            for start in range(0, n, CHUNK):
                end = min(start + CHUNK, n)
                batch = self._client.query_batch_points(
                    collection_name=name,
                    requests=[
                        QueryRequest(
                            query=vectors[i].tolist(),  # 1D: [dim1, dim2, ...]
                            params=SearchParams(hnsw_ef=256),
                            limit=top_k,
                            with_payload=True,
                        )
                        for i in range(start, end)
                    ],
                )
                for result in batch:
                    all_results.append([(r.payload["doc_id"], r.score) for r in result.points])

        return all_results

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
                values=[float(v) for v in vec.values()],
            )},
            payload={"doc_id": doc_id},
        )
    elif vector_mode == "colbert":
        return PointStruct(
            id=point_id,
            vector=vec.tolist(),
            payload={"doc_id": doc_id},
        )
    else:
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
    """
    Qdrant 공식 권장 패턴: generator + upload_points streaming.
    encode는 _ENCODE_CHUNK 단위로 chunking하여 OOM 방지.
    """
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
