"""
VectorDB 추상화 레이어.

vector_mode별 Qdrant 컬렉션 구성:
  dense  : VectorParams (cosine, on_disk) — HNSW
  sparse : SparseVectorParams             — inverted index
  colbert: VectorParams + MultiVectorConfig(MAX_SIM) — HNSW multi-vector

VectorDB 교체 방법:
  1. VectorStore 인터페이스를 구현하는 새 클래스 작성
  2. runner.py 의 build_store() 에서 해당 클래스로 교체

Qdrant 파라미터 근거:
  - bulk insert 중 m=0: HNSW graph 재구성 thrashing 방지 (Qdrant 공식 권장)
  - 완료 후 m=32, ef_construct=256: 고품질 recall 목적 벤치마크 균형값
  - sparse: IDF modifier=idf 적용으로 BM25-like 스코어링
"""
from __future__ import annotations

import gc
import math

import numpy as np

_UPSERT_CHUNK = 2_000


# ── 인터페이스 ────────────────────────────────────────────────────────────────

class VectorStore:
    def has_collection(self, name: str) -> bool:                    raise NotImplementedError
    def collection_size(self, name: str) -> int:                    raise NotImplementedError
    def create_collection(self, name: str, dim: int, vector_mode: str = "dense") -> None: raise NotImplementedError
    def upsert_vectors(self, name: str, offset: int, doc_ids: list[str], vectors) -> None: raise NotImplementedError
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
            SparseVectorParams, SparseIndexParams, ModifierType,
            MultiVectorConfig, MultiVectorComparator,
        )

        if vector_mode == "sparse":
            self._client.create_collection(
                collection_name=name,
                vectors_config={},
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(on_disk=True, full_scan_threshold=5000),
                        modifier=ModifierType.IDF,
                    )
                },
                on_disk_payload=True,
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
                on_disk_payload=True,
            )
        else:  # dense
            self._client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE, on_disk=True),
                hnsw_config=HnswConfigDiff(m=0),
                on_disk_payload=True,
            )

        print(f"  Qdrant 컬렉션 생성: {name}  mode={vector_mode}  dim={dim}", flush=True)

    def upsert_vectors(
        self,
        name:        str,
        offset:      int,
        doc_ids:     list[str],
        vectors,
        vector_mode: str = "dense",
    ) -> None:
        from qdrant_client.models import PointStruct

        if vector_mode == "sparse":
            from qdrant_client.models import SparseVector
            points = [
                PointStruct(
                    id=offset + i,
                    vector={
                        "sparse": SparseVector(
                            indices=[int(k) for k in vectors[i]],
                            values=[float(v) for v in vectors[i].values()],
                        )
                    },
                    payload={"doc_id": doc_ids[i]},
                )
                for i in range(len(doc_ids))
            ]
        elif vector_mode == "colbert":
            points = [
                PointStruct(
                    id=offset + i,
                    vector=vectors[i].tolist(),   # 2-D list [n_tokens, dim]
                    payload={"doc_id": doc_ids[i]},
                )
                for i in range(len(doc_ids))
            ]
        else:  # dense
            points = [
                PointStruct(
                    id=offset + i,
                    vector=vectors[i].tolist(),
                    payload={"doc_id": doc_ids[i]},
                )
                for i in range(len(doc_ids))
            ]

        self._client.upsert(collection_name=name, points=points)

    def finalize_index(self, name: str, vector_mode: str = "dense") -> None:
        if vector_mode == "sparse":
            # sparse는 inverted index라 HNSW 설정 불필요
            print(f"  sparse 인덱스 완료 (inverted index)", flush=True)
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

        else:  # dense or colbert (both use float vector query)
            from qdrant_client.models import QueryRequest, SearchParams
            search_params = {} if vector_mode == "colbert" else {"params": SearchParams(hnsw_ef=256)}
            for start in range(0, n, CHUNK):
                end = min(start + CHUNK, n)
                batch = self._client.query_batch_points(
                    collection_name=name,
                    requests=[
                        QueryRequest(
                            query=vectors[i].tolist(),
                            limit=top_k,
                            with_payload=True,
                            **search_params,
                        )
                        for i in range(start, end)
                    ],
                )
                for result in batch:
                    all_results.append([(r.payload["doc_id"], r.score) for r in result.points])

        return all_results

    def drop_collection(self, name: str) -> None:
        self._client.delete_collection(name)


# ── 인덱싱 오케스트레이터 ─────────────────────────────────────────────────────

def index_docs(
    store:       VectorStore,
    name:        str,
    model,
    docs:        dict,
    batch_size:  int,
) -> None:
    """docs 전체를 store에 인코딩 후 upsert. vector_mode는 model.vector_mode 참조."""
    try:
        import torch
        _has_cuda = torch.cuda.is_available()
    except ImportError:
        _has_cuda = False

    vector_mode = getattr(model, "vector_mode", "dense")
    doc_ids     = list(docs.keys())
    n_total     = len(doc_ids)
    n_chunks    = math.ceil(n_total / _UPSERT_CHUNK)
    _log_every  = max(1, n_chunks // 10)

    store.create_collection(name, model.dim, vector_mode=vector_mode)

    for ci in range(n_chunks):
        s = ci * _UPSERT_CHUNK
        e = min(s + _UPSERT_CHUNK, n_total)
        chunk_ids = doc_ids[s:e]
        texts = [
            f"{docs[d].get('title', '')} {docs[d]['chunk']}".strip()
            for d in chunk_ids
        ]

        embs = model.encode_docs(texts, batch_size)
        del texts
        store.upsert_vectors(name, s, chunk_ids, embs, vector_mode=vector_mode)
        del embs

        if _has_cuda:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()

        if (ci + 1) % _log_every == 0 or ci + 1 == n_chunks:
            print(f"  upsert {e:,}/{n_total:,}", flush=True)

    store.finalize_index(name, vector_mode=vector_mode)
    print(f"  인덱싱 완료: {n_total:,}건", flush=True)
