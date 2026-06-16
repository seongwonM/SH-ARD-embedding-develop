"""
VectorDB 추상화 레이어.

VectorDB 교체 방법:
  - Qdrant(기본): QdrantStore 그대로 사용
  - 다른 VectorDB(Milvus, Chroma, Weaviate 등):
      1. VectorStore 인터페이스를 구현하는 새 클래스 작성
      2. runner.py 의 build_store() 호출 부분에서 해당 클래스로 교체
      인터페이스만 맞추면 runner.py / evaluator.py 코드는 무수정.

Qdrant 파라미터 근거:
  - bulk insert 중 m=0: HNSW graph 재구성 thrashing 방지 (Qdrant 공식 권장)
    Source: https://qdrant.tech/articles/vector-search-resource-optimization/
  - 완료 후 m=32, ef_construct=256: 고품질 recall 목적 벤치마크 균형값
    Source: https://qdrant.tech/documentation/tutorials-search-engineering/retrieval-quality/
  - search hnsw_ef=256: ef_construct와 동일값 → 최대 recall 보장
  - search_batch 대신 query_batch_points: qdrant-client 1.7+ 공식 API
    (search_batch는 deprecated 후 removed)
"""
from __future__ import annotations

import gc
import math

import numpy as np

_UPSERT_CHUNK = 2_000


# ── 인터페이스 ────────────────────────────────────────────────────────────────

class VectorStore:
    """VectorDB 교체 시 이 인터페이스를 구현하세요."""

    def has_collection(self, name: str) -> bool:
        raise NotImplementedError

    def collection_size(self, name: str) -> int:
        raise NotImplementedError

    def create_collection(self, name: str, dim: int) -> None:
        raise NotImplementedError

    def upsert_vectors(
        self,
        name:     str,
        offset:   int,
        doc_ids:  list[str],
        vectors:  np.ndarray,
    ) -> None:
        raise NotImplementedError

    def finalize_index(self, name: str) -> None:
        raise NotImplementedError

    def search_batch(
        self,
        name:    str,
        vectors: np.ndarray,
        top_k:   int,
    ) -> list[list[tuple[str, float]]]:
        """(doc_id, score) 튜플 리스트의 리스트 반환."""
        raise NotImplementedError

    def drop_collection(self, name: str) -> None:
        raise NotImplementedError


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

    def create_collection(self, name: str, dim: int) -> None:
        from qdrant_client.models import Distance, VectorParams, HnswConfigDiff
        self._client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE, on_disk=True),
            hnsw_config=HnswConfigDiff(m=0),  # bulk insert 중 graph 구성 억제
            on_disk_payload=True,
        )
        print(f"  Qdrant 컬렉션 생성: {name}  dim={dim}", flush=True)

    def upsert_vectors(
        self,
        name:    str,
        offset:  int,
        doc_ids: list[str],
        vectors: np.ndarray,
    ) -> None:
        from qdrant_client.models import PointStruct
        self._client.upsert(
            collection_name=name,
            points=[
                PointStruct(
                    id=offset + i,
                    vector=vectors[i].tolist(),
                    payload={"doc_id": doc_ids[i]},
                )
                for i in range(len(doc_ids))
            ],
        )

    def finalize_index(self, name: str) -> None:
        """bulk insert 완료 후 HNSW 활성화."""
        from qdrant_client.models import HnswConfigDiff
        self._client.update_collection(
            collection_name=name,
            hnsw_config=HnswConfigDiff(m=32, ef_construct=256),
        )
        print(f"  HNSW 활성화: m=32, ef_construct=256 (백그라운드 구성 중...)", flush=True)

    def search_batch(
        self,
        name:    str,
        vectors: np.ndarray,
        top_k:   int,
    ) -> list[list[tuple[str, float]]]:
        from qdrant_client.models import QueryRequest, SearchParams

        CHUNK = 256
        n = len(vectors)
        all_results: list[list[tuple[str, float]]] = []

        for start in range(0, n, CHUNK):
            end = min(start + CHUNK, n)
            batch = self._client.query_batch_points(
                collection_name=name,
                requests=[
                    QueryRequest(
                        query=vectors[i].tolist(),
                        limit=top_k,
                        with_payload=True,
                        params=SearchParams(hnsw_ef=256),
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
    store:      VectorStore,
    name:       str,
    model,
    docs:       dict,
    batch_size: int,
) -> None:
    """
    docs 전체를 store 에 인코딩 후 upsert.
    벡터를 청크 단위로 처리해 RAM OOM 방지 (MIRACL 150만 건 대응).
    """
    try:
        import torch
        _has_cuda = torch.cuda.is_available()
    except ImportError:
        _has_cuda = False

    doc_ids = list(docs.keys())
    n_total = len(doc_ids)
    n_chunks  = math.ceil(n_total / _UPSERT_CHUNK)
    _log_every = max(1, n_chunks // 10)

    store.create_collection(name, model.dim)

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
        store.upsert_vectors(name, s, chunk_ids, embs)
        del embs

        if _has_cuda:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()

        if (ci + 1) % _log_every == 0 or ci + 1 == n_chunks:
            print(f"  upsert {e:,}/{n_total:,}", flush=True)

    store.finalize_index(name)
    print(f"  인덱싱 완료: {n_total:,}건", flush=True)
