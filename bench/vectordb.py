"""
VectorDB 추상화 레이어 (Qdrant Rust 서버 전용, dense 전용).

dense: VectorParams cosine, bulk upload 중 m=0, 완료 후 HNSW m=16 복원.
sparse/colbert는 model.py에 vector_mode 추가 시 확장 예정.
"""
from __future__ import annotations

import gc
import math

import numpy as np

_ENCODE_CHUNK = 2_000
_UPLOAD_BATCH = 256


# ── 인터페이스 ────────────────────────────────────────────────────────────────

class VectorStore:
    def has_collection(self, name: str) -> bool:          raise NotImplementedError
    def collection_size(self, name: str) -> int:          raise NotImplementedError
    def create_collection(self, name: str, dim: int) -> None: raise NotImplementedError
    def upload_stream(self, name: str, points) -> None:   raise NotImplementedError
    def finalize_index(self, name: str) -> None:          raise NotImplementedError
    def search_batch(self, name: str, vectors, top_k: int) -> list[list[tuple[str, float]]]: raise NotImplementedError
    def drop_collection(self, name: str) -> None:         raise NotImplementedError


# ── Qdrant 구현 ───────────────────────────────────────────────────────────────

class QdrantStore(VectorStore):
    """Qdrant Rust 서버 기반 dense VectorDB."""

    def __init__(self, url: str) -> None:
        from qdrant_client import QdrantClient
        self._client = QdrantClient(url=url, timeout=300, check_compatibility=False)
        print(f"[Qdrant] 서버: {url}", flush=True)

    def has_collection(self, name: str) -> bool:
        return name in {c.name for c in self._client.get_collections().collections}

    def collection_size(self, name: str) -> int:
        return self._client.get_collection(name).points_count

    def create_collection(self, name: str, dim: int) -> None:
        from qdrant_client.models import Distance, VectorParams, HnswConfigDiff
        self._client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE, on_disk=False),
            hnsw_config=HnswConfigDiff(m=0),  # bulk insert 중 HNSW 구성 억제
            on_disk_payload=False,
        )
        print(f"  Qdrant 컬렉션 생성: {name}  dim={dim}", flush=True)

    def upload_stream(self, name: str, points) -> None:
        print("  [upload] upload_points 시작...", flush=True)
        self._client.upload_points(
            collection_name=name,
            points=points,
            batch_size=_UPLOAD_BATCH,
            parallel=1,
            max_retries=3,
        )
        print("  [upload] upload_points 완료", flush=True)

    def finalize_index(self, name: str) -> None:
        """bulk insert 완료 후 HNSW 활성화."""
        from qdrant_client.models import HnswConfigDiff
        self._client.update_collection(
            collection_name=name,
            hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
        )
        print("  HNSW 활성화: m=16, ef_construct=100 (백그라운드 구성 중...)", flush=True)

    def search_batch(
        self,
        name:   str,
        vectors,
        top_k:  int,
    ) -> list[list[tuple[str, float]]]:
        # query_batch_points + QueryRequest: 공식 권장 API (search_batch는 1.16.0에서 제거됨)
        # unnamed vector는 using 생략이 올바른 방식 (Qdrant 공식 docs 확인)
        from qdrant_client.models import QueryRequest

        CHUNK = 256
        n = len(vectors)
        all_results: list[list[tuple[str, float]]] = []

        for start in range(0, n, CHUNK):
            end = min(start + CHUNK, n)
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

    def drop_collection(self, name: str) -> None:
        self._client.delete_collection(name)


# ── PointStruct 팩토리 ────────────────────────────────────────────────────────

def _make_point(point_id: int, doc_id: str, vec: np.ndarray):
    from qdrant_client.models import PointStruct
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

    doc_ids    = list(docs.keys())
    n_total    = len(doc_ids)
    n_chunks   = math.ceil(n_total / _ENCODE_CHUNK)
    _log_every = max(1, n_chunks // 10)

    store.create_collection(name, model.dim)

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
                yield _make_point(point_id, doc_id, vec)
                point_id += 1

            del embs
            if _has_cuda:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            gc.collect()

            if (ci + 1) % _log_every == 0 or ci + 1 == n_chunks:
                print(f"  인코딩 {e:,}/{n_total:,}", flush=True)

    store.upload_stream(name, _point_generator())
    store.finalize_index(name)
    print(f"  인덱싱 완료: {n_total:,}건", flush=True)
