"""
VectorDB 추상화 레이어.

vector_mode별 구성:
  dense  : Qdrant VectorParams (cosine, on_disk=False) — flat/exact search (m=0)
  sparse : SparseDiskIndex — scipy CSR matrix + 배치 sparse matmul 검색
           (QdrantLocal은 sparse inverted index를 Python으로 빌드 → 수십분 block)
  colbert: ColbertDiskIndex — numpy fp16 바이너리 + GPU MaxSim 검색
           (QdrantLocal은 colbert 멀티벡터를 Python list로 저장 → 42GB RAM OOM)

인덱싱 패턴:
  - encode는 _ENCODE_CHUNK 단위로 chunking (OOM 방지)
  - dense: upload_points(generator) streaming (Qdrant 공식)
  - sparse: SparseDiskIndex → scipy CSR 저장
  - colbert: ColbertDiskIndex → fp16 바이너리 저장
"""
from __future__ import annotations

import gc
import heapq
import json
import math
import os
import shutil
from dataclasses import dataclass

import numpy as np

_ENCODE_CHUNK = 2_000   # 인코딩 단위 (메모리 제어)
_UPLOAD_BATCH = 256     # Qdrant upload_points 내부 배치 크기


# ── 경량 컨테이너 (Qdrant 객체 생성 오버헤드 없음) ────────────────────────────

@dataclass
class _SparsePoint:
    doc_id:  str
    indices: list   # list[int]  token IDs
    values:  list   # list[float] weights


@dataclass
class _ColbertPoint:
    doc_id: str
    vec: np.ndarray  # [n_tokens, 1024] float16


# ── 인터페이스 ────────────────────────────────────────────────────────────────

class VectorStore:
    def has_collection(self, name: str) -> bool:                    raise NotImplementedError
    def collection_size(self, name: str) -> int:                    raise NotImplementedError
    def create_collection(self, name: str, dim: int, vector_mode: str = "dense") -> None: raise NotImplementedError
    def upload_stream(self, name: str, points, vector_mode: str = "dense") -> None: raise NotImplementedError
    def finalize_index(self, name: str, vector_mode: str = "dense") -> None:  raise NotImplementedError
    def search_batch(self, name: str, vectors, top_k: int, vector_mode: str = "dense") -> list[list[tuple[str, float]]]: raise NotImplementedError
    def drop_collection(self, name: str) -> None:                   raise NotImplementedError


# ── Sparse 디스크 인덱스 ──────────────────────────────────────────────────────

class SparseDiskIndex:
    """
    scipy CSR matrix 기반 sparse 인덱스.

    저장 레이아웃 (base_path/sparse_{name}/):
      matrix.npz   — scipy CSR sparse matrix (n_docs × vocab_size)
      doc_ids.json — list[str] doc ID 매핑
      meta.json    — {"n_docs", "n_vocab"}

    검색: 배치 sparse matrix-matrix product
      scores = Q_batch @ matrix.T  → top-k
    """

    def __init__(self, base_path: str) -> None:
        self._base = base_path

    def _dir(self, name: str) -> str:
        return os.path.join(self._base, f"sparse_{name}")

    def has(self, name: str) -> bool:
        return os.path.isfile(os.path.join(self._dir(name), "meta.json"))

    def size(self, name: str) -> int:
        with open(os.path.join(self._dir(name), "meta.json")) as f:
            return json.load(f)["n_docs"]

    def create(self, name: str) -> None:
        os.makedirs(self._dir(name), exist_ok=True)
        print(f"  Sparse disk 인덱스 생성: {self._dir(name)}", flush=True)

    def write_stream(self, name: str, points) -> None:
        """_SparsePoint generator를 받아 scipy CSR matrix로 저장."""
        from scipy.sparse import csr_matrix
        import scipy.sparse

        d = self._dir(name)
        doc_ids: list[str] = []
        rows:    list[int] = []
        cols:    list[int] = []
        data:    list[float] = []
        doc_idx  = 0
        max_col  = 0

        print("  [sparse-disk] 저장 시작...", flush=True)
        for pt in points:
            for idx, val in zip(pt.indices, pt.values):
                rows.append(doc_idx)
                cols.append(idx)
                data.append(val)
                if idx > max_col:
                    max_col = idx
            doc_ids.append(pt.doc_id)
            doc_idx += 1

        n_docs  = doc_idx
        n_vocab = max_col + 1

        print(f"  [sparse-disk] CSR 행렬 구성 중 ({len(data):,}개 비영 원소)...", flush=True)
        rows_np = np.array(rows, dtype=np.int32);  del rows
        cols_np = np.array(cols, dtype=np.int32);  del cols
        data_np = np.array(data, dtype=np.float32); del data

        matrix = csr_matrix(
            (data_np, (rows_np, cols_np)),
            shape=(n_docs, n_vocab),
            dtype=np.float32,
        )
        del rows_np, cols_np, data_np

        scipy.sparse.save_npz(os.path.join(d, "matrix.npz"), matrix)
        with open(os.path.join(d, "doc_ids.json"), "w") as f:
            json.dump(doc_ids, f)
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump({"n_docs": n_docs, "n_vocab": n_vocab}, f)

        size_mb = matrix.data.nbytes / 1e6
        print(
            f"  [sparse-disk] 저장 완료: {n_docs:,}docs "
            f"vocab={n_vocab:,} nnz={matrix.nnz:,} ({size_mb:.0f}MB)",
            flush=True,
        )

    def search(
        self, name: str, query_vecs, top_k: int
    ) -> list[list[tuple[str, float]]]:
        """배치 sparse matmul로 검색."""
        import scipy.sparse

        d = self._dir(name)
        with open(os.path.join(d, "meta.json")) as f:
            meta = json.load(f)
        n_docs  = meta["n_docs"]
        n_vocab = meta["n_vocab"]

        matrix = scipy.sparse.load_npz(os.path.join(d, "matrix.npz"))  # (n_docs, n_vocab) CSR
        with open(os.path.join(d, "doc_ids.json")) as f:
            doc_ids = json.load(f)

        n_queries = len(query_vecs)
        results: list[list[tuple[str, float]]] = []
        Q_BATCH  = 512
        log_every = max(1, (n_queries // Q_BATCH) // 5)

        print(f"  [sparse-search] {n_queries:,}queries × {n_docs:,}docs", flush=True)

        for q_s in range(0, n_queries, Q_BATCH):
            q_e   = min(q_s + Q_BATCH, n_queries)
            bsz   = q_e - q_s
            batch = query_vecs[q_s:q_e]

            # query sparse matrix: (bsz, n_vocab)
            q_rows, q_cols, q_data = [], [], []
            for qi, q in enumerate(batch):
                for k, v in q.items():
                    idx = int(k)
                    if idx < n_vocab:
                        q_rows.append(qi)
                        q_cols.append(idx)
                        q_data.append(float(v))

            Q_mat = scipy.sparse.csr_matrix(
                (q_data, (q_rows, q_cols)),
                shape=(bsz, n_vocab),
                dtype=np.float32,
            )

            # scores: (bsz, n_docs) dense — ~272MB per batch
            scores = (Q_mat @ matrix.T).toarray()

            for qi in range(bsz):
                top_idx = np.argpartition(scores[qi], -top_k)[-top_k:]
                top_idx = top_idx[np.argsort(-scores[qi][top_idx])]
                results.append([(doc_ids[i], float(scores[qi, i])) for i in top_idx])

            bi = q_s // Q_BATCH
            if bi % log_every == 0 or q_e == n_queries:
                print(f"  [sparse-search] {q_e:,}/{n_queries:,}", flush=True)

        return results

    def drop(self, name: str) -> None:
        d = self._dir(name)
        if os.path.isdir(d):
            shutil.rmtree(d)


# ── ColBERT 디스크 인덱스 ─────────────────────────────────────────────────────

class ColbertDiskIndex:
    """
    Qdrant를 우회한 ColBERT 디스크 인덱스.

    저장 레이아웃 (base_path/colbert_{name}/):
      vectors.bin  — flat float16 바이너리 (total_tokens × dim)
      offsets.npy  — int64 (n_docs+1,) 누적 토큰 수
      doc_ids.json — list[str] doc ID 매핑
      meta.json    — {"n_docs", "total_tokens", "dim"}

    검색: GPU MaxSim (fallback: CPU numpy)
      score(q, d) = sum_t max_s (q_t · d_s)
    """

    def __init__(self, base_path: str) -> None:
        self._base = base_path

    def _dir(self, name: str) -> str:
        return os.path.join(self._base, f"colbert_{name}")

    def has(self, name: str) -> bool:
        return os.path.isfile(os.path.join(self._dir(name), "meta.json"))

    def size(self, name: str) -> int:
        with open(os.path.join(self._dir(name), "meta.json")) as f:
            return json.load(f)["n_docs"]

    def create(self, name: str) -> None:
        os.makedirs(self._dir(name), exist_ok=True)
        print(f"  ColBERT disk 인덱스 생성: {self._dir(name)}", flush=True)

    def write_stream(self, name: str, points) -> None:
        """_ColbertPoint generator를 받아 fp16 바이너리로 스트리밍 저장."""
        d = self._dir(name)
        vec_path = os.path.join(d, "vectors.bin")
        DIM = 1024

        doc_ids:      list[str] = []
        offsets:      list[int] = [0]
        total_tokens  = 0
        n_docs        = 0

        print("  [colbert-disk] 저장 시작...", flush=True)
        with open(vec_path, "wb") as vf:
            for pt in points:
                arr = np.asarray(pt.vec, dtype=np.float16)
                vf.write(arr.tobytes())
                total_tokens += len(arr)
                offsets.append(total_tokens)
                doc_ids.append(pt.doc_id)
                n_docs += 1

        np.save(os.path.join(d, "offsets.npy"), np.array(offsets, dtype=np.int64))
        with open(os.path.join(d, "doc_ids.json"), "w") as f:
            json.dump(doc_ids, f)
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump({"n_docs": n_docs, "total_tokens": total_tokens, "dim": DIM}, f)

        size_gb = total_tokens * DIM * 2 / 1e9
        print(
            f"  [colbert-disk] 저장 완료: {n_docs:,}docs "
            f"{total_tokens:,}tokens {size_gb:.1f}GB",
            flush=True,
        )

    def search(
        self, name: str, query_vecs, top_k: int
    ) -> list[list[tuple[str, float]]]:
        """GPU MaxSim 검색. GPU 없으면 CPU fallback."""
        d = self._dir(name)
        with open(os.path.join(d, "meta.json")) as f:
            meta = json.load(f)
        n_docs        = meta["n_docs"]
        total_tokens  = meta["total_tokens"]
        dim           = meta["dim"]

        offsets = np.load(os.path.join(d, "offsets.npy"))
        with open(os.path.join(d, "doc_ids.json")) as f:
            doc_ids = json.load(f)

        all_tokens = np.memmap(
            os.path.join(d, "vectors.bin"),
            dtype=np.float16, mode="r", shape=(total_tokens, dim),
        )

        n_queries = len(query_vecs)
        results: list[list[tuple[str, float]]] = [[] for _ in range(n_queries)]

        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except ImportError:
            use_gpu = False

        Q_BATCH   = 512
        log_every = max(1, n_docs // 10)

        print(f"  [colbert-search] {n_queries:,}queries × {n_docs:,}docs  gpu={use_gpu}", flush=True)

        if use_gpu:
            import torch

            for q_s in range(0, n_queries, Q_BATCH):
                q_e  = min(q_s + Q_BATCH, n_queries)
                bsz  = q_e - q_s
                batch = query_vecs[q_s:q_e]
                max_q = max(len(q) for q in batch)

                Q_np = np.zeros((bsz, max_q, dim), dtype=np.float32)
                M_np = np.zeros((bsz, max_q),      dtype=np.float32)
                for i, q in enumerate(batch):
                    q_arr = np.asarray(q, dtype=np.float32)
                    Q_np[i, :len(q_arr)] = q_arr
                    M_np[i, :len(q_arr)] = 1.0

                Q_flat = torch.from_numpy(Q_np.reshape(bsz * max_q, dim)).cuda()
                M_gpu  = torch.from_numpy(M_np).cuda()

                scores = np.zeros((bsz, n_docs), dtype=np.float32)

                for d_idx in range(n_docs):
                    t_s = int(offsets[d_idx])
                    t_e = int(offsets[d_idx + 1])
                    if t_s == t_e:
                        continue

                    d_tok = torch.from_numpy(
                        np.array(all_tokens[t_s:t_e], dtype=np.float32)
                    ).cuda()

                    sim     = Q_flat @ d_tok.T
                    sim     = sim.view(bsz, max_q, -1)
                    sim     = sim * M_gpu.unsqueeze(-1)
                    max_sim = sim.max(dim=-1).values
                    scores[:, d_idx] = max_sim.sum(dim=-1).cpu().numpy()

                    if (d_idx + 1) % log_every == 0 or d_idx + 1 == n_docs:
                        print(
                            f"  [colbert-search] queries {q_e:,}/{n_queries:,} "
                            f"docs {d_idx+1:,}/{n_docs:,}",
                            flush=True,
                        )

                for qi in range(bsz):
                    top_idx = np.argpartition(scores[qi], -top_k)[-top_k:]
                    top_idx = top_idx[np.argsort(-scores[qi][top_idx])]
                    results[q_s + qi] = [
                        (doc_ids[i], float(scores[qi, i])) for i in top_idx
                    ]

        else:
            heaps: list[list] = [[] for _ in range(n_queries)]

            for d_idx in range(n_docs):
                t_s = int(offsets[d_idx])
                t_e = int(offsets[d_idx + 1])
                if t_s == t_e:
                    continue
                d_tok = np.array(all_tokens[t_s:t_e], dtype=np.float32)

                for q_idx, q in enumerate(query_vecs):
                    q_arr = np.asarray(q, dtype=np.float32)
                    sim   = q_arr @ d_tok.T
                    score = float(sim.max(axis=1).sum())
                    heap  = heaps[q_idx]
                    if len(heap) < top_k:
                        heapq.heappush(heap, (score, doc_ids[d_idx]))
                    elif score > heap[0][0]:
                        heapq.heapreplace(heap, (score, doc_ids[d_idx]))

                if (d_idx + 1) % log_every == 0 or d_idx + 1 == n_docs:
                    print(f"  [colbert-search] {d_idx+1:,}/{n_docs:,}", flush=True)

            for q_idx, heap in enumerate(heaps):
                results[q_idx] = [
                    (doc_id, score)
                    for score, doc_id in sorted(heap, reverse=True)
                ]

        return results

    def drop(self, name: str) -> None:
        d = self._dir(name)
        if os.path.isdir(d):
            shutil.rmtree(d)


# ── Qdrant 구현 ───────────────────────────────────────────────────────────────

class QdrantStore(VectorStore):
    """Qdrant (dense) + SparseDiskIndex (sparse) + ColbertDiskIndex (colbert)."""

    def __init__(self, *, path: str | None = None, url: str | None = None) -> None:
        from qdrant_client import QdrantClient
        if url:
            self._client  = QdrantClient(url=url)
            self._sparse  = None
            self._colbert = None
            print(f"[Qdrant] 원격 서버: {url}", flush=True)
        else:
            self._client  = QdrantClient(path=path)
            self._sparse  = SparseDiskIndex(path)
            self._colbert = ColbertDiskIndex(path)
            print(f"[Qdrant] on-disk: {path}", flush=True)

    def has_collection(self, name: str) -> bool:
        if self._sparse  is not None and self._sparse.has(name):   return True
        if self._colbert is not None and self._colbert.has(name):  return True
        return name in {c.name for c in self._client.get_collections().collections}

    def collection_size(self, name: str) -> int:
        if self._sparse  is not None and self._sparse.has(name):   return self._sparse.size(name)
        if self._colbert is not None and self._colbert.has(name):  return self._colbert.size(name)
        return self._client.get_collection(name).points_count

    def create_collection(self, name: str, dim: int, vector_mode: str = "dense") -> None:
        if vector_mode == "sparse":
            if self._sparse is None:
                raise ValueError("Sparse disk index는 path 기반 QdrantStore 필요")
            self._sparse.create(name)
            return
        if vector_mode == "colbert":
            if self._colbert is None:
                raise ValueError("ColBERT disk index는 path 기반 QdrantStore 필요")
            self._colbert.create(name)
            return

        from qdrant_client.models import Distance, VectorParams, HnswConfigDiff
        self._client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=dim, distance=Distance.COSINE, on_disk=False,
            ),
            hnsw_config=HnswConfigDiff(m=0),
            on_disk_payload=False,
        )
        print(f"  Qdrant 컬렉션 생성: {name}  mode={vector_mode}  dim={dim}", flush=True)

    def upload_stream(self, name: str, points, vector_mode: str = "dense") -> None:
        if vector_mode == "sparse":
            assert self._sparse is not None
            self._sparse.write_stream(name, points)
            return
        if vector_mode == "colbert":
            assert self._colbert is not None
            self._colbert.write_stream(name, points)
            return

        print("  [upload] upload_points 시작...", flush=True)
        self._client.upload_points(
            collection_name=name,
            points=points,
            batch_size=_UPLOAD_BATCH,
            parallel=1,
            max_retries=3,
        )
        print("  [upload] upload_points 완료", flush=True)

    def finalize_index(self, name: str, vector_mode: str = "dense") -> None:
        labels = {"dense": "flat cosine", "sparse": "sparse matmul", "colbert": "flat MaxSim"}
        print(f"  인덱스 완료 ({labels.get(vector_mode, vector_mode)})", flush=True)

    def search_batch(
        self,
        name:        str,
        vectors,
        top_k:       int,
        vector_mode: str = "dense",
    ) -> list[list[tuple[str, float]]]:
        if vector_mode == "sparse":
            assert self._sparse is not None
            return self._sparse.search(name, vectors, top_k)
        if vector_mode == "colbert":
            assert self._colbert is not None
            return self._colbert.search(name, vectors, top_k)

        # dense
        from qdrant_client.models import QueryRequest
        CHUNK = 256
        n = len(vectors)
        all_results: list[list[tuple[str, float]]] = []

        for start in range(0, n, CHUNK):
            end   = min(start + CHUNK, n)
            batch = self._client.query_batch_points(
                collection_name=name,
                requests=[
                    QueryRequest(
                        query=vectors[i].tolist(),
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
        if self._sparse  is not None and self._sparse.has(name):
            self._sparse.drop(name);  return
        if self._colbert is not None and self._colbert.has(name):
            self._colbert.drop(name); return
        self._client.delete_collection(name)


# ── PointStruct / 경량 컨테이너 팩토리 ───────────────────────────────────────

def _make_point(point_id: int, doc_id: str, vec, vector_mode: str):
    if vector_mode == "sparse":
        return _SparsePoint(
            doc_id=doc_id,
            indices=[int(k) for k in vec],
            values=[float(v) for v in vec.values()],
        )
    if vector_mode == "colbert":
        return _ColbertPoint(doc_id=doc_id, vec=vec)

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
