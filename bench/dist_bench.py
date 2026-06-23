"""
Milvus 분산처리 성능 비교 테스트.

기존 bench/runner.py (standalone 기준값)와 독립적으로 동작.
이미 실행 중인 Milvus 클러스터에 접속해서
num_shards × replica_number × search_workers 조합별 QPS / NDCG 를 측정한다.

사용법 (RunPod coordinator 또는 로컬):
  python -m bench.dist_bench \\
    --milvus-uri http://localhost:19530 \\
    --model BAAI/bge-m3 \\
    --num-shards 1 2 \\
    --replica-number 1 2 \\
    --search-workers 1 2 4
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bench.data_loader import load_from_dir
from bench.evaluator import evaluate
from bench.model import build_model

_INSERT_BATCH = 64
_ENCODE_CHUNK = 2_000


# ── 컬렉션 생성 ───────────────────────────────────────────────────────────────

def _create_collection(client, name: str, dim: int, vector_mode: str, num_shards: int) -> None:
    from pymilvus import MilvusClient, DataType

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id",     DataType.INT64,  is_primary=True)
    schema.add_field("doc_id", DataType.VARCHAR, max_length=512)

    index_params = MilvusClient.prepare_index_params()

    if vector_mode == "sparse":
        schema.add_field("vector", DataType.SPARSE_FLOAT_VECTOR)
        index_params.add_index("vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
    elif vector_mode == "colbert":
        ss = client.create_struct_field_schema()
        ss.add_field("emb", DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field("colbert_embs", DataType.ARRAY,
                         element_type=DataType.STRUCT, struct_schema=ss, max_capacity=512)
        index_params.add_index("colbert_embs[emb]", index_type="AUTOINDEX",
                                metric_type="MAX_SIM_COSINE")
    else:
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
        index_params.add_index("vector", index_type="HNSW", metric_type="COSINE",
                                params={"M": 16, "efConstruction": 100})

    client.create_collection(name, schema=schema, index_params=index_params,
                             num_shards=num_shards)
    print(f"  컬렉션 생성: {name}  mode={vector_mode}  dim={dim}  shards={num_shards}",
          flush=True)


# ── 인덱싱 ────────────────────────────────────────────────────────────────────

def _index_and_load(client, name: str, model, docs: dict,
                    batch_size: int, replica_number: int) -> tuple[float, float]:
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    vector_mode = getattr(model, "vector_mode", "dense")
    doc_ids = list(docs.keys())
    n_total = len(doc_ids)
    n_chunks = math.ceil(n_total / _ENCODE_CHUNK)

    t0 = time.time()
    point_id = 0

    for ci in range(n_chunks):
        s = ci * _ENCODE_CHUNK
        e = min(s + _ENCODE_CHUNK, n_total)
        texts = [
            f"{docs[d].get('title', '')} {docs[d]['chunk']}".strip()
            for d in doc_ids[s:e]
        ]
        embs = model.encode_docs(texts, batch_size)
        del texts

        batch: list[dict] = []
        for doc_id, vec in zip(doc_ids[s:e], embs):
            if vector_mode == "colbert":
                row = {"id": point_id, "doc_id": doc_id,
                       "colbert_embs": [{"emb": v.tolist()} for v in vec]}
            elif vector_mode == "sparse":
                row = {"id": point_id, "doc_id": doc_id,
                       "vector": {int(k): float(v) for k, v in vec.items()}}
            else:
                row = {"id": point_id, "doc_id": doc_id, "vector": vec.tolist()}
            batch.append(row)
            point_id += 1
            if len(batch) >= _INSERT_BATCH:
                client.insert(name, batch)
                batch.clear()
        if batch:
            client.insert(name, batch)

        del embs
        if has_cuda:
            import torch
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()
        print(f"  인코딩+인서트: {e:,}/{n_total:,}", flush=True)

    index_sec = time.time() - t0

    client.load_collection(name, replica_number=replica_number)
    print(f"  load_collection 완료 (replica={replica_number})", flush=True)

    return index_sec, (n_total / index_sec if index_sec > 0 else 0.0)


# ── 검색 (동시 worker 지원) ───────────────────────────────────────────────────

def _search_range(client, name: str, vectors, s: int, e: int,
                  top_k: int, vector_mode: str) -> list:
    CHUNK = 16 if vector_mode == "colbert" else 256
    partial: list = []

    for start in range(s, e, CHUNK):
        end = min(start + CHUNK, e)

        if vector_mode == "colbert":
            from pymilvus.client.embedding_list import EmbeddingList
            data = []
            for i in range(start, end):
                el = EmbeddingList()
                for tv in vectors[i]:
                    el.add(tv.tolist())
                data.append(el)
            res = client.search(name, data, anns_field="colbert_embs[emb]",
                                search_params={"metric_type": "MAX_SIM_COSINE"},
                                limit=top_k, output_fields=["doc_id"], timeout=120)
        elif vector_mode == "sparse":
            data = [{int(k): float(v) for k, v in vectors[i].items()}
                    for i in range(start, end)]
            res = client.search(name, data, anns_field="vector",
                                search_params={"metric_type": "IP", "params": {}},
                                limit=top_k, output_fields=["doc_id"])
        else:
            data = [vectors[i].tolist() for i in range(start, end)]
            res = client.search(name, data, anns_field="vector",
                                search_params={"metric_type": "COSINE", "params": {"ef": 100}},
                                limit=top_k, output_fields=["doc_id"])

        for hits in res:
            partial.append([(h["entity"]["doc_id"], h["distance"]) for h in hits])

    return partial


def _concurrent_search(client, name: str, vectors, top_k: int,
                        vector_mode: str, num_workers: int) -> tuple[list, float, float]:
    n = len(vectors)
    t0 = time.time()

    if num_workers <= 1:
        results = _search_range(client, name, vectors, 0, n, top_k, vector_mode)
    else:
        chunk = math.ceil(n / num_workers)
        ranges = [(w * chunk, min((w + 1) * chunk, n))
                  for w in range(num_workers) if w * chunk < n]
        with ThreadPoolExecutor(max_workers=len(ranges)) as ex:
            futures = [ex.submit(_search_range, client, name, vectors, s, e, top_k, vector_mode)
                       for s, e in ranges]
            results = []
            for f in futures:
                results.extend(f.result())

    elapsed = time.time() - t0
    qps = n / elapsed if elapsed > 0 else 0.0
    print(f"  검색 완료: {n:,}  workers={num_workers}  ({elapsed:.1f}s, {qps:.1f} q/s)",
          flush=True)
    return results, elapsed, qps


# ── 단일 시나리오 실행 ────────────────────────────────────────────────────────

def _safe_name(model_id: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", model_id.lower())[:180]


def run_scenario(
    uri: str,
    model_id: str,
    vector_mode: str,
    model_dtype: str,
    batch_size: int,
    docs: dict,
    queries: dict,
    qrels: dict,
    task_names: list[str],
    num_shards: int,
    replica_number: int,
    search_workers: int,
    out_dir: str,
) -> dict:
    tag = f"s{num_shards}_r{replica_number}_w{search_workers}"
    col_name = _safe_name(model_id) + f"_{vector_mode}_dist_{tag}"
    ckpt = os.path.join(out_dir, f"{col_name}.json")

    if os.path.exists(ckpt):
        print(f"[스킵] {tag} — 결과 존재: {ckpt}", flush=True)
        with open(ckpt, encoding="utf-8") as f:
            return json.load(f)

    from pymilvus import MilvusClient
    client = MilvusClient(uri=uri, timeout=300)

    print(f"\n{'='*60}")
    print(f"  shards={num_shards}  replica={replica_number}  workers={search_workers}")
    print(f"  컬렉션: {col_name}")
    print(f"{'='*60}")

    t_load = time.time()
    model = build_model(model_id, vector_mode=vector_mode, dtype=model_dtype)
    model_load_sec = round(time.time() - t_load, 2)
    print(f"  모델 로드: {model_load_sec}s", flush=True)

    n_docs = len(docs)
    index_build_sec = None
    index_docs_per_sec = None

    if client.has_collection(col_name):
        n_pts = int(client.get_collection_stats(col_name).get("row_count", 0))
        if n_pts == n_docs:
            print(f"  [캐시] 컬렉션 존재 ({n_pts:,}건) → load만 수행", flush=True)
            client.load_collection(col_name, replica_number=replica_number)
        else:
            print(f"  [재색인] 불완전 ({n_pts:,}/{n_docs:,})", flush=True)
            client.drop_collection(col_name)
            _create_collection(client, col_name, model.dim, vector_mode, num_shards)
            index_build_sec, index_docs_per_sec = _index_and_load(
                client, col_name, model, docs, batch_size, replica_number)
    else:
        _create_collection(client, col_name, model.dim, vector_mode, num_shards)
        index_build_sec, index_docs_per_sec = _index_and_load(
            client, col_name, model, docs, batch_size, replica_number)

    # 쿼리 인코딩
    q_ids = list(queries.keys())
    q_texts = [queries[qid] for qid in q_ids]
    print(f"  query 인코딩 ({len(q_ids):,}건)...", flush=True)
    t0 = time.time()
    q_embs = model.encode_queries(q_texts, batch_size)
    query_encode_sec = round(time.time() - t0, 2)
    query_encode_qps = round(len(q_ids) / query_encode_sec, 1)
    del q_texts
    gc.collect()

    # 검색
    raw, search_sec, search_qps = _concurrent_search(
        client, col_name, q_embs, top_k=100,
        vector_mode=vector_mode, num_workers=search_workers,
    )
    del q_embs
    gc.collect()

    run = {q_ids[i]: {doc_id: score for doc_id, score in hits}
           for i, hits in enumerate(raw)}
    metrics = evaluate(run, qrels)

    model.close()
    client.release_collection(col_name)

    result = {
        "model":              model_id,
        "vector_mode":        vector_mode,
        "datasets":           task_names,
        "batch_size":         batch_size,
        "model_dtype":        model_dtype,
        "num_shards":         num_shards,
        "replica_number":     replica_number,
        "search_workers":     search_workers,
        "model_load_sec":     model_load_sec,
        "index_build_sec":    round(index_build_sec, 2) if index_build_sec else None,
        "index_docs_per_sec": round(index_docs_per_sec, 1) if index_docs_per_sec else None,
        "query_encode_sec":   query_encode_sec,
        "query_encode_qps":   query_encode_qps,
        "search_sec":         round(search_sec, 2),
        "search_qps":         round(search_qps, 1),
        **metrics,
    }

    with open(ckpt, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  [저장] {ckpt}", flush=True)
    return result


# ── 비교 테이블 출력 ──────────────────────────────────────────────────────────

def _print_table(results: list[dict]) -> None:
    print(f"\n{'='*72}")
    print("  분산처리 성능 비교")
    print(f"{'='*72}")
    hdr = f"{'shards':>6} {'replica':>7} {'workers':>7} | {'idx docs/s':>10} {'qps':>8} {'NDCG@10':>8}"
    print(hdr)
    print("-" * 72)
    for r in results:
        idx = r.get("index_docs_per_sec")
        idx_s = f"{idx:.0f}" if idx else "cached"
        print(f"{r['num_shards']:>6} {r['replica_number']:>7} {r['search_workers']:>7} | "
              f"{idx_s:>10} {r['search_qps']:>8.1f} {r.get('ndcg_at_10', 0):>8.4f}")
    print("=" * 72)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Milvus 분산처리 성능 비교 테스트")
    ap.add_argument("--milvus-uri",      default=os.getenv("MILVUS_URI", "http://localhost:19530"))
    ap.add_argument("--data-root",       default=os.getenv("DATA_ROOT", "/workspace/datasets"))
    ap.add_argument("--model",           default=os.getenv("MODEL_ID", "BAAI/bge-m3"))
    ap.add_argument("--vector-mode",     default=os.getenv("VECTOR_MODE", "dense"),
                    choices=["dense", "sparse", "colbert"])
    ap.add_argument("--model-dtype",     default="auto",
                    choices=["auto", "fp32", "fp16", "bf16"])
    ap.add_argument("--batch-size",      type=int, default=16)
    ap.add_argument("--num-shards",      type=int, nargs="+", default=[1],
                    metavar="N", help="테스트할 num_shards 목록 (예: 1 2 4)")
    ap.add_argument("--replica-number",  type=int, nargs="+", default=[1],
                    metavar="N", help="테스트할 replica_number 목록 (예: 1 2 4)")
    ap.add_argument("--search-workers",  type=int, nargs="+", default=[1],
                    metavar="N", help="동시 검색 스레드 수 목록 (예: 1 2 4 8)")
    ap.add_argument("--out",             default="/workspace/reports/dist")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"[데이터] {args.data_root}")
    t0 = time.time()
    docs, queries, qrels = load_from_dir(args.data_root)
    task_names = [Path(args.data_root).name]
    print(f"  로드 완료 ({time.time()-t0:.0f}s)  docs={len(docs):,}  queries={len(queries):,}",
          flush=True)

    print(f"[설정]  shards={args.num_shards}  replica={args.replica_number}"
          f"  workers={args.search_workers}", flush=True)
    print(f"  총 {len(args.num_shards)*len(args.replica_number)*len(args.search_workers)}개 조합",
          flush=True)

    all_results: list[dict] = []
    for ns in args.num_shards:
        for rn in args.replica_number:
            for sw in args.search_workers:
                result = run_scenario(
                    uri=args.milvus_uri,
                    model_id=args.model,
                    vector_mode=args.vector_mode,
                    model_dtype=args.model_dtype,
                    batch_size=args.batch_size,
                    docs=docs, queries=queries, qrels=qrels,
                    task_names=task_names,
                    num_shards=ns, replica_number=rn, search_workers=sw,
                    out_dir=args.out,
                )
                all_results.append(result)

    summary = os.path.join(args.out, "dist_summary.json")
    with open(summary, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    _print_table(all_results)
    print(f"\n[저장] {summary}")


if __name__ == "__main__":
    main()
