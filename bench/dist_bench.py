"""
Milvus 분산처리 QPS 비교 테스트.

bench/runner.py 가 만든 컬렉션을 재사용하고
replica_number × search_workers 조합별 search QPS를 측정해서
Standalone 기준값과 비교한다.

사용법 (RunPod coordinator 또는 standalone pod):
  python -m bench.dist_bench \\
    --model BAAI/bge-m3 \\
    --replicas 1 2 \\
    --workers 1 2 4

환경변수로도 지정 가능 (startup_dist.sh가 자동 전달):
  MODEL_ID, VECTOR_MODE, MILVUS_URI, DATA_ROOT
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


def _safe_name(model_id: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", model_id.lower())[:200]


# ── runner.py 와 동일한 검색 파라미터 ─────────────────────────────────────────

def _search_chunk(client, col: str, vectors, s: int, e: int,
                  top_k: int, vector_mode: str) -> list:
    """vectors[s:e] 순차 검색 — runner.py 와 동일한 파라미터."""
    CHUNK = 16 if vector_mode == "colbert" else 256
    out: list = []
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
            res = client.search(col, data, anns_field="colbert_embs[emb]",
                                search_params={"metric_type": "MAX_SIM_COSINE"},
                                limit=top_k, output_fields=["doc_id"], timeout=120)
        elif vector_mode == "sparse":
            data = [{int(k): float(v) for k, v in vectors[i].items()} for i in range(start, end)]
            res = client.search(col, data, anns_field="vector",
                                search_params={"metric_type": "IP", "params": {}},
                                limit=top_k, output_fields=["doc_id"])
        else:
            data = [vectors[i].tolist() for i in range(start, end)]
            res = client.search(col, data, anns_field="vector",
                                search_params={"metric_type": "COSINE", "params": {"ef": 100}},
                                limit=top_k, output_fields=["doc_id"])
        for hits in res:
            out.append([(h["entity"]["doc_id"], h["distance"]) for h in hits])
    return out


def run_search(client, col: str, vectors, top_k: int,
               vector_mode: str, num_workers: int) -> tuple[list, float]:
    """num_workers 개 스레드로 동시 검색, (results, qps) 반환."""
    n = len(vectors)
    t0 = time.time()

    if num_workers <= 1:
        results = _search_chunk(client, col, vectors, 0, n, top_k, vector_mode)
    else:
        chunk = math.ceil(n / num_workers)
        ranges = [(w * chunk, min((w + 1) * chunk, n))
                  for w in range(num_workers) if w * chunk < n]
        with ThreadPoolExecutor(max_workers=len(ranges)) as ex:
            futures = [ex.submit(_search_chunk, client, col, vectors, s, e, top_k, vector_mode)
                       for s, e in ranges]
            results = []
            for f in futures:
                results.extend(f.result())

    elapsed = time.time() - t0
    qps = n / elapsed if elapsed > 0 else 0.0
    return results, qps


# ── 결과 테이블 출력 ──────────────────────────────────────────────────────────

def _print_table(baseline_qps: float | None, rows: list[dict]) -> None:
    print(f"\n{'='*64}")
    print("  분산처리 QPS 비교  (baseline vs distributed)")
    print(f"{'='*64}")

    if baseline_qps:
        print(f"  baseline (standalone, replica=1, workers=1): {baseline_qps:.1f} q/s")
        print()

    print(f"  {'replica':>7} {'workers':>7} | {'qps':>8} {'vs base':>8} {'NDCG@10':>8}")
    print("  " + "-" * 46)
    for r in rows:
        ratio = f"{r['search_qps']/baseline_qps:.2f}x" if baseline_qps else "-"
        print(f"  {r['replica_number']:>7} {r['search_workers']:>7} | "
              f"{r['search_qps']:>8.1f} {ratio:>8} {r.get('ndcg_at_10', 0):>8.4f}")
    print("=" * 64)


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Milvus 분산처리 QPS 비교 테스트")
    ap.add_argument("--milvus-uri",   default=os.getenv("MILVUS_URI", "http://localhost:19530"))
    ap.add_argument("--data-root",    default=os.getenv("DATA_ROOT", "/workspace/datasets"))
    ap.add_argument("--model",        default=os.getenv("MODEL_ID", "BAAI/bge-m3"))
    ap.add_argument("--vector-mode",  default=os.getenv("VECTOR_MODE", "dense"),
                    choices=["dense", "sparse", "colbert"])
    ap.add_argument("--batch-size",   type=int, default=int(os.getenv("BATCH_SIZE", "16")))
    ap.add_argument("--replicas",     type=int, nargs="+", default=[1, 2],
                    metavar="N", help="테스트할 replica_number 목록 (예: 1 2 4)")
    ap.add_argument("--workers",      type=int, nargs="+", default=[1, 2, 4],
                    metavar="N", help="동시 검색 스레드 수 목록 (예: 1 2 4 8)")
    ap.add_argument("--baseline",     default=None,
                    help="베이스라인 summary.json 경로 (비교 출력용, 선택)")
    ap.add_argument("--out",          default="/workspace/reports/dist")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ── 베이스라인 로드 (비교용) ────────────────────────────────────────────
    baseline_qps: float | None = None
    if args.baseline and os.path.exists(args.baseline):
        with open(args.baseline, encoding="utf-8") as f:
            bl = json.load(f)
        if isinstance(bl, list):
            bl = bl[0]
        baseline_qps = bl.get("search_qps")
        print(f"[baseline] {args.baseline}  search_qps={baseline_qps}", flush=True)

    # ── 컬렉션 이름 (runner.py 와 동일 규칙) ───────────────────────────────
    col = _safe_name(args.model) + f"_{args.vector_mode}"
    print(f"[설정] uri={args.milvus_uri}  col={col}", flush=True)
    print(f"       replicas={args.replicas}  workers={args.workers}", flush=True)

    from pymilvus import MilvusClient
    client = MilvusClient(uri=args.milvus_uri, timeout=300)

    if not client.has_collection(col):
        print(f"[ERROR] 컬렉션 '{col}' 없음. 먼저 runner.py 로 baseline을 실행하세요.", flush=True)
        raise SystemExit(1)

    n_docs = int(client.get_collection_stats(col).get("row_count", 0))
    print(f"[컬렉션] {col}  docs={n_docs:,}", flush=True)

    # ── 쿼리 인코딩 (1회만) ────────────────────────────────────────────────
    print(f"[데이터] {args.data_root}")
    _, queries, qrels = load_from_dir(args.data_root)
    q_ids = list(queries.keys())
    q_texts = [queries[qid] for qid in q_ids]
    print(f"  queries={len(q_ids):,}", flush=True)

    model = build_model(args.model, vector_mode=args.vector_mode, dtype="auto")
    print("  query 인코딩 중...", flush=True)
    t0 = time.time()
    q_embs = model.encode_queries(q_texts, args.batch_size)
    print(f"  인코딩 완료 ({time.time()-t0:.1f}s)", flush=True)
    del q_texts
    gc.collect()
    model.close()

    # ── replica × workers 조합 측정 ────────────────────────────────────────
    rows: list[dict] = []

    for replica in args.replicas:
        print(f"\n[load] replica_number={replica}  ...", flush=True)
        client.release_collection(col)
        client.load_collection(col, replica_number=replica)
        print(f"  load 완료", flush=True)

        for workers in args.workers:
            ckpt = os.path.join(args.out, f"{col}_r{replica}_w{workers}.json")
            if os.path.exists(ckpt):
                print(f"  [스킵] replica={replica} workers={workers} — 결과 존재", flush=True)
                with open(ckpt, encoding="utf-8") as f:
                    rows.append(json.load(f))
                continue

            print(f"  검색: replica={replica}  workers={workers} ...", flush=True)
            raw, qps = run_search(client, col, q_embs, top_k=100,
                                  vector_mode=args.vector_mode, num_workers=workers)

            run = {q_ids[i]: {doc_id: score for doc_id, score in hits}
                   for i, hits in enumerate(raw)}
            metrics = evaluate(run, qrels)

            result = {
                "model":          args.model,
                "vector_mode":    args.vector_mode,
                "replica_number": replica,
                "search_workers": workers,
                "search_qps":     round(qps, 1),
                "baseline_qps":   baseline_qps,
                **metrics,
            }
            with open(ckpt, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            rows.append(result)
            print(f"    → {qps:.1f} q/s  NDCG@10={metrics.get('ndcg_at_10', 0):.4f}",
                  flush=True)

    # ── 요약 저장 + 비교 출력 ──────────────────────────────────────────────
    summary = os.path.join(args.out, "dist_summary.json")
    with open(summary, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    _print_table(baseline_qps, rows)
    print(f"\n[저장] {summary}")


if __name__ == "__main__":
    main()
