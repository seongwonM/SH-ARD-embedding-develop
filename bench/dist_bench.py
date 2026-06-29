"""
Milvus 분산처리 concurrent QPS 비교 테스트 (VDBBench 방식).

replica_number × concurrent_workers 조합별로 search_concurrent()를 실행해
지속 부하(sustained load) 하에서의 QPS / p50 / p95 / p99를 측정한다.

측정 방법론:
  - 각 worker 스레드가 독립 커넥션으로 1개짜리 쿼리를 duration 초 동안 반복 전송
  - replica 효과는 높은 concurrency에서 뚜렷하게 나타남
    (replica=1→2 @ workers=8: 이론상 ~2×  QPS 향상, 레이턴시 감소)

사용법:
  python -m bench.dist_bench \\
    --model BAAI/bge-m3 \\
    --replicas 1 2 \\
    --workers 1 2 4 8 \\
    --duration 30

환경변수 (startup_dist.sh 자동 전달):
  MODEL_ID, VECTOR_MODE, MILVUS_URI, DATA_ROOT, SEARCH_DURATION
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from bench.data_loader import load_from_dir
from bench.evaluator import evaluate
from bench.milvus import MilvusStore, index_docs as _milvus_index_docs
from bench.model import build_model


_GH_REPO = "seongwonM/SH-ARD-embedding-develop"


def _safe_name(model_id: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", model_id.lower())[:200]


def _fetch_baseline_from_github(model_id: str, vector_mode: str) -> float | None:
    """GitHub results/에서 가장 최근 Milvus rank0 결과를 내려받아 search_qps 반환.

    네이밍 룰 (startup.sh 기준):
      {model_safe}_{vector_mode}_milvus_rank0_{YYYYMMDD_HHMMSS}.json  ← 우선
      {model_safe}_{vector_mode}_milvus_{YYYYMMDD_HHMMSS}.json        ← fallback (standalone)
    """
    api_url = f"https://api.github.com/repos/{_GH_REPO}/contents/results"
    gh_token = os.getenv("GH_TOKEN")

    headers: dict[str, str] = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "embedding-bench",
    }
    if gh_token:
        headers["Authorization"] = f"token {gh_token}"

    req = urllib.request.Request(api_url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            entries = json.loads(resp.read())
    except urllib.error.URLError as e:
        print(f"[baseline] GitHub API 접근 실패: {e}", flush=True)
        return None

    model_safe = model_id.replace("/", "_")
    prefix = f"{model_safe}_{vector_mode}_milvus_"

    # (timestamp, filename, download_url, is_rank0)
    candidates: list[tuple[str, str, str, bool]] = []
    for entry in entries:
        name: str = entry["name"]
        if not name.startswith(prefix) or not name.endswith(".json"):
            continue
        rest = name[len(prefix):-5]
        if re.match(r"^\d{8}_\d{6}$", rest):
            candidates.append((rest, name, entry["download_url"], False))
        elif rest.startswith("rank0_") and re.match(r"^\d{8}_\d{6}$", rest[6:]):
            candidates.append((rest[6:], name, entry["download_url"], True))

    if not candidates:
        print(f"[baseline] GitHub results에 '{prefix}*.json' 없음", flush=True)
        return None

    candidates.sort(key=lambda x: (x[0], x[3]), reverse=True)
    ts, filename, url, _ = candidates[0]
    print(f"[baseline] GitHub 최신: {filename}", flush=True)

    req2 = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req2, timeout=30) as resp:
            data = json.loads(resp.read())
    except urllib.error.URLError as e:
        print(f"[baseline] 다운로드 실패: {e}", flush=True)
        return None

    if isinstance(data, list):
        data = data[0]
    qps = data.get("search_qps")
    print(f"[baseline] search_qps={qps}", flush=True)
    return qps


# ── 결과 테이블 출력 ──────────────────────────────────────────────────────────

def _print_table(baseline_qps: float | None, ndcg: float | None, rows: list[dict]) -> None:
    print(f"\n{'='*76}")
    print("  분산처리 concurrent QPS  (VDBBench 방식)")
    print(f"{'='*76}")

    if baseline_qps:
        print(f"  baseline (runner.py standalone sequential): {baseline_qps:.1f} q/s")
    if ndcg is not None:
        print(f"  NDCG@10 (sequential, replica=1): {ndcg:.4f}")
    print()

    print(f"  {'replica':>7} {'workers':>7} | {'qps':>8} {'vs base':>8} "
          f"{'p50ms':>7} {'p95ms':>7} {'p99ms':>7}")
    print("  " + "-" * 60)
    for r in rows:
        qps = r.get("search_qps")
        if qps is None:
            qps_str, ratio = "N/A", "-"
        else:
            qps_str = f"{qps:.1f}"
            ratio = f"{qps/baseline_qps:.2f}x" if baseline_qps else "-"
        p50 = r.get("p50_ms", "-")
        p95 = r.get("p95_ms", "-")
        p99 = r.get("p99_ms", "-")
        print(f"  {r['replica_number']:>7} {r['search_workers']:>7} | "
              f"{qps_str:>8} {ratio:>8} {str(p50):>7} {str(p95):>7} {str(p99):>7}")
    print("=" * 76)


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Milvus 분산처리 concurrent QPS 비교 테스트")
    ap.add_argument("--milvus-uri",  default=os.getenv("MILVUS_URI", "http://localhost:19530"))
    ap.add_argument("--data-root",   default=os.getenv("DATA_ROOT", "/workspace/datasets"))
    ap.add_argument("--model",       default=os.getenv("MODEL_ID", "BAAI/bge-m3"))
    ap.add_argument("--vector-mode", default=os.getenv("VECTOR_MODE", "dense"),
                    choices=["dense", "sparse", "colbert"])
    ap.add_argument("--batch-size",  type=int, default=int(os.getenv("BATCH_SIZE", "16")))
    ap.add_argument("--replicas",    type=int, nargs="+", default=[1, 2],
                    metavar="N", help="테스트할 replica_number 목록 (예: 1 2 4)")
    ap.add_argument("--workers",     type=int, nargs="+", default=[1, 2, 4],
                    metavar="N", help="동시 클라이언트 수 목록 (예: 1 2 4 8)")
    ap.add_argument("--duration",    type=int, default=int(os.getenv("SEARCH_DURATION", "30")),
                    help="concurrent 측정 지속 시간(초, default=30)")
    ap.add_argument("--baseline",    default=None,
                    help="베이스라인 summary.json 경로 (비교 출력용, 선택)")
    ap.add_argument("--out",         default="/workspace/reports/dist")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ── 베이스라인 로드 ──────────────────────────────────────────────────────
    baseline_qps: float | None = None
    if args.baseline:
        if os.path.exists(args.baseline):
            with open(args.baseline, encoding="utf-8") as f:
                bl = json.load(f)
            if isinstance(bl, list):
                bl = bl[0]
            baseline_qps = bl.get("search_qps")
            print(f"[baseline] {args.baseline}  search_qps={baseline_qps}", flush=True)
        else:
            print(f"[baseline] 파일 없음: {args.baseline}", flush=True)
    else:
        baseline_qps = _fetch_baseline_from_github(args.model, args.vector_mode)

    # ── 컬렉션 확인 ──────────────────────────────────────────────────────────
    col = _safe_name(args.model) + f"_{args.vector_mode}"
    print(f"[설정] uri={args.milvus_uri}  col={col}", flush=True)
    print(f"       replicas={args.replicas}  workers={args.workers}  duration={args.duration}s",
          flush=True)

    store = MilvusStore(uri=args.milvus_uri)

    if not store.has_collection(col):
        print(f"[인덱싱] 컬렉션 '{col}' 없음 — 자동 인덱싱 시작...", flush=True)
        combined_docs, _, _ = load_from_dir(args.data_root)
        idx_model = build_model(args.model, vector_mode=args.vector_mode, dtype="auto")
        t0 = time.time()
        _milvus_index_docs(store, col, idx_model, combined_docs, args.batch_size)
        print(f"[인덱싱] 완료 ({time.time()-t0:.0f}s)", flush=True)
        idx_model.close()
        del combined_docs
        gc.collect()

    print(f"[컬렉션] {col}  docs={store.collection_size(col):,}", flush=True)

    # ── 쿼리 인코딩 (1회) ────────────────────────────────────────────────────
    print(f"[데이터] {args.data_root}")
    _, queries, qrels = load_from_dir(args.data_root)
    q_ids   = list(queries.keys())
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

    # ── NDCG 측정 (품질 검증, 1회) ───────────────────────────────────────────
    print("\n[NDCG] sequential 검색으로 품질 측정 (replica=1)...", flush=True)
    print("  [load] 컬렉션 메모리 로드 중 (대용량 HNSW는 수 분 소요)...", flush=True)
    store.load_collection(col, replica_number=1, timeout=600)
    print("  [load] 완료", flush=True)
    raw = store.search_batch(col, q_embs, top_k=100, vector_mode=args.vector_mode)
    run_ndcg = {q_ids[i]: {doc_id: score for doc_id, score in hits}
                for i, hits in enumerate(raw)}
    ndcg_metrics = evaluate(run_ndcg, qrels)
    ndcg_val = ndcg_metrics.get("ndcg_at_10")
    print(f"  NDCG@10={ndcg_val:.4f}", flush=True)

    # ── replica × workers concurrent QPS 측정 ────────────────────────────────
    rows: list[dict] = []

    for replica in args.replicas:
        print(f"\n[load] replica_number={replica}  ...", flush=True)
        store._client.release_collection(col)
        try:
            store.load_collection(col, replica_number=replica, timeout=600)
        except Exception as e:
            if "resource insufficient" in str(e) or "service resource" in str(e):
                print(
                    f"  [스킵] replica={replica} — StreamingNode 부족으로 불가 "
                    f"(currentStreamingNode < {replica}). replica=1 결과만 사용.",
                    flush=True,
                )
                continue
            raise
        print(f"  load 완료", flush=True)

        for workers in args.workers:
            ckpt = os.path.join(args.out, f"{col}_r{replica}_w{workers}.json")
            if os.path.exists(ckpt):
                print(f"  [스킵] replica={replica} workers={workers} — 결과 존재", flush=True)
                with open(ckpt, encoding="utf-8") as f:
                    rows.append(json.load(f))
                continue

            print(f"  측정: replica={replica}  workers={workers}  {args.duration}s ...",
                  flush=True)
            stats = store.search_concurrent(
                col, q_embs, top_k=100,
                vector_mode=args.vector_mode,
                concurrency=workers,
                duration_sec=args.duration,
            )

            qps  = stats.get("qps")
            p50  = stats.get("p50_ms")
            p95  = stats.get("p95_ms")
            p99  = stats.get("p99_ms")

            if qps is not None:
                print(f"    → {qps:.1f} q/s  p50={p50}ms  p95={p95}ms  p99={p99}ms",
                      flush=True)
            else:
                print(f"    → {stats.get('note', '결과없음')}", flush=True)

            result = {
                "model":          args.model,
                "vector_mode":    args.vector_mode,
                "replica_number": replica,
                "search_workers": workers,
                "search_qps":     qps,
                "baseline_qps":   baseline_qps,
                "p50_ms":         p50,
                "p95_ms":         p95,
                "p99_ms":         p99,
                **ndcg_metrics,
            }
            with open(ckpt, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            rows.append(result)

    # ── 요약 저장 + 비교 출력 ────────────────────────────────────────────────
    summary = os.path.join(args.out, "dist_summary.json")
    with open(summary, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    _print_table(baseline_qps, ndcg_val, rows)
    print(f"\n[저장] {summary}")


if __name__ == "__main__":
    main()
