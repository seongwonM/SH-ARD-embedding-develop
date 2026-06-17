"""
한국어 임베딩 벤치마크 메인 실행 파일.

각 컴포넌트 교체 가이드:
  모델 교체:   bench/model.py 의 _QWEN3_MODELS / _FLASH_ATTN_MODELS 확인 후
               --model 인자만 바꾸면 됨. 완전히 다른 프레임워크면 EmbeddingModel 클래스 교체.
  VectorDB 교체: bench/vectordb.py 에 새 VectorStore 구현 후 build_store() 수정.
  데이터 교체: bench/data_prep.py 로 parquet 준비 후 --tasks 인자로 선택.
  평가 교체:   bench/evaluator.py 의 evaluate() 함수 교체.

usage:
  python -m bench.runner --data-root /workspace/datasets
  python -m bench.runner --model BAAI/bge-m3 --data-root /workspace/datasets
  python -m bench.runner --models BAAI/bge-m3 Qwen/Qwen3-Embedding-0.6B
  python -m bench.runner --qdrant-url http://localhost:6333
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
import time
from pathlib import Path

from bench.data_loader import load_from_dir
from bench.evaluator import evaluate
from bench.model import build_model
from bench.vectordb import QdrantStore, index_docs

_DEFAULT_MODELS = [
    "BAAI/bge-m3",
    "Qwen/Qwen3-Embedding-0.6B",
    "Qwen/Qwen3-Embedding-4B",
    "Qwen/Qwen3-Embedding-8B",
]


# ──────────────────────────────────────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────────────────────────────────────

def _safe_name(model_id: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", model_id.lower())[:200]


def _mem(label: str = "") -> None:
    parts = []
    try:
        import psutil
        parts.append(f"CPU={psutil.Process().memory_info().rss / 1e9:.1f}GB")
    except ImportError:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            parts.append(f"GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")
    except ImportError:
        pass
    if parts:
        tag = f"[MEM:{label}] " if label else "[MEM] "
        print(tag + " ".join(parts), flush=True)


def _json_default(obj):
    try:
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return str(obj)


# ──────────────────────────────────────────────────────────────────────────────
# VectorDB 팩토리 (교체 시 여기만 수정)
# ──────────────────────────────────────────────────────────────────────────────

def build_store(args) -> QdrantStore:
    """
    VectorDB 인스턴스 생성.
    다른 VectorDB로 교체하려면 이 함수에서 해당 클래스를 반환하도록 수정.
    """
    if args.qdrant_url:
        return QdrantStore(url=args.qdrant_url)
    qdrant_path = args.qdrant_path or os.path.join(args.out, "qdrant_storage")
    os.makedirs(qdrant_path, exist_ok=True)
    return QdrantStore(path=qdrant_path)


# ──────────────────────────────────────────────────────────────────────────────
# 단일 모델 실행
# ──────────────────────────────────────────────────────────────────────────────

def run_model(
    model_id:         str,
    store,
    combined_docs:    dict,
    combined_queries: dict,
    combined_qrels:   dict,
    task_names:       list[str],
    batch_size:       int,
    model_dtype:      str,
    vector_mode:      str = "dense",
) -> dict:
    mode_suffix = f"_{vector_mode}" if vector_mode != "dense" else ""
    collection = _safe_name(model_id) + mode_suffix

    print(f"\n{'='*64}")
    print(f"  모델: {model_id}  dtype={model_dtype}  vector_mode={vector_mode}")
    print(f"  컬렉션: {collection}")
    print(f"{'='*64}")

    t_load_start = time.time()
    model = build_model(model_id, vector_mode=vector_mode, dtype=model_dtype)
    model_load_sec = round(time.time() - t_load_start, 2)
    _mem("모델 로드")
    print(f"  모델 로드: {model_load_sec}s", flush=True)

    # 인덱스 상태 확인 및 인덱싱
    n_docs = len(combined_docs)
    index_build_sec: float | None = None
    index_docs_per_sec: float | None = None

    if store.has_collection(collection):
        n_pts = store.collection_size(collection)
        if n_pts == n_docs:
            print(f"  [스킵] 인덱스 존재 ({n_pts:,}건) → 검색으로 진행", flush=True)
        else:
            print(f"  [재색인] 불완전 ({n_pts:,}/{n_docs:,}) → 삭제 후 재색인", flush=True)
            store.drop_collection(collection)
            t0 = time.time()
            index_docs(store, collection, model, combined_docs, batch_size)
            index_build_sec = round(time.time() - t0, 2)
            index_docs_per_sec = round(n_docs / index_build_sec, 1)
            print(f"  인덱싱: {index_build_sec}s  ({index_docs_per_sec} docs/s)", flush=True)
    else:
        t0 = time.time()
        index_docs(store, collection, model, combined_docs, batch_size)
        index_build_sec = round(time.time() - t0, 2)
        index_docs_per_sec = round(n_docs / index_build_sec, 1)
        print(f"  인덱싱: {index_build_sec}s  ({index_docs_per_sec} docs/s)", flush=True)

    # query 인코딩
    q_ids   = list(combined_queries.keys())
    q_texts = [combined_queries[qid] for qid in q_ids]

    print(f"  query 인코딩 ({len(q_ids):,}건)...", flush=True)
    t0 = time.time()
    q_embs = model.encode_queries(q_texts, batch_size)
    query_encode_sec = round(time.time() - t0, 2)
    query_encode_qps = round(len(q_ids) / query_encode_sec, 1)
    del q_texts
    gc.collect()
    print(f"  query 인코딩: {query_encode_sec}s  ({query_encode_qps} queries/s)", flush=True)

    # 검색
    print(f"  검색 (top-100)...", flush=True)
    t0 = time.time()
    raw_results = store.search_batch(collection, q_embs, top_k=100, vector_mode=vector_mode)
    search_sec = round(time.time() - t0, 2)
    search_qps = round(len(q_ids) / search_sec, 1)
    del q_embs
    gc.collect()
    print(f"  검색: {search_sec}s  ({search_qps} queries/s)", flush=True)

    run = {q_ids[i]: {doc_id: score for doc_id, score in hits}
           for i, hits in enumerate(raw_results)}

    # 평가
    metrics = evaluate(run, combined_qrels)

    model.close()
    _mem("모델 해제 후")

    print(
        f"\n  NDCG@10={metrics.get('ndcg_at_10')}  @20={metrics.get('ndcg_at_20')}  "
        f"@50={metrics.get('ndcg_at_50')}  @100={metrics.get('ndcg_at_100')}"
    )
    print(
        f"  MRR@10={metrics.get('mrr_at_10')}  "
        f"Recall@10={metrics.get('recall_at_10')}  @50={metrics.get('recall_at_50')}  @100={metrics.get('recall_at_100')}"
    )

    return {
        "model":        model_id,
        "vector_mode":  vector_mode,
        "datasets":     task_names,
        "batch_size":   batch_size,
        "model_dtype":  model_dtype,
        # ── 성능 지표 ──────────────────────────────
        "model_load_sec":       model_load_sec,
        "index_build_sec":      index_build_sec,      # None = 캐시 재사용
        "index_docs_per_sec":   index_docs_per_sec,   # None = 캐시 재사용
        "query_encode_sec":     query_encode_sec,
        "query_encode_qps":     query_encode_qps,
        "search_sec":           search_sec,
        "search_qps":           search_qps,
        # ── 검색 품질 지표 ─────────────────────────
        **metrics,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="SentenceTransformer + Qdrant 한국어 벤치마크")

    # 모델
    mg = ap.add_mutually_exclusive_group()
    mg.add_argument("--model",  help="단일 모델 HuggingFace ID")
    mg.add_argument("--models", nargs="+", help="복수 모델 순차 실행")
    ap.add_argument("--model-dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    ap.add_argument("--vector-mode", default=os.getenv("VECTOR_MODE", "dense"),
                    choices=["dense", "sparse", "colbert"],
                    help="BGE-M3 vector 모드 (default: $VECTOR_MODE or 'dense')")
    ap.add_argument("--batch-size", type=int, default=64)

    # 데이터
    ap.add_argument("--data-root", default=os.getenv("DATA_ROOT", "/workspace/datasets"),
                    help="데이터셋 루트 경로. 하위 디렉터리를 자동 탐색 ($DATA_ROOT 또는 /workspace/datasets)")
    # --data-root 에 파일(bench.parquet) 또는 디렉터리를 직접 지정

    # VectorDB
    ap.add_argument("--qdrant-path", default=None, help="Qdrant on-disk 경로")
    ap.add_argument("--qdrant-url",  default=None, help="원격 Qdrant URL (e.g. http://localhost:6333)")

    # 출력
    ap.add_argument("--out", default="reports", help="결과 저장 디렉터리")

    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model_ids   = [args.model] if args.model else (args.models or _DEFAULT_MODELS)
    vector_mode = args.vector_mode or "dense"

    store = build_store(args)

    print(f"[데이터] {args.data_root}")
    print(f"[모델]   {model_ids}  vector_mode={vector_mode}")

    t0 = time.time()
    combined_docs, combined_queries, combined_qrels = load_from_dir(args.data_root)
    task_names = [Path(args.data_root).name]
    print(f"  로드 완료 ({time.time()-t0:.0f}s)", flush=True)
    _mem("corpus 병합 완료")

    all_results = []
    t0_total = time.time()

    for model_id in model_ids:
        mode_suffix = f"_{vector_mode}" if vector_mode != "dense" else ""
        ckpt = os.path.join(args.out, model_id.replace("/", "_") + mode_suffix + ".json")
        if os.path.exists(ckpt):
            print(f"\n[스킵] {model_id} ({vector_mode}) — 결과 존재: {ckpt}", flush=True)
            with open(ckpt, encoding="utf-8") as f:
                all_results.append(json.load(f))
            continue

        result = run_model(
            model_id, store,
            combined_docs, combined_queries, combined_qrels, task_names,
            args.batch_size, args.model_dtype,
            vector_mode=vector_mode,
        )
        all_results.append(result)
        with open(ckpt, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=_json_default)
        print(f"  [체크포인트] {ckpt}", flush=True)

    summary = os.path.join(args.out, "summary.json")
    with open(summary, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=_json_default)

    total = time.time() - t0_total
    print(f"\n{'='*64}")
    print(f"  완료: {len(all_results)}개 모델  ({total:.0f}s)")
    print(f"  저장: {summary}")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
