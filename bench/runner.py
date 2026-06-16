"""
SentenceTransformer + Qdrant 기반 한국어 임베딩 벤치마크 러너.

embedding_bench(MTEB + numpy)의 Qdrant 대체 버전.
  - 데이터 로딩: HuggingFace datasets 직접 사용 (bench.data_loader)
  - 모델 서빙:   SentenceTransformer 직접 사용
  - 벡터 저장:   Qdrant on-disk — 배치 단위 upsert로 전체 벡터를 메모리에 올리지 않음
  - 검색:        Qdrant HNSW (코사인 유사도)
  - 평가:        pytrec_eval (embedding_bench와 동일 지표)

주요 장점 vs embedding_bench:
  - MIRACL 150만 건도 OOM 없이 처리 (배치 upsert)
  - Qdrant on-disk 인덱스 영속성 → pod 재시작 후 이미 인덱싱된 모델은 재색인 스킵

usage:
  python -m bench.runner --model BAAI/bge-m3
  python -m bench.runner --models BAAI/bge-m3 Qwen/Qwen3-0.6B
  python -m bench.runner --model BAAI/bge-m3 --qdrant-url http://localhost:6333
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
import warnings

import mteb

from bench.data_loader import load_combined, _TASK_CONFIGS

_DEFAULT_MODELS = [
    "BAAI/bge-m3",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
]

_KO_RETRIEVAL_TASKS = list(_TASK_CONFIGS.keys())

_DTYPE_MAP = {"auto": None, "fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}

_UPSERT_CHUNK = 2_000


# ──────────────────────────────────────────────────────────────────────────────
# 메모리 리포트 헬퍼
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# 데이터 로딩 (MTEB — 포맷 처리 검증됨)
# ──────────────────────────────────────────────────────────────────────────────

def _load_task_data(task) -> tuple[dict, dict, dict]:
    """
    MTEB 2.15 신규 포맷 / 구 BEIR 포맷 모두 지원.
    반환: corpus={id: {title, text}}, queries={id: text}, qrels={qid: {did: score}}
    """
    split = task.metadata.eval_splits[0]

    try:
        task._data_loaded = False
    except Exception:
        pass

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            task.load_data(eval_splits=[split])
        except TypeError:
            task.load_data()

    # ── 신규 포맷 (MTEB 2.15+) ───────────────────────────────────────────────
    ds = getattr(task, "dataset", None)
    if ds is not None and hasattr(ds, "keys"):
        config   = list(ds.keys())[0]
        cfg_data = ds[config]
        inner    = cfg_data.get(split, {}) if hasattr(cfg_data, "get") else cfg_data[split]
        if isinstance(inner, dict) and "corpus" in inner:
            corpus  = {r["id"]: {"title": r.get("title", ""), "text": r["text"]}
                       for r in inner["corpus"]}
            queries = {r["id"]: r["text"] for r in inner["queries"]}
            qrels   = dict(inner.get("relevant_docs", {}))
            print(f"    [신규포맷] {task.metadata.name}: "
                  f"corpus={len(corpus):,}  queries={len(queries):,}", flush=True)
            return corpus, queries, qrels

    # ── 구 포맷 (BEIR — PublicHealthQA) ──────────────────────────────────────
    corpus_attr = getattr(task, "corpus", None)
    if corpus_attr:
        lang_key  = list(corpus_attr.keys())[0]
        split_map = corpus_attr[lang_key]
        split_key = split if split in split_map else list(split_map.keys())[0]

        corpus_raw = split_map[split_key]
        corpus = (corpus_raw if isinstance(corpus_raw, dict)
                  else {r["_id"]: {"title": r.get("title", ""), "text": r["text"]}
                        for r in corpus_raw})

        queries_raw = (getattr(task, "queries", {}) or {}).get(lang_key, {}).get(split_key, {})
        queries = (queries_raw if isinstance(queries_raw, dict)
                   else {r["_id"]: r["text"] for r in queries_raw})

        qrels = ((getattr(task, "relevant_docs", {}) or {})
                 .get(lang_key, {}).get(split_key, {})) or {}

        print(f"    [구포맷] {task.metadata.name}: "
              f"corpus={len(corpus):,}  queries={len(queries):,}", flush=True)
        return corpus, queries, qrels

    raise ValueError(f"{task.metadata.name}: 데이터 로드 실패")


def _build_combined_corpus(tasks: list) -> tuple[dict, dict, dict, list[str]]:
    """
    모든 태스크의 corpus/queries/qrels를 '<태스크명>__' 접두어로 합산.
    반환: combined_corpus, combined_queries, combined_qrels, task_names
    """
    combined_corpus:  dict = {}
    combined_queries: dict = {}
    combined_qrels:   dict = {}

    for task in tasks:
        name   = task.metadata.name
        prefix = name + "__"
        corpus, queries, qrels = _load_task_data(task)
        print(f"  [로딩] {name}: corpus={len(corpus):,}  queries={len(queries):,}", flush=True)

        for did, doc in corpus.items():
            combined_corpus[prefix + did] = doc
        for qid, text in queries.items():
            combined_queries[prefix + qid] = text
        for qid, rels in qrels.items():
            combined_qrels[prefix + qid] = {prefix + did: s for did, s in rels.items()}

    print(
        f"  [합산] corpus {len(combined_corpus):,}건 · "
        f"queries {len(combined_queries):,}건 · "
        f"qrel pairs {sum(len(v) for v in combined_qrels.values()):,}건",
        flush=True,
    )
    return combined_corpus, combined_queries, combined_qrels, [t.metadata.name for t in tasks]


# ──────────────────────────────────────────────────────────────────────────────
# Qdrant 헬퍼
# ──────────────────────────────────────────────────────────────────────────────

def _safe_collection_name(model_id: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", model_id.lower())[:200]


def _collection_exists(client, name: str) -> bool:
    return name in {c.name for c in client.get_collections().collections}


def _index_corpus(
    client,
    collection_name: str,
    model,
    corpus: dict,
    batch_size: int,
) -> None:
    """
    corpus를 _UPSERT_CHUNK 단위로 인코딩해 Qdrant에 upsert.
    전체 벡터를 한꺼번에 메모리에 올리지 않아 MIRACL 150만 건도 처리 가능.
    """
    from qdrant_client.models import Distance, VectorParams, PointStruct
    import torch

    doc_ids = list(corpus.keys())
    n_total = len(doc_ids)

    # 임베딩 차원 파악
    sample_text = f"{corpus[doc_ids[0]].get('title', '')} {corpus[doc_ids[0]]['text']}".strip()
    embed_dim   = model.encode([sample_text], show_progress_bar=False).shape[1]

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embed_dim, distance=Distance.COSINE),
    )
    print(f"  Qdrant 컬렉션 생성: {collection_name}  dim={embed_dim}", flush=True)

    n_chunks = math.ceil(n_total / _UPSERT_CHUNK)
    _log_every = max(1, n_chunks // 10)
    for ci in range(n_chunks):
        s = ci * _UPSERT_CHUNK
        e = min(s + _UPSERT_CHUNK, n_total)
        chunk_ids   = doc_ids[s:e]
        chunk_texts = [
            f"{corpus[d].get('title', '')} {corpus[d]['text']}".strip()
            for d in chunk_ids
        ]

        embs = model.encode(
            chunk_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(id=s + i, vector=embs[i].tolist(), payload={"doc_id": chunk_ids[i]})
                for i in range(len(chunk_ids))
            ],
        )

        del embs, chunk_texts
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if (ci + 1) % _log_every == 0 or ci + 1 == n_chunks:
            print(f"  upsert {e:,}/{n_total:,}", flush=True)

    print(f"  인덱싱 완료: {n_total:,}건 → {collection_name}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# 검색 + 평가
# ──────────────────────────────────────────────────────────────────────────────

def _search_and_evaluate(
    client,
    collection_name: str,
    model,
    queries: dict,
    qrels:   dict,
    batch_size: int,
    top_k: int = 100,
) -> dict:
    import pytrec_eval

    q_ids   = list(queries.keys())
    q_texts = [queries[qid] for qid in q_ids]

    print(f"  query 인코딩 ({len(q_ids):,}건)...", flush=True)
    q_embs = model.encode(
        q_texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    del q_texts
    gc.collect()

    print(f"  Qdrant 검색 ({len(q_ids):,} queries, top-{top_k})...", flush=True)
    run: dict = {}
    SEARCH_CHUNK = 512
    for start in range(0, len(q_ids), SEARCH_CHUNK):
        end = min(start + SEARCH_CHUNK, len(q_ids))
        batch_results = client.search_batch(
            collection_name=collection_name,
            requests=[
                {"vector": q_embs[i].tolist(), "limit": top_k, "with_payload": True}
                for i in range(start, end)
            ],
        )
        for i, results in enumerate(batch_results):
            run[q_ids[start + i]] = {r.payload["doc_id"]: r.score for r in results}

    del q_embs
    gc.collect()

    binary_qrels = {
        qid: {did: 1 for did, s in rels.items() if s >= 1}
        for qid, rels in qrels.items()
        if any(s >= 1 for s in rels.values())
    }
    evaluator = pytrec_eval.RelevanceEvaluator(
        binary_qrels,
        {"ndcg_cut.10", "recip_rank", "recall.1", "recall.5", "recall.10", "map_cut.10"},
    )
    per_query = evaluator.evaluate(run)
    del run, binary_qrels
    gc.collect()

    def _avg(key: str) -> float | None:
        vals = [v.get(key, 0.0) for v in per_query.values()]
        return round(sum(vals) / len(vals), 4) if vals else None

    return {
        "ndcg_at_10":   _avg("ndcg_cut_10"),
        "mrr_at_10":    _avg("recip_rank"),
        "recall_at_1":  _avg("recall_1"),
        "recall_at_5":  _avg("recall_5"),
        "recall_at_10": _avg("recall_10"),
        "map_at_10":    _avg("map_cut_10"),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 모델별 실행
# ──────────────────────────────────────────────────────────────────────────────

def _run_model(
    model_id:         str,
    client,
    combined_corpus:  dict,
    combined_queries: dict,
    combined_qrels:   dict,
    task_names:       list[str],
    batch_size:       int,
    model_dtype:      str = "auto",
) -> dict:
    from sentence_transformers import SentenceTransformer
    import torch

    collection_name = _safe_collection_name(model_id)

    print(f"\n{'='*64}")
    print(f"  모델: {model_id}  (dtype={model_dtype})")
    print(f"  Qdrant 컬렉션: {collection_name}")
    print(f"{'='*64}")

    dtype = _DTYPE_MAP.get(model_dtype)
    model = SentenceTransformer(
        model_id,
        model_kwargs={"torch_dtype": getattr(torch, dtype)} if dtype else None,
    )
    _mem("로드")

    if _collection_exists(client, collection_name):
        n_pts = client.get_collection(collection_name).points_count
        print(f"  [스킵] 인덱스 이미 존재 ({n_pts:,}건) → 검색으로 진행", flush=True)
    else:
        print(f"  corpus 인덱싱 ({len(combined_corpus):,}건)...", flush=True)
        t0 = time.time()
        _index_corpus(client, collection_name, model, combined_corpus, batch_size)
        print(f"  인덱싱 완료 ({time.time()-t0:.0f}s)", flush=True)

    t0 = time.time()
    metrics = _search_and_evaluate(
        client, collection_name, model,
        combined_queries, combined_qrels, batch_size,
    )
    elapsed = time.time() - t0

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _mem("모델 해제 후")

    print(
        f"\n  NDCG@10={metrics.get('ndcg_at_10')}  "
        f"MRR@10={metrics.get('mrr_at_10')}  "
        f"Recall@10={metrics.get('recall_at_10')}  "
        f"({elapsed:.0f}s)"
    )

    return {
        "model":  model_id,
        "task":   "CombinedKoreanRetrieval",
        "tasks":  task_names,
        **metrics,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="SentenceTransformer + Qdrant 한국어 벤치마크")
    model_group = ap.add_mutually_exclusive_group()
    model_group.add_argument("--model",  help="단일 모델 HuggingFace ID")
    model_group.add_argument("--models", nargs="+",
                             help=f"복수 모델 순차 실행 (기본: {' '.join(_DEFAULT_MODELS)})")
    ap.add_argument("--tasks", nargs="*", default=_KO_RETRIEVAL_TASKS)
    ap.add_argument("--out",         default="reports")
    ap.add_argument("--batch-size",  type=int, default=64)
    ap.add_argument("--model-dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    ap.add_argument("--qdrant-path", default=None,
                    help="Qdrant on-disk 경로 (기본: {out}/qdrant_storage)")
    ap.add_argument("--qdrant-url",  default=None,
                    help="원격 Qdrant 서버 URL (예: http://localhost:6333)")
    args = ap.parse_args()

    try:
        from qdrant_client import QdrantClient
    except ImportError:
        sys.exit("qdrant-client 패키지가 필요합니다: pip install qdrant-client")

    os.makedirs(args.out, exist_ok=True)
    model_ids = [args.model] if args.model else (args.models or _DEFAULT_MODELS)

    if args.qdrant_url:
        client = QdrantClient(url=args.qdrant_url)
        print(f"[Qdrant] 원격 서버: {args.qdrant_url}", flush=True)
    else:
        qdrant_path = args.qdrant_path or os.path.join(args.out, "qdrant_storage")
        os.makedirs(qdrant_path, exist_ok=True)
        client = QdrantClient(path=qdrant_path)
        print(f"[Qdrant] on-disk: {qdrant_path}", flush=True)

    tasks = mteb.get_tasks(
        tasks=args.tasks,
        languages=["kor"],
        modalities=["text"],
        exclusive_modality_filter=True,
    )
    print(f"[태스크] {len(tasks)}개: {[t.metadata.name for t in tasks]}")
    print(f"[모델]   {len(model_ids)}개: {', '.join(model_ids)}")

    print(f"\n[corpus 병합] {len(tasks)}개 태스크 (1회 로딩)...", flush=True)
    t0_merge = time.time()
    combined_corpus, combined_queries, combined_qrels, task_names = _build_combined_corpus(tasks)
    print(f"  병합 완료 ({time.time()-t0_merge:.0f}s)", flush=True)
    _mem("corpus 병합 완료")

    def _json_default(obj):
        try:
            f = float(obj)
            return None if math.isnan(f) or math.isinf(f) else f
        except (TypeError, ValueError):
            return str(obj)

    all_results = []
    t0_total    = time.time()

    for model_id in model_ids:
        ckpt_path = os.path.join(args.out, model_id.replace("/", "_") + ".json")
        if os.path.exists(ckpt_path):
            print(f"\n[스킵] {model_id} — 결과 이미 존재: {ckpt_path}", flush=True)
            with open(ckpt_path, encoding="utf-8") as f:
                all_results.append(json.load(f))
            continue

        result = _run_model(
            model_id, client,
            combined_corpus, combined_queries, combined_qrels, task_names,
            args.batch_size, args.model_dtype,
        )
        if result:
            all_results.append(result)
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=_json_default)
            print(f"  [체크포인트] {ckpt_path}", flush=True)

    summary_path = os.path.join(args.out, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=_json_default)

    total = time.time() - t0_total
    print(f"\n{'='*64}")
    print(f"  전체 완료: {len(all_results)}개 모델  ({total:.0f}s)")
    print(f"  저장: {summary_path}")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
