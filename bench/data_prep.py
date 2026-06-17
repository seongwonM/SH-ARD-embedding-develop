"""
한국어 Retrieval 데이터셋을 HuggingFace에서 다운로드해
보편적 스키마로 변환 후 로컬 디스크에 저장.

저장 스키마 (parquet):
  docs.parquet    : id(str), title(str), chunk(str)
  queries.parquet : id(str), query(str)
  qrels.parquet   : query_id(str), doc_id(str), score(int)

새 데이터셋 추가 방법:
  1. _TASK_CONFIGS 에 항목 추가
  2. format이 "beir" / "miracl" 이 아니면 _prep_* 함수 하나 추가 후 main() 분기에 연결

usage:
  python -m bench.data_prep --out /workspace/datasets
  python -m bench.data_prep --out /workspace/datasets --tasks AutoRAGRetrieval Ko-StrategyQA
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset

# ── 태스크별 HuggingFace 데이터셋 설정 ────────────────────────────────────────
# 새 데이터셋 추가 시 여기에만 추가하면 됨
_TASK_CONFIGS: dict[str, dict] = {
    "AutoRAGRetrieval": {
        "path":       "mteb/AutoRAGRetrieval",
        "config":     "default",
        "eval_split": "test",
        "format":     "beir",
    },
    "Ko-StrategyQA": {
        "path":       "mteb/Ko-StrategyQA",
        "config":     "default",
        "eval_split": "dev",
        "format":     "beir",
    },
    "LawIRKo": {
        "path":       "mteb/LawIRKo",
        "config":     "default",
        "eval_split": "test",
        "format":     "beir",
    },
    "SQuADKorV1Retrieval": {
        "path":       "mteb/SQuADKorV1Retrieval",
        "config":     "default",
        "eval_split": "test",
        "format":     "beir",
    },
    "PublicHealthQA": {
        "path":       "mteb/PublicHealthQA",
        "config":     "korean",
        "eval_split": "test",
        "format":     "beir",
    },
    "MIRACLRetrieval": {
        "path":       "miracl/miracl",
        "config":     "ko",
        "eval_split": "dev",
        "format":     "miracl",
    },
}


def _prep_beir(path: str, config: str | None, eval_split: str, out_dir: Path) -> None:
    """BEIR 표준 포맷: corpus / queries / {eval_split} 세 개의 split."""
    # "default" 는 config 없음과 동일 — 명시적으로 넘기면 오류 발생
    extra = {} if (not config or config == "default") else {"name": config}
    corpus_ds  = load_dataset(path, split="corpus",    trust_remote_code=True, **extra)
    queries_ds = load_dataset(path, split="queries",   trust_remote_code=True, **extra)
    qrels_ds   = load_dataset(path, split=eval_split,  trust_remote_code=True, **extra)

    pd.DataFrame([
        {"id": r["_id"], "title": r.get("title") or "", "chunk": r["text"]}
        for r in corpus_ds
    ]).to_parquet(out_dir / "docs.parquet", index=False)

    pd.DataFrame([
        {"id": r["_id"], "query": r["text"]}
        for r in queries_ds
    ]).to_parquet(out_dir / "queries.parquet", index=False)

    pd.DataFrame([
        {"query_id": r["query-id"], "doc_id": r["corpus-id"], "score": int(r.get("score", 1))}
        for r in qrels_ds
    ]).to_parquet(out_dir / "qrels.parquet", index=False)


def _prep_miracl(path: str, config: str | None, eval_split: str, out_dir: Path) -> None:
    """MIRACL 포맷: corpus split + eval split (positive_passages 내장)."""
    extra = {} if (not config or config == "default") else {"name": config}
    corpus_ds = load_dataset(path, split="corpus",    trust_remote_code=True, **extra)
    eval_ds   = load_dataset(path, split=eval_split,  trust_remote_code=True, **extra)

    pd.DataFrame([
        {"id": r["docid"], "title": r.get("title") or "", "chunk": r["text"]}
        for r in corpus_ds
    ]).to_parquet(out_dir / "docs.parquet", index=False)

    query_rows: list[dict] = []
    qrel_rows:  list[dict] = []
    for r in eval_ds:
        qid = r["query_id"]
        query_rows.append({"id": qid, "query": r["query"]})
        for doc in r.get("positive_passages", []):
            qrel_rows.append({"query_id": qid, "doc_id": doc["docid"], "score": 1})

    pd.DataFrame(query_rows).to_parquet(out_dir / "queries.parquet", index=False)
    pd.DataFrame(qrel_rows).to_parquet(out_dir / "qrels.parquet", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="한국어 Retrieval 데이터셋 다운로드 및 변환")
    ap.add_argument("--out",   default="/workspace/datasets", help="저장 루트 경로")
    ap.add_argument("--tasks", nargs="*", default=list(_TASK_CONFIGS),
                    help="변환할 태스크 (기본: 전체)")
    args = ap.parse_args()

    root = Path(args.out)
    for task in args.tasks:
        cfg     = _TASK_CONFIGS[task]
        out_dir = root / task

        if (out_dir / "docs.parquet").exists():
            print(f"[스킵] {task} — 이미 저장됨: {out_dir}")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[다운로드] {task}  ({cfg['path']}, {cfg['config']})...", flush=True)

        fmt = cfg["format"]
        if fmt == "beir":
            _prep_beir(cfg["path"], cfg["config"], cfg["eval_split"], out_dir)
        elif fmt == "miracl":
            _prep_miracl(cfg["path"], cfg["config"], cfg["eval_split"], out_dir)
        else:
            raise ValueError(f"지원하지 않는 포맷: {fmt} (task={task})")

        n = len(pd.read_parquet(out_dir / "docs.parquet"))
        print(f"  → {out_dir}  (docs={n:,})", flush=True)

    print(f"\n[완료] {root}")


if __name__ == "__main__":
    main()
