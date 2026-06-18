"""
범용 데이터 로더.

기본 포맷 (corpus_all / queries_all / qrels_all parquet):
  corpus_all.parquet  : _id, text, title, source
  queries_all.parquet : _id, text, source
  qrels_all.parquet   : query-id, corpus-id, score, source

반환 타입:
  docs    : {doc_id:   {"title": str, "chunk": str}}
  queries : {query_id: str}
  qrels   : {query_id: {doc_id: score}}
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_from_dir(data_dir: str | Path) -> tuple[dict, dict, dict]:
    d = Path(data_dir)

    corpus_df  = pd.read_parquet(d / "corpus_all.parquet")
    queries_df = pd.read_parquet(d / "queries_all.parquet")
    qrels_df   = pd.read_parquet(d / "qrels_all.parquet")

    docs = {
        str(r["_id"]): {"title": str(r.get("title") or ""), "chunk": str(r["text"])}
        for r in corpus_df.to_dict("records")
    }
    queries = {
        str(r["_id"]): str(r["text"])
        for r in queries_df.to_dict("records")
    }
    qrels: dict = {}
    for r in qrels_df.to_dict("records"):
        qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])

    total_pairs = sum(len(v) for v in qrels.values())
    relevant_pairs = sum(s >= 1 for v in qrels.values() for s in v.values())
    queries_with_rel = sum(1 for v in qrels.values() if any(s >= 1 for s in v.values()))
    rel_sizes = [sum(1 for s in v.values() if s >= 1) for v in qrels.values() if any(s >= 1 for s in v.values())]
    avg_rel = sum(rel_sizes) / len(rel_sizes) if rel_sizes else 0
    sample_qid = next(iter(qrels))
    sample_docs = list(qrels[sample_qid].items())[:3]
    print(
        f"[데이터] corpus={len(docs):,}  queries={len(queries):,}  "
        f"qrel_pairs={total_pairs:,}  relevant(score≥1)={relevant_pairs:,}",
        flush=True,
    )
    print(
        f"[데이터] queries_with_relevant={queries_with_rel:,}  avg_relevant_per_query={avg_rel:.2f}",
        flush=True,
    )
    print(f"[데이터] qrel sample qid={sample_qid!r} → {sample_docs}", flush=True)
    print(f"[데이터] corpus sample _id={next(iter(docs))!r}  query sample _id={next(iter(queries))!r}", flush=True)
    return docs, queries, qrels
