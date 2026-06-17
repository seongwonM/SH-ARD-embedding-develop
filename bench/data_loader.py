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

    print(
        f"[데이터] corpus={len(docs):,}  queries={len(queries):,}  "
        f"qrel_pairs={sum(len(v) for v in qrels.values()):,}",
        flush=True,
    )
    return docs, queries, qrels
