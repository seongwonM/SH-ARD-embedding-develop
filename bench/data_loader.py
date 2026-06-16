"""
로컬 디스크 parquet → (docs, queries, qrels) 로더.
data_prep.py로 미리 변환된 파일을 읽음.

스키마:
  docs.parquet    : id(str), title(str), chunk(str)
  queries.parquet : id(str), query(str)
  qrels.parquet   : query_id(str), doc_id(str), score(int)

커스텀 데이터셋 사용법:
  위 스키마에 맞는 parquet 파일을 <data_root>/<task_name>/ 에 저장하면
  load_task() / load_combined() 로 바로 사용 가능.
  별도 코드 수정 불필요.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# data_prep.py가 지원하는 기본 태스크 목록
# 커스텀 태스크는 --tasks 인자로 직접 지정하면 됨
DEFAULT_TASK_NAMES: list[str] = [
    "AutoRAGRetrieval",
    "Ko-StrategyQA",
    "LawIRKo",
    "SQuADKorV1Retrieval",
    "PublicHealthQA",
    "MIRACLRetrieval",
]


def load_task(task_name: str, data_root: str | Path) -> tuple[dict, dict, dict]:
    """
    (docs, queries, qrels) 반환.

    docs    : {id: {"title": str, "chunk": str}}
    queries : {id: str}
    qrels   : {query_id: {doc_id: score}}
    """
    root = Path(data_root) / task_name
    if not (root / "docs.parquet").exists():
        raise FileNotFoundError(
            f"데이터 없음: {root}\n"
            f"먼저 실행: python -m bench.data_prep --out {data_root} --tasks {task_name}"
        )

    df_docs    = pd.read_parquet(root / "docs.parquet")
    df_queries = pd.read_parquet(root / "queries.parquet")
    df_qrels   = pd.read_parquet(root / "qrels.parquet")

    # to_dict("records")가 iterrows()보다 대용량에서 유의미하게 빠름
    docs = {
        r["id"]: {"title": r.get("title") or "", "chunk": r["chunk"]}
        for r in df_docs.to_dict("records")
    }
    queries = {r["id"]: r["query"] for r in df_queries.to_dict("records")}

    qrels: dict[str, dict[str, int]] = {}
    for r in df_qrels.to_dict("records"):
        qrels.setdefault(str(r["query_id"]), {})[str(r["doc_id"])] = int(r["score"])

    print(
        f"  [로딩] {task_name}: docs={len(docs):,}  "
        f"queries={len(queries):,}  qrels={len(qrels):,}",
        flush=True,
    )
    return docs, queries, qrels


def load_combined(
    task_names: list[str],
    data_root:  str | Path,
) -> tuple[dict, dict, dict, list[str]]:
    """
    여러 태스크를 '<태스크명>__' 접두어로 합산해 단일 대형 코퍼스 생성.

    반환: combined_docs, combined_queries, combined_qrels, task_names
    """
    combined_docs:    dict = {}
    combined_queries: dict = {}
    combined_qrels:   dict = {}

    for name in task_names:
        prefix = name + "__"
        docs, queries, qrels = load_task(name, data_root)

        for did, doc in docs.items():
            combined_docs[prefix + did] = doc
        for qid, text in queries.items():
            combined_queries[prefix + qid] = text
        for qid, rels in qrels.items():
            combined_qrels[prefix + qid] = {prefix + did: s for did, s in rels.items()}

    print(
        f"\n[합산] docs {len(combined_docs):,}건 · "
        f"queries {len(combined_queries):,}건 · "
        f"qrel pairs {sum(len(v) for v in combined_qrels.values()):,}건",
        flush=True,
    )
    return combined_docs, combined_queries, combined_qrels, task_names
