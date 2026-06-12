"""
HuggingFace datasets 라이브러리를 사용한 한국어 Retrieval 태스크 로더.

BEIR 표준 포맷과 MIRACL 포맷 두 가지를 명시적으로 처리.
반환 타입은 공통: corpus={id: {title, text}}, queries={id: text}, qrels={qid: {did: score}}
"""
from __future__ import annotations

from datasets import load_dataset

# ── 태스크별 HuggingFace 데이터셋 설정 ────────────────────────────────────────
# format:
#   "beir"   — 표준 BEIR 포맷: corpus / queries / {eval_split} 세 개의 split
#              corpus  columns: _id, title, text
#              queries columns: _id, text
#              qrels   columns: query-id, corpus-id, score
#   "miracl" — MIRACL 포맷: corpus split + eval split (positive_passages 내장)
#              corpus columns: docid, title, text
#              eval   columns: query_id, query, positive_passages, negative_passages

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


def _load_beir(path: str, config: str, eval_split: str) -> tuple[dict, dict, dict]:
    """
    BEIR 표준 포맷 로드.
    corpus / queries 는 고정 split 이름, qrels 는 eval_split.
    """
    print(f"    load_dataset({path!r}, {config!r}, split='corpus') ...", flush=True)
    corpus_ds  = load_dataset(path, config, split="corpus")
    queries_ds = load_dataset(path, config, split="queries")
    qrels_ds   = load_dataset(path, config, split=eval_split)

    corpus = {
        r["_id"]: {"title": r.get("title", ""), "text": r["text"]}
        for r in corpus_ds
    }
    queries = {r["_id"]: r["text"] for r in queries_ds}

    qrels: dict[str, dict[str, int]] = {}
    for r in qrels_ds:
        qid   = r["query-id"]
        did   = r["corpus-id"]
        score = int(r.get("score", 1))
        qrels.setdefault(qid, {})[did] = score

    return corpus, queries, qrels


def _load_miracl(path: str, config: str, eval_split: str) -> tuple[dict, dict, dict]:
    """
    MIRACL 포맷 로드.
    corpus split 에 전체 문서, eval split 에 쿼리 + positive/negative passages 내장.
    """
    print(f"    load_dataset({path!r}, {config!r}, split='corpus') ...", flush=True)
    corpus_ds = load_dataset(path, config, split="corpus")
    eval_ds   = load_dataset(path, config, split=eval_split)

    corpus = {
        r["docid"]: {"title": r.get("title", ""), "text": r["text"]}
        for r in corpus_ds
    }

    queries: dict[str, str] = {}
    qrels:   dict[str, dict[str, int]] = {}
    for r in eval_ds:
        qid = r["query_id"]
        queries[qid] = r["query"]
        qrels[qid] = {
            doc["docid"]: 1
            for doc in r.get("positive_passages", [])
        }
        # negative_passages (score=0)는 평가 시 무시 → 저장 안 함

    return corpus, queries, qrels


def load_task(task_name: str) -> tuple[dict, dict, dict]:
    """
    태스크 이름으로 (corpus, queries, qrels) 반환.
    지원 태스크: AutoRAGRetrieval, Ko-StrategyQA, LawIRKo,
               SQuADKorV1Retrieval, PublicHealthQA, MIRACLRetrieval
    """
    if task_name not in _TASK_CONFIGS:
        raise ValueError(
            f"지원하지 않는 태스크: {task_name}\n"
            f"지원 목록: {list(_TASK_CONFIGS)}"
        )

    cfg    = _TASK_CONFIGS[task_name]
    path   = cfg["path"]
    config = cfg["config"]
    split  = cfg["eval_split"]
    fmt    = cfg["format"]

    print(f"  [로딩] {task_name}  ({path}, {config}, {split})", flush=True)

    if fmt == "beir":
        corpus, queries, qrels = _load_beir(path, config, split)
    elif fmt == "miracl":
        corpus, queries, qrels = _load_miracl(path, config, split)
    else:
        raise ValueError(f"알 수 없는 포맷: {fmt}")

    print(
        f"  [완료] corpus={len(corpus):,}  "
        f"queries={len(queries):,}  qrels={len(qrels):,}",
        flush=True,
    )
    return corpus, queries, qrels


def load_combined(task_names: list[str]) -> tuple[dict, dict, dict, list[str]]:
    """
    여러 태스크를 '<태스크명>__' 접두어로 합산.
    반환: combined_corpus, combined_queries, combined_qrels, task_names
    """
    combined_corpus:  dict = {}
    combined_queries: dict = {}
    combined_qrels:   dict = {}

    for name in task_names:
        prefix = name + "__"
        corpus, queries, qrels = load_task(name)

        for did, doc in corpus.items():
            combined_corpus[prefix + did] = doc
        for qid, text in queries.items():
            combined_queries[prefix + qid] = text
        for qid, rels in qrels.items():
            combined_qrels[prefix + qid] = {prefix + did: s for did, s in rels.items()}

    print(
        f"\n[합산] corpus {len(combined_corpus):,}건 · "
        f"queries {len(combined_queries):,}건 · "
        f"qrel pairs {sum(len(v) for v in combined_qrels.values()):,}건",
        flush=True,
    )
    return combined_corpus, combined_queries, combined_qrels, task_names
