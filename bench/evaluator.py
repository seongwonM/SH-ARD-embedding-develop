"""
검색 결과 평가 모듈 — 외부 라이브러리 없이 직접 계산.

지표: NDCG@10, MRR@10, Recall@1/5/10, MAP@10
(binary relevance: score >= 1 → relevant)
"""
from __future__ import annotations
import math


def _dcg(ranked_docs: list[str], relevant: set[str], k: int) -> float:
    return sum(
        1.0 / math.log2(i + 2)
        for i, d in enumerate(ranked_docs[:k])
        if d in relevant
    )


def _ndcg(ranked_docs: list[str], relevant: set[str], k: int) -> float:
    dcg = _dcg(ranked_docs, relevant, k)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def _mrr(ranked_docs: list[str], relevant: set[str], k: int) -> float:
    for i, d in enumerate(ranked_docs[:k]):
        if d in relevant:
            return 1.0 / (i + 1)
    return 0.0


def _recall(ranked_docs: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for d in ranked_docs[:k] if d in relevant) / len(relevant)


def _map_at_k(ranked_docs: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits, precision_sum = 0, 0.0
    for i, d in enumerate(ranked_docs[:k]):
        if d in relevant:
            hits += 1
            precision_sum += hits / (i + 1)
    return precision_sum / len(relevant)


def evaluate(
    run:   dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
) -> dict[str, float | None]:
    """
    Args:
      run   : {query_id: {doc_id: score}}
      qrels : {query_id: {doc_id: score}}
    Returns:
      {ndcg_at_10, mrr_at_10, recall_at_1, recall_at_5, recall_at_10, map_at_10}
    """
    ndcg, mrr, r1, r5, r10, map10 = [], [], [], [], [], []

    for qid, docs_scores in run.items():
        if qid not in qrels:
            continue
        relevant = {did for did, s in qrels[qid].items() if s >= 1}
        if not relevant:
            continue

        ranked_docs = sorted(docs_scores, key=docs_scores.__getitem__, reverse=True)

        ndcg.append(_ndcg(ranked_docs, relevant, 10))
        mrr.append(_mrr(ranked_docs, relevant, 10))
        r1.append(_recall(ranked_docs, relevant, 1))
        r5.append(_recall(ranked_docs, relevant, 5))
        r10.append(_recall(ranked_docs, relevant, 10))
        map10.append(_map_at_k(ranked_docs, relevant, 10))

    def _avg(vals: list[float]) -> float | None:
        return round(sum(vals) / len(vals), 4) if vals else None

    return {
        "ndcg_at_10":   _avg(ndcg),
        "mrr_at_10":    _avg(mrr),
        "recall_at_1":  _avg(r1),
        "recall_at_5":  _avg(r5),
        "recall_at_10": _avg(r10),
        "map_at_10":    _avg(map10),
    }
