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
      NDCG@10/20/50/100, MRR@10, Recall@1/5/10/20/50/100, MAP@10/20/50/100
    """
    buckets: dict[str, list[float]] = {k: [] for k in [
        "ndcg_10", "ndcg_20", "ndcg_50", "ndcg_100",
        "mrr_10",
        "r1", "r5", "r10", "r20", "r50", "r100",
        "map_10", "map_20", "map_50", "map_100",
    ]}

    for qid, docs_scores in run.items():
        if qid not in qrels:
            continue
        relevant = {did for did, s in qrels[qid].items() if s >= 1}
        if not relevant:
            continue

        ranked_docs = sorted(docs_scores, key=docs_scores.__getitem__, reverse=True)

        for k in (10, 20, 50, 100):
            buckets[f"ndcg_{k}"].append(_ndcg(ranked_docs, relevant, k))
            buckets[f"map_{k}"].append(_map_at_k(ranked_docs, relevant, k))
        buckets["mrr_10"].append(_mrr(ranked_docs, relevant, 10))
        for k, key in ((1, "r1"), (5, "r5"), (10, "r10"), (20, "r20"), (50, "r50"), (100, "r100")):
            buckets[key].append(_recall(ranked_docs, relevant, k))

    def _avg(vals: list[float]) -> float | None:
        return round(sum(vals) / len(vals), 4) if vals else None

    return {
        "ndcg_at_10":    _avg(buckets["ndcg_10"]),
        "ndcg_at_20":    _avg(buckets["ndcg_20"]),
        "ndcg_at_50":    _avg(buckets["ndcg_50"]),
        "ndcg_at_100":   _avg(buckets["ndcg_100"]),
        "mrr_at_10":     _avg(buckets["mrr_10"]),
        "recall_at_1":   _avg(buckets["r1"]),
        "recall_at_5":   _avg(buckets["r5"]),
        "recall_at_10":  _avg(buckets["r10"]),
        "recall_at_20":  _avg(buckets["r20"]),
        "recall_at_50":  _avg(buckets["r50"]),
        "recall_at_100": _avg(buckets["r100"]),
        "map_at_10":     _avg(buckets["map_10"]),
        "map_at_20":     _avg(buckets["map_20"]),
        "map_at_50":     _avg(buckets["map_50"]),
        "map_at_100":    _avg(buckets["map_100"]),
    }
