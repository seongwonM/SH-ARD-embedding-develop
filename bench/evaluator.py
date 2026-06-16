"""
검색 결과 평가 모듈.

평가 지표 교체 방법:
  - _METRICS 세트에 pytrec_eval 지원 지표 추가/제거
  - 완전히 다른 평가 방식(e.g. LLM-as-judge)이 필요하면
    evaluate() 와 동일한 시그니처로 새 함수를 만들어 runner.py 에서 교체

지표:
  NDCG@10, MRR@10, Recall@1/5/10, MAP@10
  (binary relevance: score >= 1 → relevant)

"""
from __future__ import annotations

_METRICS = {"ndcg_cut.10", "recip_rank", "recall.1", "recall.5", "recall.10", "map_cut.10"}


def evaluate(
    run:   dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
) -> dict[str, float | None]:
    """
    Args:
      run   : {query_id: {doc_id: score}} — 검색 결과
      qrels : {query_id: {doc_id: score}} — 정답 레이블

    Returns:
      {ndcg_at_10, mrr_at_10, recall_at_1, recall_at_5, recall_at_10, map_at_10}
    """
    import pytrec_eval

    binary_qrels = {
        qid: {did: 1 for did, s in rels.items() if s >= 1}
        for qid, rels in qrels.items()
        if any(s >= 1 for s in rels.values())
    }

    evaluator = pytrec_eval.RelevanceEvaluator(binary_qrels, _METRICS)
    per_query = evaluator.evaluate(run)

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
