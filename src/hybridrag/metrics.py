"""Standard IR evaluation metrics via ranx, an established evaluation library
(used in IR research and pip-installable without a C-extension build step).
Hand-rolling nDCG/Recall/MRR is a well-known way to get subtly wrong numbers
(tie handling, log-base choice, cutoff semantics).
"""

from __future__ import annotations

from ranx import Qrels, Run
from ranx import evaluate as ranx_evaluate

_MEASURES = ["ndcg@10", "recall@100", "mrr@10"]


def evaluate(qrels: dict[str, dict[str, int]], run: dict[str, dict[str, float]]) -> dict[str, float]:
    """Scores a run (query_id -> {doc_id: score}) against ground-truth qrels.

    Returns the mean nDCG@10, Recall@100, and MRR@10 across all queries.
    """
    scores = ranx_evaluate(Qrels(qrels), Run(run), _MEASURES)
    return {measure: float(scores[measure]) for measure in _MEASURES}
