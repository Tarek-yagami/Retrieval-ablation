"""Reciprocal Rank Fusion: combines two ranked lists (sparse + dense) into one,
using each document's rank position rather than its raw score. Sparse (BM25)
and dense (cosine) scores live on incomparable scales, so fusing by rank
avoids the mismatch a naive weighted-score sum would run into.
"""

from __future__ import annotations

_DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: list[list[tuple[str, float]]],
    k: int = _DEFAULT_RRF_K,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
