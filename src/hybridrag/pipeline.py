"""Wires the retrieval building blocks into the five fixed configs this
project benchmarks: sparse only, dense only, hybrid (RRF), hybrid+rerank, and
dense+rerank (reranking without fusing in the weaker retriever at all).
dense_rerank exists to test a claim, not to round out coverage: when hybrid
fusion drags a strong retriever down by fusing in a much weaker one (see
README), reranking-without-fusion is the natural alternative to check before
concluding that reranking-after-fusion was actually the fix.
Indexing (BM25 + embeddings) happens once per corpus and is reused across
every query and every config, since it's the expensive part.
"""

from __future__ import annotations

from hybridrag.retrieval.dense import DenseRetriever
from hybridrag.retrieval.fusion import reciprocal_rank_fusion
from hybridrag.retrieval.rerank import CrossEncoderReranker
from hybridrag.retrieval.sparse import SparseRetriever

CONFIGS = ["sparse", "dense", "hybrid", "hybrid_rerank", "dense_rerank"]

_SPARSE_TOP_K = 100
_DENSE_TOP_K = 100
_RERANK_SHORTLIST_MIN = 20  # rerank at least this many candidates, whatever top_k asks for


class RetrievalPipeline:
    """Indexes one corpus, then answers a query under any of CONFIGS."""

    def __init__(self, corpus: dict[str, dict[str, str]]) -> None:
        self._corpus = corpus
        self._sparse = SparseRetriever(corpus)
        self._dense = DenseRetriever(corpus)
        self._reranker: CrossEncoderReranker | None = None

    def _rerank_lazy(self) -> CrossEncoderReranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker()
        return self._reranker

    def search(self, query: str, config: str, top_k: int = 10) -> list[tuple[str, float]]:
        if config not in CONFIGS:
            raise ValueError(f"unknown config {config!r}, expected one of {CONFIGS}")

        if config == "sparse":
            return self._sparse.search(query, top_k)
        if config == "dense":
            return self._dense.search(query, top_k)

        shortlist_size = max(top_k, _RERANK_SHORTLIST_MIN)

        if config == "dense_rerank":
            shortlist = self._dense.search(query, shortlist_size)
            return self._rerank_lazy().rerank(query, shortlist, self._corpus, top_k)

        sparse_ranking = self._sparse.search(query, _SPARSE_TOP_K)
        dense_ranking = self._dense.search(query, _DENSE_TOP_K)
        fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])

        if config == "hybrid":
            return fused[:top_k]

        shortlist = fused[:shortlist_size]
        return self._rerank_lazy().rerank(query, shortlist, self._corpus, top_k)
