"""BM25 sparse retrieval over a fixed corpus."""

from __future__ import annotations

from rank_bm25 import BM25Okapi

from hybridrag.text import doc_text


class SparseRetriever:
    def __init__(self, corpus: dict[str, dict[str, str]]) -> None:
        self._doc_ids = list(corpus.keys())
        texts = [doc_text(corpus[d]) for d in self._doc_ids]
        self._bm25 = BM25Okapi([t.lower().split() for t in texts])

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        scores = self._bm25.get_scores(query.lower().split())
        ranked = sorted(zip(self._doc_ids, scores, strict=True), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
