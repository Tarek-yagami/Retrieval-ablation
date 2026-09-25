"""Cross-encoder reranking: scores each (query, candidate) pair jointly rather
than comparing independently-computed vectors, which catches relevance nuances
bi-encoder cosine similarity misses. Only worth running on the shortlist RRF
already narrowed down to, since cross-encoders are far more expensive per pair.
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from hybridrag.text import doc_text

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self._model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, float]],
        corpus: dict[str, dict[str, str]],
        top_k: int,
    ) -> list[tuple[str, float]]:
        doc_ids = [doc_id for doc_id, _ in candidates]
        pairs = [(query, doc_text(corpus[doc_id])) for doc_id in doc_ids]
        scores = self._model.predict(pairs)
        reranked = sorted(zip(doc_ids, scores, strict=True), key=lambda x: x[1], reverse=True)
        return [(doc_id, float(score)) for doc_id, score in reranked[:top_k]]
