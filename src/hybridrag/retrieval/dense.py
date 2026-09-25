"""Dense retrieval: embed the corpus once with a sentence-transformer, then
score queries against it with cosine similarity. Corpora at the scale used
here (BEIR's scifact/fiqa subsets) fit comfortably in memory, so this uses
plain matrix multiplication rather than an ANN index like FAISS. Exact
search is simpler and the extra dependency wouldn't earn its keep at this size.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from hybridrag.text import doc_text

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class DenseRetriever:
    def __init__(
        self,
        corpus: dict[str, dict[str, str]],
        model_name: str = _DEFAULT_MODEL,
        precomputed: tuple[list[str], np.ndarray] | None = None,
    ) -> None:
        """precomputed lets a caller hand over embeddings computed ahead of
        time (see build_playground_cache.py), instead of re-embedding the
        whole corpus on every process start. That one-time cost is cheap
        for the ablation experiments (run offline, once), but too heavy for
        a deployed demo's shared, limited CPU, where it happens on every
        cold start.
        """
        self._doc_ids = list(corpus.keys())

        self._model = SentenceTransformer(model_name)
        if precomputed is not None:
            cached_ids, embeddings = precomputed
            if cached_ids != self._doc_ids:
                raise ValueError("precomputed embeddings don't match this corpus's document order")
            self._embeddings = embeddings
        else:
            texts = [doc_text(corpus[d]) for d in self._doc_ids]
            embeddings = self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            self._embeddings = np.asarray(embeddings, dtype=np.float32)

    @property
    def precomputed(self) -> tuple[list[str], np.ndarray]:
        """This retriever's (doc_ids, embeddings), in the format __init__'s
        precomputed argument expects. Used to cache embeddings to disk once
        instead of re-embedding the corpus on every process start."""
        return self._doc_ids, self._embeddings

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        query_vec = self._model.encode([query], show_progress_bar=False, normalize_embeddings=True)[0]
        scores = self._embeddings @ query_vec
        top_idx = np.argsort(-scores)[:top_k]
        return [(self._doc_ids[i], float(scores[i])) for i in top_idx]
