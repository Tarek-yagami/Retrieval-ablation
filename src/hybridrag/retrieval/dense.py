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
    def __init__(self, corpus: dict[str, dict[str, str]], model_name: str = _DEFAULT_MODEL) -> None:
        self._doc_ids = list(corpus.keys())
        texts = [doc_text(corpus[d]) for d in self._doc_ids]

        self._model = SentenceTransformer(model_name)
        embeddings = self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        self._embeddings = np.asarray(embeddings, dtype=np.float32)

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        query_vec = self._model.encode([query], show_progress_bar=False, normalize_embeddings=True)[0]
        scores = self._embeddings @ query_vec
        top_idx = np.argsort(-scores)[:top_k]
        return [(self._doc_ids[i], float(scores[i])) for i in top_idx]
