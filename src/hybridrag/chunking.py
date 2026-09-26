"""Splits documents into fixed-size, non-overlapping word chunks, and maps
chunk-level retrieval results back to their parent document. BEIR's relevance
judgments are per-document, not per-chunk, so there's no ground truth to
score chunk-level hits against directly; a chunk hit only means something
once it's attributed back to the document it came from.
"""

from __future__ import annotations

_CHUNK_ID_SEPARATOR = "::chunk"


def chunk_text(text: str, chunk_size: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]


def chunk_corpus(corpus: dict[str, dict[str, str]], chunk_size: int) -> dict[str, dict[str, str]]:
    """Returns a new corpus where each document is split into chunk_size-word
    chunks, keyed by f"{doc_id}::chunk{i}". The title is repeated on every
    chunk, it's short and carries useful signal regardless of which chunk
    it's paired with.
    """
    chunked: dict[str, dict[str, str]] = {}
    for doc_id, doc in corpus.items():
        for i, chunk in enumerate(chunk_text(doc.get("text", ""), chunk_size)):
            chunked[f"{doc_id}{_CHUNK_ID_SEPARATOR}{i}"] = {"title": doc.get("title", ""), "text": chunk}
    return chunked


def parent_doc_id(chunk_id: str) -> str:
    return chunk_id.split(_CHUNK_ID_SEPARATOR)[0]


def dedupe_to_documents(hits: list[tuple[str, float]], top_k: int) -> list[tuple[str, float]]:
    """Collapses chunk-level hits down to one entry per parent document,
    keeping each document's best-scoring chunk, then returns the top_k
    highest-scoring documents.
    """
    best_per_doc: dict[str, float] = {}
    for chunk_id, score in hits:
        doc_id = parent_doc_id(chunk_id)
        if doc_id not in best_per_doc or score > best_per_doc[doc_id]:
            best_per_doc[doc_id] = score

    ranked = sorted(best_per_doc.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
