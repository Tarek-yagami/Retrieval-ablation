"""Owns the query playground's precomputed dense-embedding cache: building it
offline and loading it at runtime, so the Streamlit app never re-embeds the
playground's corpus on process start.

That embedding step is cheap to run once, offline, the same way this project
already does it once per dataset for the ablation experiments, but too heavy
for a free-tier deployment's shared, limited CPU, run fresh on every restart.
A deployed instance of app.py hit exactly that: visiting the query playground
triggered embedding scifact's ~5,000 documents live and got the app throttled.

Usage:
    uv run build-playground-cache
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hybridrag.data.beir_loader import load_beir_dataset
from hybridrag.retrieval.dense import DenseRetriever

PLAYGROUND_DATASET = "scifact"
CACHE_DIR = Path("data/playground_cache")


def _paths(dataset_name: str) -> tuple[Path, Path]:
    return (
        CACHE_DIR / f"{dataset_name}_embeddings.npy",
        CACHE_DIR / f"{dataset_name}_doc_ids.json",
    )


def build(dataset_name: str = PLAYGROUND_DATASET) -> None:
    print(f"[{dataset_name}] loading...")
    dataset = load_beir_dataset(dataset_name)

    print(f"[{dataset_name}] embedding {len(dataset.corpus)} docs...")
    retriever = DenseRetriever(dataset.corpus)
    doc_ids, embeddings = retriever.precomputed

    embeddings_path, doc_ids_path = _paths(dataset_name)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings)
    doc_ids_path.write_text(json.dumps(doc_ids))
    print(f"wrote {embeddings_path} and {doc_ids_path}")


def load(dataset_name: str = PLAYGROUND_DATASET) -> tuple[list[str], np.ndarray] | None:
    embeddings_path, doc_ids_path = _paths(dataset_name)
    if not (embeddings_path.exists() and doc_ids_path.exists()):
        return None
    return json.loads(doc_ids_path.read_text()), np.load(embeddings_path)


def main() -> None:
    build()


if __name__ == "__main__":
    main()
