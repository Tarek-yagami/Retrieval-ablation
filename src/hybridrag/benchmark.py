"""Benchmark settings shared by every experiment: datasets, retrieval depth, query sample."""

from __future__ import annotations

from hybridrag.data.beir_loader import BeirDataset

DATASETS = ["scifact", "fiqa", "nfcorpus"]
MAX_QUERIES = 200
TOP_K = 100


def select_query_ids(dataset: BeirDataset, limit: int) -> list[str]:
    """Return the first `limit` judged query ids by sorted id, so every run scores the same queries."""
    return sorted(dataset.qrels.keys())[:limit]
