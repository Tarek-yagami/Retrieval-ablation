"""Loads a BEIR benchmark dataset (corpus + queries + qrels) by name, downloading
and caching it under data/beir/ on first use. BEIR ships these as a standard
format specifically so retrieval systems can be scored against real relevance
judgments instead of hand-built eval sets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from beir import util
from beir.datasets.data_loader import GenericDataLoader

_BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"


@dataclass
class BeirDataset:
    name: str
    corpus: dict[str, dict[str, str]]  # doc_id -> {"title": ..., "text": ...}
    queries: dict[str, str]  # query_id -> query text
    qrels: dict[str, dict[str, int]]  # query_id -> {doc_id: relevance}


def load_beir_dataset(name: str, data_dir: str = "data/beir", split: str = "test") -> BeirDataset:
    dataset_dir = os.path.join(data_dir, name)
    if not os.path.exists(dataset_dir):
        os.makedirs(data_dir, exist_ok=True)
        zip_path = os.path.join(data_dir, f"{name}.zip")
        util.download_url(_BEIR_URL.format(name), zip_path)
        util.unzip(zip_path, data_dir)

    corpus, queries, qrels = GenericDataLoader(data_folder=dataset_dir).load(split=split)
    return BeirDataset(name=name, corpus=corpus, queries=queries, qrels=qrels)
