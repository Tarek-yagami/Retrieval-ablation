"""RQ4: does rewriting the query with an LLM (HyDE, multi-query) improve dense retrieval?

The two phases run as separate commands so a local LLM and the retrieval models
are never in memory at the same time. Both resume where they stopped.

Usage:
    python experiments/query_rewrite_ablation/run.py rewrite   # needs HYBRIDRAG_MODEL
    python experiments/query_rewrite_ablation/run.py retrieve
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hybridrag.benchmark import DATASETS, TOP_K, select_query_ids  # noqa: E402
from hybridrag.data.beir_loader import BeirDataset, load_beir_dataset  # noqa: E402
from hybridrag.metrics import evaluate  # noqa: E402
from hybridrag.query_rewrite import paraphrase_query, write_hypothetical_passage  # noqa: E402
from hybridrag.retrieval.dense import DenseRetriever  # noqa: E402
from hybridrag.retrieval.fusion import reciprocal_rank_fusion  # noqa: E402
from hybridrag.retrieval.rerank import CrossEncoderReranker  # noqa: E402

QUERY_LIMIT = 50
PARAPHRASE_COUNT = 3
# config -> which cached rewrite timing it pays for, None for the baselines
CONFIGS = {
    "dense": None,
    "dense_rerank": None,
    "dense_hyde": "hyde_ms",
    "dense_hyde_rerank": "hyde_ms",
    "dense_multiquery": "paraphrase_ms",
}
RESULTS_DIR = Path(__file__).parent / "results"
REWRITES_PATH = RESULTS_DIR / "rewrites.json"
RESULTS_PATH = RESULTS_DIR / "results.json"


class RewriteRetriever:
    """Dense retrieval over one corpus under each of CONFIGS, given a query's cached rewrites."""

    def __init__(self, corpus: dict[str, dict[str, str]]) -> None:
        self._corpus = corpus
        self._dense = DenseRetriever(corpus)
        self._reranker = CrossEncoderReranker()

    def search(self, query: str, rewrite: dict, config: str) -> list[tuple[str, float]]:
        hyde_text = f"{query}\n{rewrite['hyde_passage']}"
        if config == "dense":
            return self._dense.search(query, TOP_K)
        if config == "dense_rerank":
            return self._reranker.rerank(query, self._dense.search(query, TOP_K), self._corpus, TOP_K)
        if config == "dense_hyde":
            return self._dense.search(hyde_text, TOP_K)
        if config == "dense_hyde_rerank":
            # rerank against the original query, so the reranker judges the user's question
            return self._reranker.rerank(query, self._dense.search(hyde_text, TOP_K), self._corpus, TOP_K)
        if config == "dense_multiquery":
            rankings = [self._dense.search(q, TOP_K) for q in [query, *rewrite["paraphrases"]]]
            return reciprocal_rank_fusion(rankings)[:TOP_K]
        raise ValueError(f"unknown config {config!r}, expected one of {list(CONFIGS)}")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_json(path: Path, data: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def rewrite_query(query: str) -> dict:
    """Return the query's HyDE passage and paraphrases, with how long each LLM call took."""
    start = time.perf_counter()
    hyde_passage = write_hypothetical_passage(query)
    hyde_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    paraphrases = paraphrase_query(query, PARAPHRASE_COUNT)
    paraphrase_ms = (time.perf_counter() - start) * 1000

    return {
        "hyde_passage": hyde_passage,
        "hyde_ms": hyde_ms,
        "paraphrases": paraphrases,
        "paraphrase_ms": paraphrase_ms,
    }


def run_rewrite_phase() -> None:
    rewrites = read_json(REWRITES_PATH)
    for name in DATASETS:
        dataset = load_beir_dataset(name)
        done = rewrites.setdefault(name, {})
        query_ids = select_query_ids(dataset, QUERY_LIMIT)
        for i, qid in enumerate(query_ids, start=1):
            if qid in done:
                continue
            print(f"[{name} rewrite {i}/{len(query_ids)}] {dataset.queries[qid][:60]!a}")
            done[qid] = rewrite_query(dataset.queries[qid])
            write_json(REWRITES_PATH, rewrites)
    print(f"wrote {REWRITES_PATH}")


def score_config(
    retriever: RewriteRetriever,
    dataset: BeirDataset,
    rewrites: dict[str, dict],
    config: str,
) -> dict[str, float]:
    query_ids = list(rewrites)
    # warm-up query, so model loading isn't timed
    retriever.search(dataset.queries[query_ids[0]], rewrites[query_ids[0]], config)

    run: dict[str, dict[str, float]] = {}
    start = time.perf_counter()
    for qid in query_ids:
        run[qid] = dict(retriever.search(dataset.queries[qid], rewrites[qid], config))
    search_ms = (time.perf_counter() - start) / len(query_ids) * 1000

    timing_key = CONFIGS[config]
    rewrite_ms = sum(rewrites[qid][timing_key] for qid in query_ids) / len(query_ids) if timing_key else 0.0

    scores = evaluate({qid: dataset.qrels[qid] for qid in query_ids}, run)
    return {**scores, "rewrite_ms": rewrite_ms, "search_ms": search_ms}


def run_retrieve_phase() -> None:
    rewrites = read_json(REWRITES_PATH)
    results = read_json(RESULTS_PATH)
    for name in DATASETS:
        dataset_results = results.setdefault(name, {})
        missing = [c for c in CONFIGS if c not in dataset_results]
        if not missing:
            print(f"[{name}] all configs already computed, skipping")
            continue

        dataset = load_beir_dataset(name)
        expected = select_query_ids(dataset, QUERY_LIMIT)
        dataset_rewrites = rewrites.get(name, {})
        if any(qid not in dataset_rewrites for qid in expected):
            sys.exit(f"[{name}] rewrites are incomplete, run the 'rewrite' phase first")

        print(f"[{name}] indexing {len(dataset.corpus)} docs...")
        retriever = RewriteRetriever(dataset.corpus)
        for config in missing:
            dataset_results[config] = score_config(retriever, dataset, dataset_rewrites, config)
            print(f"[{name}] {config}: {dataset_results[config]}")
            write_json(RESULTS_PATH, results)
    print(f"wrote {RESULTS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("phase", choices=["rewrite", "retrieve"])
    phase = parser.parse_args().phase
    if phase == "rewrite":
        run_rewrite_phase()
    else:
        run_retrieve_phase()


if __name__ == "__main__":
    main()
