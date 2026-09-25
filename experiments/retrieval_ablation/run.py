"""RQ1: does hybrid retrieval, and hybrid+rerank, actually beat single-method
retrieval? Runs all five fixed configs against each benchmark dataset and
scores them against BEIR's ground-truth qrels.

Query count per dataset is capped (deterministically, by sorted query id) to
keep the cross-encoder pass tractable. This is a fixed, reproducible ablation,
not a full-corpus leaderboard run.

Usage:
    python experiments/retrieval_ablation/run.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hybridrag.data.beir_loader import load_beir_dataset  # noqa: E402
from hybridrag.metrics import evaluate  # noqa: E402
from hybridrag.pipeline import CONFIGS, RetrievalPipeline  # noqa: E402

DATASETS = ["scifact", "fiqa"]
MAX_QUERIES = 200
TOP_K = 100
RESULTS_DIR = Path(__file__).parent / "results"


def run_dataset(name: str) -> dict[str, dict[str, float]]:
    print(f"[{name}] loading...")
    dataset = load_beir_dataset(name)
    query_ids = sorted(dataset.qrels.keys())[:MAX_QUERIES]

    print(f"[{name}] indexing {len(dataset.corpus)} docs...")
    pipeline = RetrievalPipeline(dataset.corpus)

    results: dict[str, dict[str, float]] = {}
    for config in CONFIGS:
        print(f"[{name}] running config={config} over {len(query_ids)} queries...")
        run: dict[str, dict[str, float]] = {}
        for qid in query_ids:
            hits = pipeline.search(dataset.queries[qid], config, top_k=TOP_K)
            run[qid] = dict(hits)

        qrels_subset = {qid: dataset.qrels[qid] for qid in query_ids}
        results[config] = evaluate(qrels_subset, run)
        print(f"[{name}] {config}: {results[config]}")

    return results


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {name: run_dataset(name) for name in DATASETS}

    out_path = RESULTS_DIR / "results.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
