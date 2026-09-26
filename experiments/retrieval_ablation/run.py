"""RQ1: does hybrid retrieval, and hybrid+rerank, actually beat single-method
retrieval, and what does each one cost per query to run? Runs all fixed
configs against each benchmark dataset, scores them against BEIR's
ground-truth qrels, and times them, since a quality win that costs 10x the
latency is a different decision than one that costs 10%, and nothing else in
this project measures that trade-off.

Latency is per-query search time only, not one-time indexing cost: a RAG
system indexes once and serves queries many times, so serving-time cost is
what's operationally relevant. Each config's model (in particular the
cross-encoder reranker, loaded lazily on first use) is warmed up with one
untimed query before the timed loop starts, so a one-time model-load cost
doesn't get misattributed to whichever config happens to run first.

Query count per dataset is capped (deterministically, by sorted query id) to
keep the cross-encoder pass tractable. This is a fixed, reproducible ablation,
not a full-corpus leaderboard run.

Skips configs a dataset already has both quality and latency results for,
instead of recomputing everything on every run (indexing a 57k-doc corpus
just to redo a config that hasn't changed is wasted work), and writes results
after each dataset so an interruption doesn't lose already-computed configs.

Usage:
    python experiments/retrieval_ablation/run.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hybridrag.data.beir_loader import load_beir_dataset  # noqa: E402
from hybridrag.metrics import evaluate  # noqa: E402
from hybridrag.pipeline import CONFIGS, RetrievalPipeline  # noqa: E402

DATASETS = ["scifact", "fiqa"]
MAX_QUERIES = 200
TOP_K = 100
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_PATH = RESULTS_DIR / "results.json"


def run_dataset(name: str, existing: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    missing = [c for c in CONFIGS if c not in existing or "latency_ms" not in existing[c]]
    if not missing:
        print(f"[{name}] all configs already computed, skipping")
        return existing

    print(f"[{name}] loading...")
    dataset = load_beir_dataset(name)
    query_ids = sorted(dataset.qrels.keys())[:MAX_QUERIES]
    qrels_subset = {qid: dataset.qrels[qid] for qid in query_ids}

    print(f"[{name}] indexing {len(dataset.corpus)} docs...")
    pipeline = RetrievalPipeline(dataset.corpus)

    results = dict(existing)
    for config in missing:
        print(f"[{name}] running config={config} over {len(query_ids)} queries...")

        # warm-up: not timed, not scored, just forces any lazily-loaded model
        # (the reranker) to load before the timed loop starts.
        pipeline.search(dataset.queries[query_ids[0]], config, top_k=TOP_K)

        run: dict[str, dict[str, float]] = {}
        start = time.perf_counter()
        for qid in query_ids:
            hits = pipeline.search(dataset.queries[qid], config, top_k=TOP_K)
            run[qid] = dict(hits)
        elapsed_s = time.perf_counter() - start

        results[config] = evaluate(qrels_subset, run)
        results[config]["latency_ms"] = (elapsed_s / len(query_ids)) * 1000
        print(f"[{name}] {config}: {results[config]}")

    return results


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {}

    for name in DATASETS:
        all_results[name] = run_dataset(name, all_results.get(name, {}))
        RESULTS_PATH.write_text(json.dumps(all_results, indent=2))

    print(f"wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
