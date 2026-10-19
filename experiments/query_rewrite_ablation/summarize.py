"""Turns results.json into a markdown table for the README.

Usage:
    python experiments/query_rewrite_ablation/summarize.py
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS_PATH = Path(__file__).parent / "results" / "results.json"


def main() -> None:
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

    lines = [
        "| dataset | config | nDCG@10 | Recall@100 | MRR@10 | rewrite (ms/query) | search (ms/query) |",
        "|---|---|---|---|---|---|---|",
    ]
    for dataset, configs in results.items():
        for config, metrics in configs.items():
            lines.append(
                f"| {dataset} | {config} | {metrics['ndcg@10']:.3f} | "
                f"{metrics['recall@100']:.3f} | {metrics['mrr@10']:.3f} | "
                f"{metrics['rewrite_ms']:.0f} | {metrics['search_ms']:.1f} |"
            )

    print("\n".join(lines))


if __name__ == "__main__":
    main()
