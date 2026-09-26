"""Turns results.json into a markdown table for the README.

Usage:
    python experiments/chunking_ablation/summarize.py
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS_PATH = Path(__file__).parent / "results" / "results.json"


def main() -> None:
    results = json.loads(RESULTS_PATH.read_text())

    lines = ["| dataset | variant | nDCG@10 | Recall@100 | MRR@10 |", "|---|---|---|---|---|"]
    for dataset, variants in results.items():
        for variant, metrics in variants.items():
            lines.append(
                f"| {dataset} | {variant} | {metrics['ndcg@10']:.3f} | "
                f"{metrics['recall@100']:.3f} | {metrics['mrr@10']:.3f} |"
            )

    print("\n".join(lines))


if __name__ == "__main__":
    main()
