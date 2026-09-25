"""Aggregates per-question judge scores into a mean-per-config table.

Usage:
    python experiments/answer_quality/summarize.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

RESULTS_PATH = Path(__file__).parent / "results" / "results.json"


def main() -> None:
    records = json.loads(RESULTS_PATH.read_text())

    by_config: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_config[r["config"]].append(r)

    lines = ["| config | n | mean faithfulness | mean relevance |", "|---|---|---|---|"]
    for config, rows in by_config.items():
        n = len(rows)
        mean_faith = sum(r["faithfulness"] for r in rows) / n
        mean_rel = sum(r["relevance"] for r in rows) / n
        lines.append(f"| {config} | {n} | {mean_faith:.2f} | {mean_rel:.2f} |")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
