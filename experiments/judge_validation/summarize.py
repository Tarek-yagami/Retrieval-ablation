"""Compares the 7B judge, the 120B judge and (once collected) human labels, as markdown tables.

Usage:
    python experiments/judge_validation/summarize.py
"""

from __future__ import annotations

import json
from pathlib import Path

from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

AXES = ["faithfulness", "relevance"]
CONFIGS = ["sparse", "dense_rerank"]
RESULTS_DIR = Path(__file__).parent / "results"
ANSWER_QUALITY_RESULTS = Path(__file__).parents[1] / "answer_quality" / "results" / "results.json"
HUMAN_LABELS_PATH = RESULTS_DIR / "human_labels.json"

Key = tuple[str, str]  # (query_id, config)
Scores = dict[Key, dict[str, int]]


def read_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_scores(path: Path) -> Scores:
    return {(r["query_id"], r["config"]): {a: r[a] for a in AXES} for r in read_json(path)}


def load_human_scores() -> Scores:
    labels = {r["id"]: r for r in read_json(HUMAN_LABELS_PATH)}
    sample = read_json(RESULTS_DIR / "sample.json")
    return {(s["query_id"], s["config"]): {a: labels[s["id"]][a] for a in AXES} for s in sample}


def load_refusals() -> set[Key]:
    return {(r["query_id"], r["config"]) for r in read_json(RESULTS_DIR / "refusals.json")}


def agreement_row(name: str, a: Scores, b: Scores, keys: list[Key], axis: str) -> str:
    x = [a[k][axis] for k in keys]
    y = [b[k][axis] for k in keys]
    exact = sum(i == j for i, j in zip(x, y, strict=True)) / len(keys)
    within_one = sum(abs(i - j) <= 1 for i, j in zip(x, y, strict=True)) / len(keys)
    kappa = cohen_kappa_score(x, y, weights="quadratic", labels=[1, 2, 3, 4, 5])
    rho = spearmanr(x, y).statistic
    return f"| {name} | {axis} | {len(keys)} | {exact:.0%} | {within_one:.0%} | {kappa:.2f} | {rho:.2f} |"


def config_means_row(name: str, scores: Scores, keys: list[Key]) -> str:
    cells = []
    for config in CONFIGS:
        subset = [k for k in keys if k[1] == config]
        cells += [f"{sum(scores[k][a] for k in subset) / len(subset):.2f}" for a in AXES]
    return f"| {name} | {len(keys)} | " + " | ".join(cells) + " |"


def group_means_row(name: str, scores: Scores, keys: list[Key]) -> str:
    cells = [f"{sum(scores[k][a] for k in keys) / len(keys):.2f}" for a in AXES]
    return f"| {name} | {len(keys)} | " + " | ".join(cells) + " |"


def main() -> None:
    small_judge = load_scores(ANSWER_QUALITY_RESULTS)
    large_judge = load_scores(RESULTS_DIR / "second_judge.json")
    human = load_human_scores() if HUMAN_LABELS_PATH.exists() else None
    refusals = load_refusals()
    everything = sorted(small_judge)
    answered = [k for k in everything if k not in refusals]

    print("| pair | axis | n | exact | within 1 | weighted kappa | Spearman |")
    print("|---|---|---|---|---|---|---|")
    for axis in AXES:
        print(agreement_row("7B vs 120B, all answers", small_judge, large_judge, everything, axis))
        print(agreement_row("7B vs 120B, excluding refusals", small_judge, large_judge, answered, axis))
        if human:
            print(agreement_row("human vs 7B", human, small_judge, sorted(human), axis))
            print(agreement_row("human vs 120B", human, large_judge, sorted(human), axis))

    print()
    print("| judge, answers | n | mean faithfulness | mean relevance |")
    print("|---|---|---|---|")
    for name, scores in [("7B", small_judge), ("120B", large_judge)]:
        print(group_means_row(f"{name}, refusals", scores, sorted(refusals)))
        print(group_means_row(f"{name}, other answers", scores, answered))

    print()
    print("| rater | n | sparse faithfulness | sparse relevance | dense_rerank faithfulness | dense_rerank relevance |")
    print("|---|---|---|---|---|---|")
    print(config_means_row("7B judge", small_judge, everything))
    print(config_means_row("120B judge", large_judge, everything))
    print(config_means_row("7B judge, excluding refusals", small_judge, answered))
    print(config_means_row("120B judge, excluding refusals", large_judge, answered))
    if human:
        print(config_means_row("human", human, sorted(human)))

    print()
    for config in CONFIGS:
        count = sum(1 for k in refusals if k[1] == config)
        print(f"{config}: {count} of {sum(1 for k in everything if k[1] == config)} answers are refusals")


if __name__ == "__main__":
    main()
