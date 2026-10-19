"""RQ5: can the answer-quality experiment's 7B LLM judge be trusted?

`sample` picks the answers to hand-label, blind to config and judge score.
`judge` re-scores every answer with a stronger second judge.
Human labels are saved to results/human_labels.json; summarize.py compares all three.

Usage:
    python experiments/judge_validation/run.py sample
    python experiments/judge_validation/run.py judge   # needs GROQ_API_KEY in .env
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hybridrag.judge import judge_answer  # noqa: E402

SAMPLE_PER_CONFIG = 15
SECOND_JUDGE_MODEL = "groq/openai/gpt-oss-120b"
_SEED = 0
ANSWER_QUALITY_RESULTS = Path(__file__).parents[1] / "answer_quality" / "results"
RESULTS_DIR = Path(__file__).parent / "results"
SAMPLE_PATH = RESULTS_DIR / "sample.json"
SECOND_JUDGE_PATH = RESULTS_DIR / "second_judge.json"


def load_answers() -> list[dict]:
    return json.loads((ANSWER_QUALITY_RESULTS / "results.json").read_text(encoding="utf-8"))


def load_passages() -> dict[tuple[str, str], list[str]]:
    cache_path = ANSWER_QUALITY_RESULTS / "retrieval_cache.json"
    if not cache_path.exists():
        sys.exit(f"{cache_path} is missing, run experiments/answer_quality/run.py to rebuild it")
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    return {(item["query_id"], item["config"]): item["passages"] for item in cache}


def run_sample_phase() -> None:
    if SAMPLE_PATH.exists():
        sys.exit(f"{SAMPLE_PATH} already exists and labels refer to its ids, delete it to resample")

    rng = random.Random(_SEED)
    answers = load_answers()
    sample = []
    for config in sorted({a["config"] for a in answers}):
        pool = sorted((a for a in answers if a["config"] == config), key=lambda a: a["query_id"])
        sample += rng.sample(pool, SAMPLE_PER_CONFIG)
    rng.shuffle(sample)

    rows = [{"id": str(uuid.uuid4()), "query_id": a["query_id"], "config": a["config"]} for a in sample]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} items to {SAMPLE_PATH}")


def run_judge_phase() -> None:
    passages = load_passages()
    scores = json.loads(SECOND_JUDGE_PATH.read_text(encoding="utf-8")) if SECOND_JUDGE_PATH.exists() else []
    done = {(s["query_id"], s["config"]) for s in scores}

    answers = load_answers()
    for i, a in enumerate(answers, start=1):
        key = (a["query_id"], a["config"])
        if key in done:
            continue
        print(f"[judge {i}/{len(answers)}] {a['question'][:60]!a} ({a['config']})")
        score = judge_answer(a["question"], a["answer"], passages[key], model=SECOND_JUDGE_MODEL)
        scores.append(
            {
                "query_id": a["query_id"],
                "config": a["config"],
                "faithfulness": score.faithfulness,
                "relevance": score.relevance,
                "reasoning": score.reasoning,
            }
        )
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        SECOND_JUDGE_PATH.write_text(json.dumps(scores, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {SECOND_JUDGE_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("phase", choices=["sample", "judge"])
    phase = parser.parse_args().phase
    if phase == "sample":
        run_sample_phase()
    else:
        run_judge_phase()


if __name__ == "__main__":
    main()
