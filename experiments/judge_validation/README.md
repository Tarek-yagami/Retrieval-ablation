# Judge validation (RQ5)

**Question:** can the 7B LLM judge behind the
[answer-quality experiment](../answer_quality/README.md) be trusted, and does
RQ2's conclusion (`dense_rerank` answers beat `sparse` answers) depend on it?

## Setup

- A second, much larger judge, `openai/gpt-oss-120b` via Groq, re-scored all
  80 answers using the same prompt, the same passages and the same 1-5
  rubric.
- Agreement is reported as exact match, match within one point, quadratically
  weighted Cohen's kappa (0 is chance, 1 is perfect) and Spearman correlation.
- 21 of the 80 answers are refusals: the generator said the passages don't
  contain an answer. They were identified by reading every answer, and the
  list is in `results/refusals.json`.
- Human labels have not been collected yet. A blind sample of 30 answers is
  fixed in `results/sample.json`, and `summarize.py` adds the human
  comparisons once `results/human_labels.json` exists.

## Results

| pair | axis | n | exact | within 1 | weighted kappa | Spearman |
|---|---|---|---|---|---|---|
| 7B vs 120B, all answers | faithfulness | 80 | 40% | 70% | 0.20 | 0.25 |
| 7B vs 120B, excluding refusals | faithfulness | 59 | 47% | 88% | 0.64 | 0.65 |
| 7B vs 120B, all answers | relevance | 80 | 44% | 75% | 0.41 | 0.43 |
| 7B vs 120B, excluding refusals | relevance | 59 | 49% | 81% | 0.11 | 0.21 |

| judge, answers | n | mean faithfulness | mean relevance |
|---|---|---|---|
| 7B, refusals | 21 | 1.29 | 1.33 |
| 7B, other answers | 59 | 4.20 | 4.25 |
| 120B, refusals | 21 | 4.33 | 3.19 |
| 120B, other answers | 59 | 3.86 | 4.56 |

| rater | n | sparse faithfulness | sparse relevance | dense_rerank faithfulness | dense_rerank relevance |
|---|---|---|---|---|---|
| 7B judge | 80 | 2.85 | 2.83 | 4.03 | 4.15 |
| 120B judge | 80 | 3.88 | 3.95 | 4.10 | 4.45 |
| 7B judge, excluding refusals | 59 | 3.92 | 3.88 | 4.40 | 4.51 |
| 120B judge, excluding refusals | 59 | 3.29 | 4.46 | 4.26 | 4.63 |

16 of the 40 `sparse` answers are refusals, against 5 of the 40
`dense_rerank` answers.

## Findings

**The judges disagree mainly on refusals.** Across all 80 answers,
faithfulness agreement is low (kappa 0.20). Nearly all of that comes from
the 21 refusals. The 7B judge scores a refusal as unfaithful (mean 1.29),
and the 120B judge scores it as faithful (mean 4.33), since a refusal makes
no claim the passages fail to support. The rubric doesn't say which reading
is right, so this is a gap in the rubric more than an error by either judge.
With refusals excluded, faithfulness agreement is moderate (kappa 0.64, 88%
of scores within one point).

**Relevance agreement on answered questions is weak, partly from a ceiling
effect.** The 120B judge gives almost every non-refusal answer a 4 or 5
(mean 4.56). With so little spread, kappa stays low (0.11) even though 81%
of scores are within one point.

**RQ2's conclusion holds under both judges, but its size doesn't.** Both
judges rate `dense_rerank` above `sparse` on both axes. The faithfulness gap
is +1.18 points under the 7B judge and only +0.22 under the 120B judge. Most
of the 7B gap comes from penalizing `sparse`'s refusals, which are three
times as common (16 against 5).

**Excluding refusals, both judges still prefer `dense_rerank`, and the
larger judge prefers it more.** On answered questions the faithfulness gap is
+0.48 (7B) and +0.97 (120B). When `sparse` retrieval returns weak passages and
the generator answers anyway, the answer tends to contain claims the passages
don't support. The larger judge catches more of those.

**The most judge-independent signal is the refusal rate.** `sparse` left the
generator with nothing usable for 40% of questions, and `dense_rerank` for
12.5%. Counting refusals needs no judge at all.

**Takeaway for LLM-as-judge setups:** the rubric needs an explicit rule for
"the context doesn't contain the answer". Without one, two capable judges
apply opposite conventions, and the headline number swings by a full point
while every individual score still looks reasonable.

## Run

```bash
uv run experiments/judge_validation/run.py sample  # once: fixes the 30-answer human-labeling sample
uv run experiments/judge_validation/run.py judge   # needs GROQ_API_KEY in .env
uv run experiments/judge_validation/summarize.py   # prints the tables above
```
