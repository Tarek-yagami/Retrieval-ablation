# Answer quality (RQ2)

**Question:** does a retrieval-metric win show up in the quality of the
generated answer?

## Setup

- 40 fixed FiQA questions.
- Reduced corpus: every document relevant to those questions, plus a
  deterministic random sample of 5,000 others. Recall@100 isn't measured
  here, so the full ~57k-document corpus isn't needed, and the reduced corpus
  keeps the run feasible on modest hardware.
- Two retrieval configs: `sparse` (baseline) and `dense_rerank` (best on
  fiqa in the [retrieval ablation](../retrieval_ablation/README.md)).
- Top 5 passages go to the generator, a local `qwen2.5-coder:7b-instruct`
  via Ollama.
- An LLM judge (the same model) scores each answer from 1 to 5 on
  faithfulness (is it supported by the retrieved context?) and relevance
  (does it address the question?).

## Results

| config | n | mean faithfulness | mean relevance |
|---|---|---|---|
| sparse | 40 | 2.85 | 2.83 |
| **dense_rerank** | 40 | **4.03** | **4.15** |

## Findings

`dense_rerank` beats the BM25 baseline by more than a full point on both
scores, so the retrieval gain carried through to the answers. The
[judge validation](../judge_validation/README.md) shows that the direction
holds under a much larger judge, but the size of the gap depends on how the
judge scores refusals. That outcome
can't be taken for granted. A fixed prompt or a verbose model can absorb a
retrieval improvement.

One question shows the difference directly. For "Forex independent
investments", BM25 retrieved nothing relevant, and the generator correctly
said the context had no answer, which the judge scored 1/1. `dense_rerank`
retrieved a passage explaining that forex trading is zero-sum and not an
investment in the usual sense. The answer stuck to it and scored 5/5.

A much smaller judge (`qwen2.5:0.5b`) was tried first and gave 78 of 80
answers a flat 5/5, which made it useless as a signal. Very small models
make unreliable judges. Keep that in mind before setting `HYBRIDRAG_MODEL`
to a tiny model.

## Run

Needs an LLM configured through `HYBRIDRAG_MODEL` (see
[.env.example](../../.env.example)).

```bash
uv run experiments/answer_quality/run.py        # retrieval, then generation and judging
uv run experiments/answer_quality/summarize.py  # prints the results table
```
