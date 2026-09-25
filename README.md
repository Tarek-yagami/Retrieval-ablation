# Retrieval Ablation

"Hybrid retrieval + reranking" gets recommended as the sophisticated default
for RAG systems, on the assumption that more components means better results.
This project tests that assumption against real ground-truth relevance
judgments instead of taking it on faith, and finds it doesn't hold: on both
datasets tested, reranking *without* fusion beats the full hybrid+rerank
pipeline. Fusing a weaker retriever into a strong one, then reranking, does
worse than just reranking the strong one's own candidates. The added
complexity wasn't free, and nobody would know that without measuring it.
There is no configuration that's safe to assume by default, including the
ones that sound more sophisticated. Every added piece has to prove it earns
its cost on your own data.

**Live demo:** _(deployed link goes here once published to Streamlit Community Cloud)_

## The problem

Retrieval architecture choices in RAG systems are usually made by consensus
("hybrid beats single-method, add a reranker, ship it") rather than by testing
against the specific corpus and query distribution at hand. That consensus
isn't wrong often enough to get caught casually. Hybrid retrieval *does* win
on many benchmarks, and that's exactly what makes it risky to apply
automatically: the cases where it doesn't hold don't announce themselves,
they just quietly ship a worse system.

This project treats retrieval technique choice as an empirical question, not
a default: fix a small set of concrete strategies, measure each one against
real relevance judgments on more than one kind of corpus, and check whether
a retrieval-metric improvement actually survives all the way to a better
generated answer. The goal isn't to crown the best generic technique, since
that question doesn't have a stable answer. It's to demonstrate the process
for finding out, and to walk through what the process turned up.

## Approach

Five retrieval configs are built so each added piece of complexity (dense
retrieval, then fusion, then reranking) can be isolated and checked
individually rather than judged as one bundle: sparse, dense, hybrid (fusion),
hybrid+rerank, and dense+rerank (reranking with the fusion step removed). That
last one wasn't part of the original design. It was added after the first run
showed fusion hurting on one dataset, specifically to test whether
reranking-without-fusion was the actual fix or just a plausible-sounding
guess. It turned out to matter on both datasets, not just the one that
prompted it. Two BEIR datasets with real relevance judgments were run side by
side deliberately for contrast, not coverage. A second experiment then checks
whether a retrieval-side win (nDCG, recall) actually shows up in judged
answer quality, since those two things are often assumed to move together
without anyone checking.

## What's benchmarked

Five fixed retrieval configs, run against two [BEIR](https://github.com/beir-cellar/beir)
datasets with real relevance judgments (`scifact`: claim verification;
`fiqa`: financial QA):

- **sparse**: BM25 only
- **dense**: sentence-transformer embeddings, cosine similarity
- **hybrid**: Reciprocal Rank Fusion of sparse + dense
- **hybrid_rerank**: hybrid, then cross-encoder reranking of the top candidates
- **dense_rerank**: dense only, then cross-encoder reranking, no fusion (added mid-project, see below)

Scored with nDCG@10, Recall@100, and MRR@10.

### Retrieval ablation results

Run on the first 200 qrels-covered queries of each dataset (deterministic, by
sorted query id, see `experiments/retrieval_ablation/run.py`):

| dataset | config | nDCG@10 | Recall@100 | MRR@10 |
|---|---|---|---|---|
| scifact | sparse | 0.561 | 0.762 | 0.531 |
| scifact | dense | 0.672 | 0.930 | 0.637 |
| scifact | hybrid | 0.630 | 0.912 | 0.600 |
| scifact | hybrid_rerank | 0.691 | 0.912 | 0.660 |
| scifact | **dense_rerank** | **0.700** | **0.930** | **0.665** |
| fiqa | sparse | 0.189 | 0.384 | 0.233 |
| fiqa | dense | 0.444 | 0.769 | 0.535 |
| fiqa | hybrid | 0.344 | 0.749 | 0.430 |
| fiqa | hybrid_rerank | 0.448 | 0.749 | 0.529 |
| fiqa | **dense_rerank** | **0.458** | **0.769** | **0.545** |

The first four rows told a "consensus recommendation works on one dataset,
fails on the other" story: hybrid+rerank won on scifact as expected, but on
fiqa plain RRF fusion (`hybrid`) was *worse* than dense alone, because BM25 is
much weaker there (0.189 nDCG vs dense's 0.444) and rank-based fusion still
gives it equal weight, dragging the strong retriever down.

Adding `dense_rerank` sharpened that into a stronger finding: reranking
without fusion beats the full hybrid+rerank pipeline on **both** datasets,
not just the one where fusion visibly hurt. The mechanism is in the recall
column, not just nDCG. Fused recall@100 (0.912 scifact, 0.749 fiqa) is lower
than dense-alone's recall (0.930, 0.769) on both datasets, so RRF fusion is
quietly dropping some of dense's genuinely relevant documents out of the
candidate pool before the reranker ever sees them, comparable-quality
retrievers or not. Reranking can only reorder whatever candidates it's given,
and it can't recover a document fusion already discarded.

**The diagnostic, not just the result:** "more components" (fusion, on top
of a reranker that's already correcting for a lot) is not the same as "better
results," and the only way to know is to benchmark the simpler alternative
before committing to the more complex one. Concretely here, if you already
have a good reranker, check whether fusing in a second retriever changes
which documents make the candidate shortlist. Recall@100 for hybrid versus
the stronger single retriever answers that directly, before assuming fusion
is free upside.

### Answer quality

On a fixed 40-question FiQA sample, retrieved over a reduced corpus (every
document relevant to those 40 questions, plus a 5,000-document random
background: the full ~57k-doc corpus isn't needed here the way it is for the
ablation's recall@100, and it keeps this runnable on modest hardware), the
sparse baseline and the best retrieval config (`dense_rerank`) each generate
an answer (via a local `qwen2.5-coder:7b-instruct` through Ollama, see
`.env.example` to swap in Claude/GPT/Groq instead), which is then scored by
an LLM judge on faithfulness (is it actually supported by the retrieved
context) and relevance (does it address the question), 1-5 each:

| config | n | mean faithfulness | mean relevance |
|---|---|---|---|
| sparse | 40 | 2.85 | 2.83 |
| **dense_rerank** | 40 | **4.03** | **4.15** |

This experiment exists to check an assumption the ablation results don't
verify on their own: that a retrieval-metric win actually shows up where it
matters, in the answer the user sees. Here it does, decisively: `dense_rerank`
beats the sparse baseline by more than a full point on both faithfulness and
relevance, so the nDCG improvement wasn't just a number that stayed on paper.
That's not guaranteed in general (a system could improve retrieval metrics
while a fixed prompt template or a verbose LLM absorbs the difference), which
is exactly why this project checks it explicitly instead of assuming it. The
judge's reasoning is legible too. When a question has no real answer in the
retrieved context (e.g. "Forex independent investments"), both configs
correctly score 1/1 for saying so instead of confabulating, which is itself
part of what "faithfulness" is meant to catch.

(Note: an earlier pass judging with a much smaller local model, `qwen2.5:0.5b`,
scored 78/80 answers a flat 5/5, a rubber-stamp, not a real signal. Small
models make unreliable judges; that failure mode is worth knowing if you swap
`HYBRIDRAG_MODEL` for something tiny.)

## Try it

Uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync

# Ablation (no API key needed)
uv run experiments/retrieval_ablation/run.py
uv run experiments/retrieval_ablation/summarize.py

# Answer quality (needs an LLM, see .env.example)
uv run experiments/answer_quality/run.py
uv run experiments/answer_quality/summarize.py

# Interactive demo
uv run streamlit run app.py
```

Switching the LLM provider is a one-line env var change (`HYBRIDRAG_MODEL`,
see `.env.example`). The generation and judge code never touch a vendor SDK
directly (`src/hybridrag/llm.py`).

## Project layout

```
src/hybridrag/
├── data/beir_loader.py     # loads a BEIR dataset (corpus + queries + qrels)
├── retrieval/               # sparse, dense, RRF fusion, cross-encoder rerank
├── pipeline.py               # wires the five fixed configs together
├── llm.py                    # provider-agnostic completion() via litellm
├── generation.py             # answer generation from retrieved context
├── judge.py                  # LLM-as-judge: faithfulness + relevance
└── metrics.py                 # nDCG@10 / Recall@100 / MRR@10

experiments/
├── retrieval_ablation/       # RQ1: does hybrid/rerank beat single-method retrieval?
└── answer_quality/           # RQ2: does that translate into better-judged answers?

app.py                        # Streamlit: ablation dashboard + query playground
```

## Roadmap

Not built, and deliberately out of scope for v1, but the natural next steps:

- Test the quality-gap diagnostic (see above) against more corpora, to see how general the pattern actually is beyond these two datasets.
- More retrieval strategies: query rewriting / HyDE, late-interaction models (ColBERT).
- A custom domain corpus beyond the two BEIR benchmarks.
- Cost/latency tracking per config, not quality alone, since that's a quality-vs-cost frontier.
- Validating the LLM judge against a small human-labeled sample.
- Generalizing this into a reusable CLI (`hybridrag-eval run --config ...`) that evaluates
  any RAG pipeline, not just this benchmark.
- CI that reruns the ablation when retrieval code changes, to catch silent regressions.

## License

MIT
