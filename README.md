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
its cost on your own data, and "cost" turns out to be literal too: reranking's
quality gain comes with a measured 150-500x latency increase per query.

**Live demo:** https://retrieval-ablation.streamlit.app/

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

Six retrieval configs are built so each added piece of complexity (dense
retrieval, then fusion, then reranking) can be isolated and checked
individually rather than judged as one bundle: sparse, dense, hybrid (fusion),
hybrid+rerank, dense+rerank, and sparse+rerank. Two of these weren't part of
the original design. dense_rerank was added after the first run showed fusion
hurting on one dataset, to test whether reranking-without-fusion was the
actual fix or just a plausible-sounding guess. It turned out to matter on
both datasets, not just the one that prompted it. sparse_rerank was added
afterward to complete the resulting grid (retriever in {sparse, dense} x
reranked in {no, yes}), and confirms that reranking is bounded by the
underlying retriever's recall rather than a substitute for it. Two BEIR
datasets with real relevance judgments were run side by side deliberately for
contrast, not coverage. A second experiment then checks whether a
retrieval-side win (nDCG, recall) actually shows up in judged answer quality,
since those two things are often assumed to move together without anyone
checking. A third experiment applies the same "measure it, don't assume it"
discipline to chunk size, a different, widely repeated RAG claim ("there's no
universal best chunk size") that this project happened to independently
verify rather than just cite.

## What's benchmarked

Six fixed retrieval configs, run against two [BEIR](https://github.com/beir-cellar/beir)
datasets with real relevance judgments (`scifact`: claim verification;
`fiqa`: financial QA):

- **sparse**: BM25 only
- **dense**: sentence-transformer embeddings, cosine similarity
- **hybrid**: Reciprocal Rank Fusion of sparse + dense
- **hybrid_rerank**: hybrid, then cross-encoder reranking of the top candidates
- **dense_rerank**: dense only, then cross-encoder reranking, no fusion (added mid-project, see below)
- **sparse_rerank**: sparse only, then cross-encoder reranking, no fusion (completes the grid: does reranking help a weak retriever the same way it helps a strong one)

Scored with nDCG@10, Recall@100, and MRR@10.

### Retrieval ablation results

Run on the first 200 qrels-covered queries of each dataset (deterministic, by
sorted query id, see `experiments/retrieval_ablation/run.py`):

| dataset | config | nDCG@10 | Recall@100 | MRR@10 | latency (ms/query) |
|---|---|---|---|---|---|
| scifact | sparse | 0.561 | 0.762 | 0.531 | 27.8 |
| scifact | dense | 0.672 | 0.930 | 0.637 | 12.9 |
| scifact | hybrid | 0.630 | 0.912 | 0.600 | 40.4 |
| scifact | hybrid_rerank | 0.691 | 0.912 | 0.660 | 6381.8 |
| scifact | **dense_rerank** | **0.700** | **0.930** | **0.665** | 6260.3 |
| scifact | sparse_rerank | 0.635 | 0.762 | 0.615 | 6685.9 |
| fiqa | sparse | 0.189 | 0.384 | 0.233 | 258.1 |
| fiqa | dense | 0.444 | 0.769 | 0.535 | 17.4 |
| fiqa | hybrid | 0.344 | 0.749 | 0.430 | 293.7 |
| fiqa | hybrid_rerank | 0.448 | 0.749 | 0.529 | 4732.5 |
| fiqa | **dense_rerank** | **0.458** | **0.769** | **0.545** | 4925.8 |
| fiqa | sparse_rerank | 0.313 | 0.384 | 0.423 | 6324.4 |

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

`sparse_rerank` completes the grid (retriever in {sparse, dense} x reranked
in {no, yes}) and confirms why reranking has a ceiling. Its recall@100
(0.762 scifact, 0.384 fiqa) is identical to plain sparse's, on both datasets,
because reranking only reorders candidates the retriever already found; it
can't add ones BM25 missed. nDCG@10 still improves over plain sparse (0.561
to 0.635 on scifact, 0.189 to 0.313 on fiqa), so reranking is doing real work
within that fixed candidate set, but it stays well below `dense_rerank` on
both datasets because dense simply found more of the relevant documents to
begin with. Reranking is not a substitute for a retriever with good recall,
it's a refinement on top of one.

**The quality numbers aren't the whole decision, cost is the other half.**
The three reranked configs cost roughly 5-6.7 *seconds* per query, against
13-300ms for the ones without a reranker, a 150-500x latency jump. That's the
cost of a cross-encoder forward pass over a 100-candidate shortlist (the
shortlist size this experiment uses to keep recall@100 comparable across
configs; a production system reranking a smaller top-k, like this project's
own demo does with 20, would pay proportionally less). So `dense_rerank`'s
quality edge over plain `dense` (+4% nDCG on scifact, +3% on fiqa) comes
bundled with roughly 500x the latency. Whether that's worth it depends
entirely on the application's latency budget, a batch offline pipeline can
absorb 6 seconds a query without anyone noticing; an interactive chat
response usually can't. This project doesn't pick a winner here on purpose:
"is the quality gain worth the cost" is the same kind of question as "is
fusion worth its cost," it needs the actual cost measured for your own
latency budget, not assumed away.

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

### Chunking

A widely repeated lesson in RAG write-ups is that there's no universal best
chunk size, it has to be tested per application. That's exactly this
project's own thesis applied to a different variable, so it's worth testing
here too, on `dense` retrieval specifically (chunking is an embedding-dilution
concern; BM25's term-frequency scoring isn't sensitive to it the same way).
Chunk-level hits are mapped back to their parent document, since BEIR's
relevance judgments are per-document, not per-chunk, before scoring with the
same metrics as the main ablation:

| dataset | variant | nDCG@10 | Recall@100 | MRR@10 |
|---|---|---|---|---|
| scifact | whole_document | 0.672 | 0.930 | 0.637 |
| scifact | **chunks_100** | **0.694** | **0.950** | **0.653** |
| scifact | chunks_250 | 0.683 | 0.935 | 0.650 |
| fiqa | whole_document | 0.444 | 0.769 | 0.535 |
| fiqa | chunks_100 | 0.446 | 0.750 | 0.530 |
| fiqa | chunks_250 | 0.448 | 0.766 | 0.534 |

The prediction going in was that fiqa, with its longer tail of documents,
would benefit from chunking more than scifact's short abstracts. The data
says the opposite: chunking clearly helps on scifact (every metric improves,
chunks_100 most), and does essentially nothing on fiqa (differences are
within noise, and recall@100 for chunks_100 is actually slightly worse than
the whole-document baseline).

Document length wasn't the variable that mattered, document *structure* was.
Scifact's abstracts mix background, method, and result in one paragraph, and
the actual evidence for a claim is usually one specific sentence buried in
that mix. Chunking isolates that sentence instead of averaging its embedding
against the surrounding paragraph. Fiqa's forum answers are already focused
on one topic each, being short and single-purpose, so splitting them further
doesn't concentrate anything, it just risks cutting a coherent answer in
half. The same "measure it, don't assume it" diagnostic from the retrieval
ablation applies again, just with a different underlying cause: chunking
helps when a document bundles multiple ideas together, not simply when it's
long.

## Try it

Uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync

# Ablation (no API key needed)
uv run experiments/retrieval_ablation/run.py
uv run experiments/retrieval_ablation/summarize.py

# Chunking (no API key needed)
uv run experiments/chunking_ablation/run.py
uv run experiments/chunking_ablation/summarize.py

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
├── data/beir_loader.py       # loads a BEIR dataset (corpus + queries + qrels)
├── data/playground_cache.py  # builds/loads the demo's precomputed dense embeddings
├── retrieval/                 # sparse, dense, RRF fusion, cross-encoder rerank
├── chunking.py                # fixed-size chunking + chunk-to-document dedup
├── pipeline.py                 # wires the six fixed retrieval configs together
├── llm.py                      # provider-agnostic completion() via litellm
├── generation.py               # answer generation from retrieved context
├── judge.py                    # LLM-as-judge: faithfulness + relevance
└── metrics.py                   # nDCG@10 / Recall@100 / MRR@10

experiments/
├── retrieval_ablation/       # RQ1: does hybrid/rerank beat single-method retrieval?
├── chunking_ablation/        # RQ3: does chunk size affect dense retrieval quality?
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
