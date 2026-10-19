# Query rewriting ablation (RQ4)

**Question:** does rewriting the query with an LLM before retrieval improve
dense retrieval, and does it help most where queries and documents use
different vocabulary?

## Setup

All configs use the same dense retriever as the
[retrieval ablation](../retrieval_ablation/README.md).

| config | pipeline |
|---|---|
| `dense` | the query as written |
| `dense_rerank` | dense, then cross-encoder reranking |
| `dense_hyde` | the LLM writes a short passage that answers the query (HyDE), and dense search runs on the query plus that passage |
| `dense_hyde_rerank` | dense_hyde, then reranking against the original query |
| `dense_multiquery` | the LLM writes 3 paraphrases, dense search runs on each plus the original, and the 4 rankings are fused with RRF |

- Rewrites come from `gemma3:4b`, running locally through Ollama on CPU.
  HyDE uses temperature 0. Paraphrases use 0.3, because at 0 the model
  sometimes looped on queries with typos. Rewrites are cached in
  `results/rewrites.json`.
- 50 queries per dataset (first 50 by sorted query id), to keep the LLM
  phase to about an hour on a laptop. Both baselines are recomputed on the
  same 50 queries. That's why `dense` scores differ from the 200-query
  retrieval ablation.
- The prediction before running: HyDE would help most on nfcorpus (lay
  questions, expert documents) and least on scifact (claims that already
  read like the documents).

## Results

| dataset | config | nDCG@10 | Recall@100 | MRR@10 | rewrite (ms/query) | search (ms/query) |
|---|---|---|---|---|---|---|
| scifact | dense | 0.619 | 0.900 | 0.565 | 0 | 14.0 |
| scifact | dense_rerank | 0.701 | 0.900 | 0.677 | 0 | 6219.6 |
| scifact | dense_hyde | 0.689 | **0.960** | 0.645 | 8842 | 24.0 |
| scifact | **dense_hyde_rerank** | **0.721** | **0.960** | **0.686** | 8842 | 6193.6 |
| scifact | dense_multiquery | 0.627 | 0.900 | 0.569 | 10827 | 67.8 |
| fiqa | dense | 0.394 | 0.744 | 0.422 | 0 | 26.9 |
| fiqa | **dense_rerank** | **0.407** | 0.744 | 0.450 | 0 | 3990.7 |
| fiqa | dense_hyde | 0.385 | **0.746** | 0.438 | 9342 | 30.5 |
| fiqa | dense_hyde_rerank | 0.405 | **0.746** | **0.470** | 9342 | 4795.3 |
| fiqa | dense_multiquery | 0.376 | 0.743 | 0.434 | 10979 | 138.7 |
| nfcorpus | dense | 0.271 | 0.236 | 0.465 | 0 | 25.3 |
| nfcorpus | dense_rerank | 0.300 | 0.236 | 0.540 | 0 | 7181.9 |
| nfcorpus | **dense_hyde** | **0.312** | 0.244 | 0.527 | 8600 | 20.0 |
| nfcorpus | dense_hyde_rerank | 0.311 | 0.244 | **0.556** | 8600 | 5997.7 |
| nfcorpus | dense_multiquery | 0.307 | **0.270** | 0.495 | 8684 | 51.6 |

## Findings

**HyDE helped on scifact and nfcorpus and did nothing on fiqa.** nDCG@10
rose from 0.619 to 0.689 on scifact and from 0.271 to 0.312 on nfcorpus.
On fiqa it moved from 0.394 to 0.385, which is within noise at 50 queries.

**The prediction was half right.** nfcorpus gained the most in relative
terms (+15% against +11%), but scifact gained the most in absolute terms,
when it was expected to gain least. Vocabulary alone doesn't explain that.
The difference in form between query and document does. A scifact query is
a one-line claim, and the documents are abstracts. A HyDE passage reads like
an abstract, so it lands closer to the right documents, and Recall@100 rose
from 0.900 to 0.960. Fiqa's questions and forum answers are already written
the same way, so the rewrite adds the model's generic financial text and
nothing useful.

**The HyDE passages are often wrong, and it still works.** For the claim
"0-dimensional biomaterials show inductive properties", the model wrote
about electromagnetic induction, while the claim is about inducing tissue
growth. The passage only needs to use the vocabulary and style of the
relevant documents. Its facts don't have to be correct.

**On nfcorpus, HyDE beat the reranker.** `dense_hyde` (0.312) outscored
`dense_rerank` (0.300), and reranking the HyDE results added nothing
(0.311). On scifact the two combine: `dense_hyde_rerank` (0.721) is the best
config, because HyDE raised recall and the reranker then had more relevant
candidates to order.

**Multi-query fusion was mostly neutral.** It barely moved nDCG@10 on scifact
and fiqa, but it gave the best Recall@100 on nfcorpus (0.270 against 0.236).
Fusing rankings from the same retriever didn't cause the damage seen in the
retrieval ablation, where RRF fused in a much weaker BM25 ranking.

**The rewrite costs about as much time as the reranker on this hardware.**
Each rewrite took about 9 seconds with a 4B model on a laptop CPU, comparable
to the 4-7 seconds the reranker takes. A hosted model would be much faster,
but that wasn't measured here.

**Caveat:** 50 queries per dataset is a small sample. Treat differences
under about 0.02 nDCG@10 as noise.

## Run

```bash
uv run experiments/query_rewrite_ablation/run.py rewrite   # needs HYBRIDRAG_MODEL, see .env.example
uv run experiments/query_rewrite_ablation/run.py retrieve  # stop the local LLM first if memory is tight
uv run experiments/query_rewrite_ablation/summarize.py     # prints the results table
```
