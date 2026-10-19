# Retrieval ablation (RQ1)

**Question:** does hybrid retrieval, with or without reranking, beat a single
retriever, and what does each added component cost per query?

## Setup

Six configs, built so each added component can be measured on its own:

| config | pipeline |
|---|---|
| `sparse` | BM25 |
| `dense` | `all-MiniLM-L6-v2` embeddings, cosine similarity |
| `hybrid` | Reciprocal Rank Fusion (RRF) of sparse + dense |
| `hybrid_rerank` | hybrid, then cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`) |
| `dense_rerank` | dense, then cross-encoder reranking, no fusion |
| `sparse_rerank` | sparse, then cross-encoder reranking, no fusion |

`dense_rerank` was added after the first run showed fusion hurting on fiqa,
to test whether reranking without fusion was the better fix.
`sparse_rerank` completes the grid (retriever in {sparse, dense} x reranked
in {no, yes}).

Datasets, chosen for contrast:

- **scifact**: scientific claim verification.
- **fiqa**: financial question answering, where BM25 is weak.
- **nfcorpus**: lay medical questions against expert documents. It was added
  later to check whether the pattern from the first two held on a third
  dataset.

Each dataset uses its first 200 queries with relevance judgments, sorted by
query id. Metrics are nDCG@10, Recall@100 and MRR@10. Latency is per-query
search time, measured after one untimed warm-up query, and excludes one-time
indexing.

## Results

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
| nfcorpus | sparse | 0.272 | 0.178 | 0.494 | 5.9 |
| nfcorpus | dense | 0.320 | 0.291 | 0.522 | 13.4 |
| nfcorpus | hybrid | 0.313 | 0.273 | 0.536 | 20.2 |
| nfcorpus | **hybrid_rerank** | **0.345** | 0.273 | **0.575** | 8214.7 |
| nfcorpus | dense_rerank | 0.338 | **0.291** | 0.568 | 7592.6 |
| nfcorpus | sparse_rerank | 0.317 | 0.178 | 0.559 | 8234.8 |

## Findings

**Fusion can hurt a strong retriever.** On fiqa, BM25 is far weaker than
dense (0.189 vs 0.444 nDCG@10), but RRF weights the two rankings equally, so
`hybrid` (0.344) lands well below `dense` alone.

**Reranking without fusion beat hybrid + rerank on scifact and fiqa.** The
cause shows in Recall@100. Fused recall (0.912 scifact, 0.749 fiqa) is lower
than dense's (0.930, 0.769), so fusion pushes some relevant documents out of
the candidate pool. The reranker can only reorder the candidates it receives,
so it can't recover them. A useful check before adopting fusion is to compare
hybrid's Recall@100 with the stronger single retriever's.

**Reranking is capped by the retriever's recall.** `sparse_rerank` has
exactly the same Recall@100 as `sparse` on every dataset, since reranking
only reorders. It still improves nDCG@10 (0.561 to 0.635 on scifact, 0.189
to 0.313 on fiqa), but stays well below `dense_rerank` because dense found
more relevant documents to begin with.

**The pattern did not hold on nfcorpus.** There `hybrid_rerank` beat
`dense_rerank` (0.345 vs 0.338 nDCG@10). The margin is small, but it changes
which configuration is best. Scores on nfcorpus are far below scifact's for
every config, which reflects the vocabulary gap between lay questions and
expert documents.
Two out of three datasets is not enough to call it a rule.

**Reranking is expensive.** The reranked configs take 4.7 to 8.2 seconds per
query, against 6 to 294 ms for the others. `dense_rerank` is 280-570x slower
than `dense` for a 3-6% nDCG@10 gain. Each query reranks a 100-document
shortlist so that Recall@100 stays comparable across configs. A system that
reranks a smaller top-k (the demo uses 20) pays proportionally less. Whether
the gain is worth it depends on the latency budget. An offline batch job can
absorb it, but an interactive chat usually can't.

## Run

```bash
uv run experiments/retrieval_ablation/run.py        # skips configs already in results/
uv run experiments/retrieval_ablation/summarize.py  # prints the results table
```
