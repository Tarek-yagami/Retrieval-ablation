# Chunking ablation (RQ3)

**Question:** does chunk size change dense retrieval quality, and is the
effect consistent across corpora?

## Setup

- Dense retrieval only. Chunking matters because a long document's
  embedding averages over everything in it, and BM25 term scoring doesn't
  have that problem.
- Variants: whole documents (reused from the
  [retrieval ablation](../retrieval_ablation/README.md)'s `dense` row),
  100-word chunks, and 250-word chunks.
- BEIR's relevance judgments are per document, so each document is scored
  by its best-matching chunk. That keeps the metrics directly comparable to
  the main ablation.
- Same datasets, 200-query cap and metrics as the retrieval ablation.

## Results

| dataset | variant | nDCG@10 | Recall@100 | MRR@10 |
|---|---|---|---|---|
| scifact | whole_document | 0.672 | 0.930 | 0.637 |
| scifact | **chunks_100** | **0.694** | **0.950** | **0.653** |
| scifact | chunks_250 | 0.683 | 0.935 | 0.650 |
| fiqa | whole_document | 0.444 | 0.769 | 0.535 |
| fiqa | chunks_100 | 0.446 | 0.750 | 0.530 |
| fiqa | chunks_250 | 0.448 | 0.766 | 0.534 |
| nfcorpus | whole_document | 0.320 | 0.291 | 0.522 |
| nfcorpus | **chunks_100** | **0.335** | **0.297** | **0.538** |
| nfcorpus | chunks_250 | 0.311 | 0.285 | 0.505 |

## Findings

**Document structure mattered more than document length.** The prediction
going in was that fiqa, with its longer documents, would gain the most. The
opposite happened. Chunking improved every metric on scifact and made no
meaningful difference on fiqa, where chunks_100 even lost a little Recall@100.

Scifact abstracts mix background, method and result in one paragraph, and
the evidence for a claim is usually a single sentence. Small chunks isolate
that sentence instead of averaging it with the rest. Fiqa's forum answers
each cover one topic already, so splitting them concentrates nothing and can
cut a coherent answer in half.

**Chunk size has to fit the documents.** nfcorpus documents are medical
abstracts with the same structure as scifact's, and chunks_100 improves
every metric there too. But its median document is 237 words, so 250-word
chunks leave most documents whole. They only split the few long ones, which
costs some coherence and doesn't isolate anything, and chunks_250 ends up
slightly below whole documents.

## Run

```bash
uv run experiments/chunking_ablation/run.py        # skips variants already in results/
uv run experiments/chunking_ablation/summarize.py  # prints the results table
```
