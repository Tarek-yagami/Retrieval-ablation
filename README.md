# Retrieval Ablation

A controlled benchmark of RAG retrieval strategies (BM25, dense embeddings,
hybrid fusion, cross-encoder reranking) against ground-truth relevance
judgments on three BEIR datasets. It tests the common advice that
"hybrid + rerank" is the safe default, and checks whether retrieval gains
carry through to better generated answers.

**Live demo:** https://retrieval-ablation.streamlit.app/

## Key findings

- **Hybrid + rerank was not the safe default.** Reranking dense results
  without fusion beat the full hybrid + rerank pipeline on two of three
  datasets. Fusion dropped relevant documents from the candidate pool
  before the reranker ever saw them (lower Recall@100 than dense alone).
- **The winner changed with the dataset.** On the third dataset (nfcorpus),
  hybrid + rerank won back by a small margin. No configuration was best
  everywhere.
- **Reranking has a large latency cost.** `dense_rerank` was 280-570x slower
  per query than plain `dense`, for a 3-6% nDCG@10 gain.
- **LLM query rewriting (HyDE) helped where queries and documents differ in
  form.** It lifted dense nDCG@10 by 11-15% on scifact and nfcorpus, where it
  beat the reranker, and did nothing on fiqa.
- **Better retrieval gave better answers.** On 40 judged FiQA questions,
  answers built on `dense_rerank` scored more than a full point higher (out
  of 5) on faithfulness and relevance than answers built on BM25.

## Results at a glance

nDCG@10, first 200 queries of each dataset (best per row in bold):

| dataset | sparse | dense | hybrid | hybrid_rerank | dense_rerank | sparse_rerank |
|---|---|---|---|---|---|---|
| scifact | 0.561 | 0.672 | 0.630 | 0.691 | **0.700** | 0.635 |
| fiqa | 0.189 | 0.444 | 0.344 | 0.448 | **0.458** | 0.313 |
| nfcorpus | 0.272 | 0.320 | 0.313 | **0.345** | 0.338 | 0.317 |

## Experiments

Each experiment has its own folder with the code, the raw results, and a
write-up of the method and findings.

| experiment | question | write-up |
|---|---|---|
| Retrieval ablation | Does hybrid retrieval and reranking beat a single retriever, and at what latency? | [experiments/retrieval_ablation](experiments/retrieval_ablation/README.md) |
| Answer quality | Do retrieval gains show up in LLM-judged answer quality? | [experiments/answer_quality](experiments/answer_quality/README.md) |
| Chunking | Does chunk size change dense retrieval quality, and is it consistent across corpora? | [experiments/chunking_ablation](experiments/chunking_ablation/README.md) |
| Query rewriting | Do LLM query rewrites (HyDE, multi-query) improve dense retrieval? | [experiments/query_rewrite_ablation](experiments/query_rewrite_ablation/README.md) |

## Stack

Python, [BEIR](https://github.com/beir-cellar/beir) datasets, `rank-bm25`,
`sentence-transformers` (embeddings and cross-encoder reranker), `ranx` for
IR metrics, `litellm` for a provider-agnostic LLM layer (Claude, GPT, Groq or
local Ollama), Streamlit for the demo, GitHub Actions for lint and tests.

## Quickstart

Uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run streamlit run app.py                     # interactive demo
uv run experiments/retrieval_ablation/run.py    # no API key needed
```

The answer-quality experiment needs an LLM. Set `HYBRIDRAG_MODEL` as shown in
[.env.example](.env.example). Each experiment's write-up lists its own run
commands.

## Project layout

```
src/hybridrag/
├── data/          # BEIR loading, precomputed demo embeddings
├── retrieval/     # sparse, dense, RRF fusion, cross-encoder rerank
├── pipeline.py    # the six retrieval configs
├── benchmark.py   # datasets, retrieval depth and query sample shared by experiments
├── chunking.py    # fixed-size chunking, chunk-to-document mapping
├── query_rewrite.py  # LLM query rewriting: HyDE passages, paraphrases
├── metrics.py     # nDCG@10, Recall@100, MRR@10
├── llm.py         # provider-agnostic completion via litellm
├── generation.py  # answer generation from retrieved passages
└── judge.py       # LLM-as-judge: faithfulness and relevance

experiments/       # one folder per experiment: run.py, summarize.py, results/
app.py             # Streamlit: results dashboard and query playground
tests/             # unit tests, run in CI
```

## Limitations

- Queries are capped at 200 per dataset to keep the reranker pass tractable,
  and at 50 for query rewriting, which needs an LLM call per query.
- One embedding model (`all-MiniLM-L6-v2`) and one reranker were tested.
- Answers were judged by a single local 7B model, which has not yet been
  validated against human labels.

## License

MIT
