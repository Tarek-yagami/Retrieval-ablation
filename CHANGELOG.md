# Changelog

## 0.1.0 - initial release

- Hybrid retrieval pipeline: BM25 (sparse), sentence-transformer embeddings (dense), Reciprocal Rank Fusion, cross-encoder reranking.
- Retrieval ablation across six fixed configs (sparse / dense / hybrid / hybrid+rerank / dense+rerank / sparse+rerank) on two BEIR benchmarks (SciFact, FiQA), scored with IR metrics (nDCG@10, Recall@100, MRR@10) via `ranx`.
- `dense_rerank` (reranking without fusion) added specifically to test the project's own claim that it's a viable alternative when hybrid fusion drags a strong retriever down.
- `sparse_rerank` added afterward to complete the retriever x reranked grid; confirms reranking is bounded by the underlying retriever's recall, not a substitute for it.
- `retrieval_ablation/run.py` skips configs a dataset already has results for instead of recomputing everything on every run.
- Provider-agnostic generation + LLM-as-judge layer (Claude / GPT / Groq / Ollama via litellm).
- Small judged-answer sample (FiQA, 40 questions) comparing the sparse baseline against the best retrieval config.
- Streamlit app: ablation dashboard + live query playground.
