# Changelog

## 0.1.0 - initial release

- Retrieval pipeline: BM25, dense embeddings, Reciprocal Rank Fusion and cross-encoder reranking, combined into six configs.
- Retrieval ablation on three BEIR datasets (scifact, fiqa, nfcorpus), scored with nDCG@10, Recall@100 and MRR@10, with per-query latency.
- Answer-quality experiment: 40 FiQA questions, generated and scored by an LLM judge for faithfulness and relevance.
- Chunking ablation: 100- and 250-word chunks against whole documents for dense retrieval.
- Provider-agnostic LLM layer via litellm (Claude, GPT, Groq, Ollama).
- Streamlit demo with a results dashboard and a query playground, using precomputed embeddings.
- CI: ruff and pytest on every push and pull request.
