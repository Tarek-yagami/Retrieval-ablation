"""RQ2: does better retrieval actually lead to better-judged answers, not
just better IR metrics? Generates and judges answers on a fixed FiQA question
sample for two configs: sparse (baseline) and dense_rerank (the ablation's
best performer), so the two experiments' claims connect to each other.

Requires an LLM configured via HYBRIDRAG_MODEL (see hybridrag/llm.py).
This is the only experiment that spends API credits, and it's fixed at a
small sample size on purpose to keep that cost negligible.

Unlike the ablation experiment, this one indexes a reduced corpus: every
document relevant to the sampled queries, plus a deterministic random
background sample, rather than the full ~57k-doc FiQA corpus. That's a
correctness requirement for the ablation's IR metrics (recall@100 means
nothing over a shrunk corpus) but not for this experiment, whose only job is
generating and judging answers over a realistic-but-small retrieval pool.

Retrieval and generation also run as two separate phases rather than
interleaved: the retrieval models (embeddings + cross-encoder) are freed
from memory before any LLM call starts, instead of both being resident at
once. That's the difference between fitting in memory alongside a local LLM
and not. Retrieved passages are cached to disk between phases so a failure
during the (slower, more failure-prone) generation phase never requires
re-running retrieval, and results are written incrementally so a crash
partway through doesn't lose already-judged answers.

Usage:
    python experiments/answer_quality/run.py
"""

from __future__ import annotations

import gc
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hybridrag.data.beir_loader import load_beir_dataset  # noqa: E402
from hybridrag.generation import generate_answer  # noqa: E402
from hybridrag.judge import judge_answer  # noqa: E402
from hybridrag.pipeline import RetrievalPipeline  # noqa: E402
from hybridrag.text import doc_text  # noqa: E402

DATASET = "fiqa"
CONFIGS = ["sparse", "dense_rerank"]
SAMPLE_SIZE = 40
BACKGROUND_CORPUS_SIZE = 5000
TOP_K_CONTEXT = 5
RESULTS_DIR = Path(__file__).parent / "results"
RETRIEVAL_CACHE_PATH = RESULTS_DIR / "retrieval_cache.json"
RESULTS_PATH = RESULTS_DIR / "results.json"
_SEED = 0


def build_reduced_corpus(dataset, query_ids: list[str]) -> dict[str, dict[str, str]]:
    relevant_ids = {doc_id for qid in query_ids for doc_id in dataset.qrels[qid]}

    background_pool = sorted(set(dataset.corpus.keys()) - relevant_ids)
    rng = random.Random(_SEED)
    background = rng.sample(background_pool, min(BACKGROUND_CORPUS_SIZE, len(background_pool)))

    keep_ids = relevant_ids | set(background)
    return {doc_id: dataset.corpus[doc_id] for doc_id in keep_ids}


def run_retrieval_phase() -> list[dict]:
    """Retrieves passages for every (question, config) pair, then frees the
    retrieval models before returning, so they're not resident in memory
    during the generation phase.
    """
    print(f"[{DATASET}] loading...")
    dataset = load_beir_dataset(DATASET)
    query_ids = sorted(dataset.qrels.keys())[:SAMPLE_SIZE]

    corpus = build_reduced_corpus(dataset, query_ids)
    print(f"[{DATASET}] indexing {len(corpus)} docs (reduced from {len(dataset.corpus)})...")
    pipeline = RetrievalPipeline(corpus)

    cache = []
    for i, qid in enumerate(query_ids, start=1):
        question = dataset.queries[qid]
        print(f"[retrieval {i}/{len(query_ids)}] {question[:60]!r}")
        for config in CONFIGS:
            hits = pipeline.search(question, config, top_k=TOP_K_CONTEXT)
            passages = [doc_text(corpus[doc_id]) for doc_id, _ in hits]
            cache.append({"query_id": qid, "question": question, "config": config, "passages": passages})

    del pipeline
    gc.collect()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RETRIEVAL_CACHE_PATH.write_text(json.dumps(cache, indent=2))
    print(f"wrote {RETRIEVAL_CACHE_PATH}")
    return cache


def run_generation_phase(cache: list[dict]) -> None:
    records = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else []
    done = {(r["query_id"], r["config"]) for r in records}

    for i, item in enumerate(cache, start=1):
        key = (item["query_id"], item["config"])
        if key in done:
            continue

        print(f"[generation {i}/{len(cache)}] {item['question'][:60]!r} ({item['config']})")
        answer = generate_answer(item["question"], item["passages"])
        score = judge_answer(item["question"], answer, item["passages"])

        records.append(
            {
                "query_id": item["query_id"],
                "question": item["question"],
                "config": item["config"],
                "answer": answer,
                "faithfulness": score.faithfulness,
                "relevance": score.relevance,
                "reasoning": score.reasoning,
            }
        )
        RESULTS_PATH.write_text(json.dumps(records, indent=2))

    print(f"wrote {RESULTS_PATH}")


def _load_cache_if_current() -> list[dict] | None:
    """Returns the cached retrieval results only if they were built for the
    exact CONFIGS this run expects. Otherwise a change to CONFIGS (like
    swapping which config is compared against sparse) would silently reuse
    stale passages from a previous run instead of retrieving fresh ones.
    """
    if not RETRIEVAL_CACHE_PATH.exists():
        return None
    cache = json.loads(RETRIEVAL_CACHE_PATH.read_text())
    cached_configs = {item["config"] for item in cache}
    if cached_configs != set(CONFIGS):
        print(f"retrieval cache is for {sorted(cached_configs)}, expected {CONFIGS}, rebuilding")
        return None
    return cache


def main() -> None:
    cache = _load_cache_if_current() or run_retrieval_phase()
    run_generation_phase(cache)


if __name__ == "__main__":
    main()
