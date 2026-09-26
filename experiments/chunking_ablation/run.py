"""RQ3: does chunk size affect dense retrieval quality, and does the answer
generalize across corpora the same way retrieval architecture choice did?

Chunking only runs through the dense config: chunking is specifically an
embedding-dilution concern (a long document's embedding averages over
everything in it), not something BM25's term-frequency scoring has the same
sensitivity to, and BM25 is already covered by the main retrieval ablation.

BEIR's relevance judgments are per-document, not per-chunk, so there's no
ground truth to score chunk-level hits against directly. Chunk-level results
are mapped back to their parent document (hybridrag.chunking.dedupe_to_documents,
keeping each document's best-scoring chunk) before scoring with the same
nDCG/Recall/MRR metrics the main ablation uses, so these numbers are directly
comparable to that table's "dense" row, which is reused here as the
whole-document baseline instead of being recomputed.

Skips variants a dataset already has results for, same as the main ablation.

Usage:
    python experiments/chunking_ablation/run.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hybridrag.chunking import chunk_corpus, dedupe_to_documents  # noqa: E402
from hybridrag.data.beir_loader import load_beir_dataset  # noqa: E402
from hybridrag.metrics import evaluate  # noqa: E402
from hybridrag.retrieval.dense import DenseRetriever  # noqa: E402

DATASETS = ["scifact", "fiqa"]
CHUNK_SIZES = {"chunks_100": 100, "chunks_250": 250}
MAX_QUERIES = 200
TOP_K = 100
CHUNK_SEARCH_TOP_K = 1000  # over-fetch chunks so dedup still yields TOP_K unique documents
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_PATH = RESULTS_DIR / "results.json"
ABLATION_RESULTS_PATH = Path(__file__).resolve().parents[1] / "retrieval_ablation" / "results" / "results.json"


def whole_document_baseline(dataset_name: str) -> dict[str, float]:
    """Reuses the main ablation's already-computed "dense" result instead of
    recomputing the same thing here.
    """
    ablation_results = json.loads(ABLATION_RESULTS_PATH.read_text())
    return ablation_results[dataset_name]["dense"]


def run_variant(dataset, query_ids: list[str], qrels_subset: dict, chunk_size: int) -> dict[str, float]:
    chunked_corpus = chunk_corpus(dataset.corpus, chunk_size)
    retriever = DenseRetriever(chunked_corpus)

    run: dict[str, dict[str, float]] = {}
    for qid in query_ids:
        chunk_hits = retriever.search(dataset.queries[qid], CHUNK_SEARCH_TOP_K)
        run[qid] = dict(dedupe_to_documents(chunk_hits, TOP_K))

    return evaluate(qrels_subset, run)


def run_dataset(name: str, existing: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    results = dict(existing)
    results.setdefault("whole_document", whole_document_baseline(name))

    missing = [variant for variant in CHUNK_SIZES if variant not in results]
    if not missing:
        print(f"[{name}] all variants already computed, skipping")
        return results

    print(f"[{name}] loading...")
    dataset = load_beir_dataset(name)
    query_ids = sorted(dataset.qrels.keys())[:MAX_QUERIES]
    qrels_subset = {qid: dataset.qrels[qid] for qid in query_ids}

    for variant in missing:
        chunk_size = CHUNK_SIZES[variant]
        print(f"[{name}] running variant={variant} (chunk_size={chunk_size}) over {len(query_ids)} queries...")
        results[variant] = run_variant(dataset, query_ids, qrels_subset, chunk_size)
        print(f"[{name}] {variant}: {results[variant]}")

    return results


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {}

    for name in DATASETS:
        all_results[name] = run_dataset(name, all_results.get(name, {}))
        RESULTS_PATH.write_text(json.dumps(all_results, indent=2))

    print(f"wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
