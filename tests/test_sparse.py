from hybridrag.retrieval.sparse import SparseRetriever

CORPUS = {
    "d1": {"title": "", "text": "the cat sat on the mat"},
    "d2": {"title": "", "text": "dogs are loyal companions"},
    "d3": {"title": "", "text": "cats and dogs can be friends"},
}


def test_ranks_lexically_matching_doc_first():
    retriever = SparseRetriever(CORPUS)

    hits = retriever.search("cat", top_k=3)
    hit_ids = [doc_id for doc_id, _ in hits]

    assert hit_ids[0] in {"d1", "d3"}
    assert set(hit_ids) == {"d1", "d2", "d3"}


def test_top_k_limits_results():
    retriever = SparseRetriever(CORPUS)

    hits = retriever.search("dog", top_k=1)

    assert len(hits) == 1
