from hybridrag.retrieval.fusion import reciprocal_rank_fusion


def test_agreement_boosts_shared_doc():
    sparse = [("a", 5.0), ("b", 3.0), ("c", 1.0)]
    dense = [("b", 0.9), ("a", 0.8), ("d", 0.5)]

    fused = reciprocal_rank_fusion([sparse, dense])
    fused_ids = [doc_id for doc_id, _ in fused]

    # "a" and "b" appear near the top of both rankings, so RRF should rank
    # them above "c"/"d", which each appear in only one ranking.
    assert set(fused_ids[:2]) == {"a", "b"}


def test_doc_only_in_one_ranking_still_included():
    sparse = [("a", 1.0)]
    dense = [("z", 1.0)]

    fused = reciprocal_rank_fusion([sparse, dense])
    fused_ids = {doc_id for doc_id, _ in fused}

    assert fused_ids == {"a", "z"}


def test_empty_rankings_produce_empty_fusion():
    assert reciprocal_rank_fusion([[], []]) == []
