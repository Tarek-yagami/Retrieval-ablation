from hybridrag.metrics import evaluate


def test_perfect_ranking_scores_maximally():
    qrels = {"q1": {"d1": 1}}
    run = {"q1": {"d1": 1.0, "d2": 0.5}}

    scores = evaluate(qrels, run)

    assert scores["ndcg@10"] == 1.0
    assert scores["recall@100"] == 1.0
    assert scores["mrr@10"] == 1.0


def test_relevant_doc_missing_from_run_scores_zero():
    qrels = {"q1": {"d1": 1}}
    run = {"q1": {"d2": 1.0}}

    scores = evaluate(qrels, run)

    assert scores["ndcg@10"] == 0.0
    assert scores["recall@100"] == 0.0
    assert scores["mrr@10"] == 0.0


def test_averages_across_queries():
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}}
    run = {
        "q1": {"d1": 1.0},  # perfect
        "q2": {"other": 1.0},  # miss
    }

    scores = evaluate(qrels, run)

    assert scores["recall@100"] == 0.5
