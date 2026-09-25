import pytest

from hybridrag import judge


def test_parses_well_formed_json(monkeypatch):
    monkeypatch.setattr(
        judge,
        "complete",
        lambda *a, **k: '{"faithfulness": 4, "relevance": 5, "reasoning": "grounded and on-topic"}',
    )

    score = judge.judge_answer("q", "a", ["context"])

    assert score.faithfulness == 4
    assert score.relevance == 5
    assert score.reasoning == "grounded and on-topic"


def test_extracts_json_surrounded_by_extra_text(monkeypatch):
    monkeypatch.setattr(
        judge,
        "complete",
        lambda *a, **k: 'Sure, here is my score:\n{"faithfulness": 2, "relevance": 3, "reasoning": "partial"}\nThanks!',
    )

    score = judge.judge_answer("q", "a", ["context"])

    assert score.faithfulness == 2
    assert score.relevance == 3


def test_repairs_minor_json_malformation(monkeypatch):
    # missing comma: the kind of near-JSON a small/local judge model tends to emit
    monkeypatch.setattr(
        judge,
        "complete",
        lambda *a, **k: '{"faithfulness": 3, "relevance": 4 "reasoning": "ok"}',
    )

    score = judge.judge_answer("q", "a", ["context"])

    assert score.faithfulness == 3
    assert score.relevance == 4


def test_raises_on_unparseable_response(monkeypatch):
    monkeypatch.setattr(judge, "complete", lambda *a, **k: "I refuse to answer in JSON.")

    with pytest.raises(ValueError):
        judge.judge_answer("q", "a", ["context"])
