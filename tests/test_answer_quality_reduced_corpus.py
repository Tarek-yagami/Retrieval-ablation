import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "answer_quality"))

from run import build_reduced_corpus  # noqa: E402


@dataclass
class _FakeDataset:
    corpus: dict
    qrels: dict


def test_keeps_every_relevant_doc_and_caps_background():
    corpus = {f"d{i}": {"title": "", "text": f"doc {i}"} for i in range(50)}
    qrels = {"q1": {"d0": 1, "d1": 1}, "q2": {"d2": 1}}
    dataset = _FakeDataset(corpus=corpus, qrels=qrels)

    reduced = build_reduced_corpus(dataset, ["q1", "q2"])

    assert {"d0", "d1", "d2"} <= reduced.keys()


def test_deterministic_across_calls():
    corpus = {f"d{i}": {"title": "", "text": f"doc {i}"} for i in range(200)}
    qrels = {"q1": {"d0": 1}}
    dataset = _FakeDataset(corpus=corpus, qrels=qrels)

    first = build_reduced_corpus(dataset, ["q1"])
    second = build_reduced_corpus(dataset, ["q1"])

    assert set(first.keys()) == set(second.keys())
