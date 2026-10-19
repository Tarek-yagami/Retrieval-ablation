import pytest

from hybridrag import llm, query_rewrite


def test_hypothetical_passage_prompt_contains_query(monkeypatch):
    captured = {}

    def fake_complete(prompt, **kwargs):
        captured["prompt"] = prompt
        return "  Statins are not linked to breast cancer.  "

    monkeypatch.setattr(query_rewrite, "complete", fake_complete)

    passage = query_rewrite.write_hypothetical_passage("do statins cause breast cancer?")

    assert passage == "Statins are not linked to breast cancer."
    assert "do statins cause breast cancer?" in captured["prompt"]


def test_paraphrases_requested_with_schema_and_parsed(monkeypatch):
    captured = {}

    def fake_complete(prompt, **kwargs):
        captured.update(kwargs)
        return '{"paraphrases": ["statins and breast cancer risk", " statin carcinogenicity ", "cholesterol cancer"]}'

    monkeypatch.setattr(query_rewrite, "complete", fake_complete)

    paraphrases = query_rewrite.paraphrase_query("do statins cause breast cancer?", 3)

    assert paraphrases == ["statins and breast cancer risk", "statin carcinogenicity", "cholesterol cancer"]
    assert captured["response_format"] is query_rewrite._Paraphrases


def test_extra_paraphrases_are_dropped(monkeypatch):
    monkeypatch.setattr(query_rewrite, "complete", lambda *a, **k: '{"paraphrases": ["a", "b", "c", "d"]}')

    assert query_rewrite.paraphrase_query("q", 3) == ["a", "b", "c"]


def test_truncated_json_is_retried(monkeypatch):
    truncated = '{\n"paraphrases": [\n    "domoic acid toxicity \xa0 \xa0 \xa0'
    replies = iter([truncated, '{"paraphrases": ["a", "b", "c"]}'])
    monkeypatch.setattr(query_rewrite, "complete", lambda *a, **k: next(replies))

    assert query_rewrite.paraphrase_query("domoic acid", 3) == ["a", "b", "c"]


def test_too_few_paraphrases_raises_after_retries(monkeypatch):
    calls = []

    def fake_complete(*args, **kwargs):
        calls.append(1)
        return '{"paraphrases": ["a", "  "]}'

    monkeypatch.setattr(query_rewrite, "complete", fake_complete)

    with pytest.raises(ValueError, match="expected 3 paraphrases"):
        query_rewrite.paraphrase_query("q", 3)
    assert len(calls) == query_rewrite._PARAPHRASE_ATTEMPTS


def test_complete_forwards_response_format_to_litellm(monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return {"choices": [{"message": {"content": '{"paraphrases": []}'}}]}

    monkeypatch.setattr(llm.litellm, "completion", fake_completion)

    llm.complete("prompt", model="gpt-4o", response_format=query_rewrite._Paraphrases)

    assert captured["response_format"] is query_rewrite._Paraphrases
