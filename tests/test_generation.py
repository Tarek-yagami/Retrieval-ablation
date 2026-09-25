from hybridrag import generation


def test_includes_all_passages_and_question_in_prompt(monkeypatch):
    captured = {}

    def fake_complete(prompt, **kwargs):
        captured["prompt"] = prompt
        return "  the answer  "

    monkeypatch.setattr(generation, "complete", fake_complete)

    result = generation.generate_answer("what is X?", ["passage one", "passage two"])

    assert result == "the answer"
    assert "what is X?" in captured["prompt"]
    assert "passage one" in captured["prompt"]
    assert "passage two" in captured["prompt"]
