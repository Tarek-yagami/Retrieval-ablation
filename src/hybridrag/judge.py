"""LLM-as-judge: scores a generated answer for faithfulness (is it actually
supported by the retrieved context, not hallucinated) and relevance (does it
address the question asked). Two separate, narrow questions rather than one
vague "rate this answer" prompt, matching the same review-conservatively
spirit as the validation layer this borrows from. The judge should only
flag what it can support, not offer a fuzzy overall verdict.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from json_repair import repair_json

from hybridrag.llm import complete

_JUDGE_PROMPT = """You are grading a RAG system's answer. Score it on two axes, \
1 (worst) to 5 (best):

- faithfulness: is every claim in the answer actually supported by the context? \
  A correct-sounding claim not found in the context still scores low.
- relevance: does the answer actually address the question asked?

Context:
{context}

Question: {question}

Answer: {answer}

Respond with ONLY a JSON object, no other text:
{{"faithfulness": <1-5>, "relevance": <1-5>, "reasoning": "<one sentence>"}}"""


@dataclass
class JudgeScore:
    faithfulness: int
    relevance: int
    reasoning: str


def judge_answer(question: str, answer: str, passages: list[str], *, model: str | None = None) -> JudgeScore:
    context = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    prompt = _JUDGE_PROMPT.format(context=context, question=question, answer=answer)
    raw = complete(prompt, model=model)

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"judge did not return parseable JSON: {raw!r}")

    # Weaker/local judge models sometimes emit near-JSON (a missing comma, a
    # stray quote). repair_json fixes that instead of failing the whole run
    # on one malformed response.
    parsed = repair_json(match.group(0), return_objects=True)
    if not isinstance(parsed, dict) or "faithfulness" not in parsed:
        raise ValueError(f"judge did not return parseable JSON: {raw!r}")

    return JudgeScore(
        faithfulness=int(parsed["faithfulness"]),
        relevance=int(parsed["relevance"]),
        reasoning=str(parsed["reasoning"]),
    )
