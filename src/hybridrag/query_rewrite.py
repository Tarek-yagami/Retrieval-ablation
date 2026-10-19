"""LLM query rewriting before retrieval: HyDE passages and query paraphrases."""

from __future__ import annotations

from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt

from hybridrag.llm import complete

_HYDE_MAX_TOKENS = 120
_PARAPHRASE_MAX_TOKENS = 300
# at 0.0, small local models can loop on a typo'd query until Ollama aborts the generation
_PARAPHRASE_TEMPERATURE = 0.3
# a runaway generation can still truncate the JSON; each retry draws a new sample
_PARAPHRASE_ATTEMPTS = 3

_HYDE_PROMPT = """Write a short passage, 2 or 3 sentences, that answers the \
question below. Write it in the style of a document that would contain the \
answer. Output only the passage as plain text, with no title, heading or \
markdown.

Question: {query}"""

_PARAPHRASE_PROMPT = """Rewrite the search query below in {n} different ways. \
Keep the meaning, but vary the wording and vocabulary, including the terms an \
expert would use.

Query: {query}"""


class _Paraphrases(BaseModel):
    paraphrases: list[str]


def write_hypothetical_passage(query: str, *, model: str | None = None) -> str:
    """Return an LLM-written passage that answers the query, for HyDE retrieval."""
    prompt = _HYDE_PROMPT.format(query=query)
    return complete(prompt, model=model, max_tokens=_HYDE_MAX_TOKENS).strip()


@retry(
    retry=retry_if_exception_type(ValueError),
    stop=stop_after_attempt(_PARAPHRASE_ATTEMPTS),
    reraise=True,
)
def paraphrase_query(query: str, n: int, *, model: str | None = None) -> list[str]:
    """Return n LLM-written rewordings of the query."""
    prompt = _PARAPHRASE_PROMPT.format(query=query, n=n)
    raw = complete(
        prompt,
        model=model,
        temperature=_PARAPHRASE_TEMPERATURE,
        max_tokens=_PARAPHRASE_MAX_TOKENS,
        response_format=_Paraphrases,
    )
    paraphrases = [p.strip() for p in _Paraphrases.model_validate_json(raw).paraphrases if p.strip()]
    if len(paraphrases) < n:
        raise ValueError(f"expected {n} paraphrases of {query!r}, got {len(paraphrases)}: {raw!r}")
    return paraphrases[:n]
