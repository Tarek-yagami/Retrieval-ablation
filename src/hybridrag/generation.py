"""Generates an answer to a query from a fixed set of retrieved passages,
via the provider-agnostic LLM layer in llm.py.
"""

from __future__ import annotations

from hybridrag.llm import complete

_PROMPT_TEMPLATE = """Answer the question using ONLY the context below. If the \
context doesn't contain the answer, say so instead of guessing.

Context:
{context}

Question: {question}

Answer:"""


def generate_answer(question: str, passages: list[str], *, model: str | None = None) -> str:
    context = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    prompt = _PROMPT_TEMPLATE.format(context=context, question=question)
    return complete(prompt, model=model).strip()
