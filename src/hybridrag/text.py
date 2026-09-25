"""Shared helper for turning a BEIR corpus document into indexable/displayable text."""

from __future__ import annotations


def doc_text(doc: dict[str, str]) -> str:
    return f"{doc.get('title', '')} {doc.get('text', '')}".strip()
