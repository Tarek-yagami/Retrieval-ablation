"""Provider-agnostic completion layer: the rest of the codebase calls
`complete()` and never touches a vendor SDK directly. litellm resolves the
model string to Claude, OpenAI, Groq, or a local Ollama model, so switching
providers is a config change here, not a rewrite anywhere else.

Model is read from HYBRIDRAG_MODEL, e.g.:
    claude-sonnet-5              -> Anthropic
    gpt-4o                       -> OpenAI
    groq/llama-3.3-70b-versatile -> Groq
    ollama/llama3.1               -> local Ollama
"""

from __future__ import annotations

import os

import litellm

_DEFAULT_MODEL = "claude-sonnet-5"


def complete(prompt: str, *, model: str | None = None, temperature: float = 0.0, max_tokens: int = 1024) -> str:
    resolved_model = model or os.environ.get("HYBRIDRAG_MODEL", _DEFAULT_MODEL)

    extra_kwargs = {}
    if resolved_model.startswith("ollama/"):
        # Force CPU-only inference: on modest/shared hardware, Ollama's GPU
        # offload can fail with a CUDA host-buffer allocation error when VRAM
        # is tight, while plain CPU inference (slower, but reliable) works.
        # This also keeps memory accounting simple alongside this project's
        # own torch-based retrieval models, which already run on CPU.
        extra_kwargs["num_gpu"] = 0

    response = litellm.completion(
        model=resolved_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        **extra_kwargs,
    )
    return response["choices"][0]["message"]["content"] or ""
