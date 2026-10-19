"""Provider-agnostic completion layer: the rest of the codebase calls
`complete()` and never touches a vendor SDK directly. litellm resolves the
model string to Claude, OpenAI, Groq, or a local Ollama model, so switching
providers is a config change here, not a rewrite anywhere else.

Model is read from HYBRIDRAG_MODEL, e.g.:
    claude-sonnet-5              -> Anthropic
    gpt-4o                       -> OpenAI
    groq/openai/gpt-oss-120b     -> Groq
    ollama/llama3.1               -> local Ollama
"""

from __future__ import annotations

import os

import litellm
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

_DEFAULT_MODEL = "claude-sonnet-5"
_RETRYABLE_ERRORS = (litellm.RateLimitError, litellm.APIConnectionError)
_MAX_ATTEMPTS = 8
_MAX_WAIT_SECONDS = 60


@retry(
    retry=retry_if_exception_type(_RETRYABLE_ERRORS),
    wait=wait_random_exponential(max=_MAX_WAIT_SECONDS),
    stop=stop_after_attempt(_MAX_ATTEMPTS),
    reraise=True,
)
def complete(
    prompt: str,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    response_format: type[BaseModel] | None = None,
) -> str:
    """Return the model's reply; with response_format, the reply is JSON matching that schema."""
    resolved_model = model or os.environ.get("HYBRIDRAG_MODEL", _DEFAULT_MODEL)

    extra_kwargs = {}
    if response_format is not None:
        extra_kwargs["response_format"] = response_format
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
