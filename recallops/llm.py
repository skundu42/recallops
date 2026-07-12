"""Chat-LLM clients for golden-dataset generation (PRD FR-3.1).

Mirrors ``pipeline/providers.py``: raw HTTP via the shared ``_post_json``
(no SDK dependency, bounded timeout), an explicit price table that raises on
unknown models rather than silently mispricing the cost gate, and pure
request/parse helpers so tests exercise the wire format offline.

The LLM path is strictly opt-in (``recall dataset generate --llm openai``).
Unlike the offline heuristic generator it is NOT deterministic across runs,
even at temperature 0; the seed still controls which chunks are sampled.
Generated questions carry ``origin: synthetic`` and go through the same
dedup and curation flow as heuristic ones.
"""
from __future__ import annotations

from .pipeline.providers import _post_json

# USD per 1k (input_tokens, output_tokens), from platform.openai.com/pricing.
# OWNER: re-verify against the vendor page before billing-sensitive use (same
# caveat as the Cohere/Voyage embedding tables).
OPENAI_CHAT_PRICE_PER_1K = {
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.01),
}
DEFAULT_CHAT_MODEL = "gpt-4o-mini"
_OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
# One generated question is short; the estimate errs high on purpose.
_OUT_TOKENS_PER_QUESTION = 60
_PROMPT_OVERHEAD_TOKENS = 40


def openai_chat_request_body(model: str, prompt: str) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 120,
    }


def parse_openai_chat(payload: dict) -> str:
    return payload["choices"][0]["message"]["content"]


class OpenAIChatLLM:
    """Callable ``prompt -> completion`` for ``dataset.generate(llm=...)``."""

    def __init__(self, model: str = DEFAULT_CHAT_MODEL, api_key: str | None = None):
        if model not in OPENAI_CHAT_PRICE_PER_1K:
            raise ValueError(f"unknown OpenAI chat model: {model!r}")
        self.provider = "openai"
        self.model = model
        self._api_key = api_key

    def _key(self) -> str:
        import os

        key = self._api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set; cannot call the OpenAI chat API")
        return key

    def __call__(self, prompt: str) -> str:
        payload = _post_json(
            _OPENAI_CHAT_URL,
            openai_chat_request_body(self.model, prompt),
            {"Authorization": f"Bearer {self._key()}"},
        )
        return parse_openai_chat(payload)


def get_llm(spec: str) -> OpenAIChatLLM:
    """Parse ``provider[:model]`` (currently only ``openai``) into a client."""
    provider, _, model = spec.partition(":")
    if provider == "openai":
        return OpenAIChatLLM(model=model or DEFAULT_CHAT_MODEL)
    raise ValueError(f"unknown LLM provider {provider!r}; expected 'openai[:model]'")


def estimate_generation_cost(model: str, n_cases: int, avg_chunk_tokens: float) -> float:
    """Upper-bound USD estimate for generating ``n_cases`` questions: each
    prompt carries one chunk plus template overhead; token estimate matches
    the providers' ``len(text) // 4`` convention."""
    price_in, price_out = OPENAI_CHAT_PRICE_PER_1K[model]
    in_tokens = n_cases * (avg_chunk_tokens + _PROMPT_OVERHEAD_TOKENS)
    out_tokens = n_cases * _OUT_TOKENS_PER_QUESTION
    return in_tokens / 1000 * price_in + out_tokens / 1000 * price_out
