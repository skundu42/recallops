"""Embedding providers.

``LocalHashProvider`` is a deterministic, offline feature-hashing embedder, a
pure function of (text, model params), so every test and demo runs without
network or cost. ``OpenAIProvider`` is the real-money path; it is lazy and is
never exercised in tests.
"""
from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from collections import Counter

import numpy as np

from .. import hashing
from ..models import StageSpec

_WORD_RE = re.compile(r"[a-z0-9]+")

OPENAI_PRICE_PER_1K = {
    "text-embedding-3-small": 0.00002,
    "text-embedding-3-large": 0.00013,
}
_OPENAI_DEFAULT_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}
_OPENAI_URL = "https://api.openai.com/v1/embeddings"
_OPENAI_BATCH = 256
_REMOTE_TOKENS_PER_S = 5000.0
_LOCAL_TOKENS_PER_S = 200_000.0


class EmbeddingProvider(ABC):
    provider: str
    model: str
    dims: int
    params: dict

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return (n, dims) fp32 L2-normalized vectors."""

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        """Embed retrieval *queries*. Defaults to ``embed`` (documents);
        providers whose models distinguish query from document inputs
        (Cohere/Voyage ``input_type``) override this."""
        return self.embed(texts)

    @abstractmethod
    def price_per_1k_tokens(self) -> float: ...

    @property
    def model_key(self) -> str:
        return f"{self.provider}_{self.model}_{self.dims}_{hashing.params_hash(self.params)}"


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (mat / norms).astype(np.float32)


class LocalHashProvider(EmbeddingProvider):
    def __init__(self, dims: int = 256, seed: int = 0,
                 ngram: tuple[int, int] = (1, 2), model: str = "hash-v1"):
        self.provider = "local"
        self.model = model
        self.dims = int(dims)
        self.seed = int(seed)
        self.ngram = (int(ngram[0]), int(ngram[1]))
        self.params = {"dims": self.dims, "seed": self.seed, "ngram": list(self.ngram)}
        self._salt = f"{self.model}\x1f{self.seed}\x1f".encode()

    def _features(self, text: str) -> Counter[str]:
        words = _WORD_RE.findall(text.lower())
        feats: Counter[str] = Counter()
        lo, hi = self.ngram
        for n in range(lo, hi + 1):
            for i in range(len(words) - n + 1):
                feats[" ".join(words[i:i + n])] += 1
        return feats

    def _bucket_sign(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(self._salt + feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        return (value >> 1) % self.dims, 1.0 if value & 1 else -1.0

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dims), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature, count in self._features(text).items():
                bucket, sign = self._bucket_sign(feature)
                out[row, bucket] += sign * count
        return _l2_normalize(out)

    def price_per_1k_tokens(self) -> float:
        return 0.0


class OpenAIProvider(EmbeddingProvider):
    def __init__(self, model: str = "text-embedding-3-small", dims: int | None = None,
                 api_key: str | None = None):
        if model not in OPENAI_PRICE_PER_1K:
            raise ValueError(f"unknown OpenAI embedding model: {model!r}")
        self.provider = "openai"
        self.model = model
        self.dims = int(dims) if dims is not None else _OPENAI_DEFAULT_DIMS[model]
        self.params = {"dims": self.dims}
        self._api_key = api_key

    def _key(self) -> str:
        import os

        key = self._api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set; cannot call the OpenAI embeddings API")
        return key

    def embed(self, texts: list[str]) -> np.ndarray:
        key = self._key()
        import json
        import urllib.request

        rows: list[list[float]] = []
        for start in range(0, len(texts), _OPENAI_BATCH):
            batch = texts[start:start + _OPENAI_BATCH]
            body = json.dumps({"model": self.model, "input": batch, "dimensions": self.dims})
            req = urllib.request.Request(
                _OPENAI_URL,
                data=body.encode("utf-8"),
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            rows.extend(d["embedding"] for d in sorted(payload["data"], key=lambda d: d["index"]))
        mat = np.array(rows, dtype=np.float32).reshape(len(texts), self.dims)
        return _l2_normalize(mat)

    def price_per_1k_tokens(self) -> float:
        return OPENAI_PRICE_PER_1K[self.model]


def embed_stage_spec(provider: str, model: str, dims: int, params: dict) -> StageSpec:
    merged = {"provider": provider, "model": model, "dims": int(dims)}
    for k, v in params.items():
        if k in merged:
            continue
        merged[k] = list(v) if isinstance(v, tuple) else v
    return StageSpec(id="embed", tool=provider, version="1", params=merged, inputs=("chunk",))


def get_provider(spec: dict) -> EmbeddingProvider:
    provider = spec.get("provider") or spec.get("tool")
    model = spec["model"]
    dims = int(spec["dims"])
    if provider == "local":
        ngram = spec.get("ngram", (1, 2))
        return LocalHashProvider(
            dims=dims,
            seed=int(spec.get("seed", 0)),
            ngram=(int(ngram[0]), int(ngram[1])),
            model=model,
        )
    if provider == "openai":
        return OpenAIProvider(model=model, dims=dims)
    raise ValueError(f"unknown embedding provider: {provider!r}")


def estimate_embed_cost(provider: EmbeddingProvider, texts: list[str]) -> dict:
    est_tokens = sum(len(t) // 4 for t in texts)
    price = provider.price_per_1k_tokens()
    rate = _LOCAL_TOKENS_PER_S if price == 0.0 else _REMOTE_TOKENS_PER_S
    return {
        "n_texts": len(texts),
        "est_tokens": est_tokens,
        "usd": est_tokens / 1000 * price,
        "wall_s": est_tokens / rate,
    }
