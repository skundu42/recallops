"""Embedding providers.

``LocalHashProvider`` is a deterministic, offline feature-hashing embedder, a
pure function of (text, model params), so every test and demo runs without
network or cost. ``OpenAIProvider``, ``CohereProvider``, and ``VoyageProvider``
are real-money paths; they share the ``_post_json`` path with a bounded timeout
to prevent CI hangs.
"""
from __future__ import annotations

import hashlib
import json
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

# Prices in USD per 1k tokens and native output dims, from the providers'
# public pricing/API docs. Unknown models raise rather than silently
# mispricing the cost gate.
COHERE_PRICE_PER_1K = {
    "embed-english-v3.0": 0.0001,
    "embed-multilingual-v3.0": 0.0001,
    "embed-v4.0": 0.00012,
}
_COHERE_DEFAULT_DIMS = {
    "embed-english-v3.0": 1024,
    "embed-multilingual-v3.0": 1024,
    "embed-v4.0": 1536,
}
_COHERE_URL = "https://api.cohere.com/v2/embed"
_COHERE_BATCH = 96

VOYAGE_PRICE_PER_1K = {
    "voyage-3.5": 0.00006,
    "voyage-3.5-lite": 0.00002,
    "voyage-3-large": 0.00018,
}
_VOYAGE_DEFAULT_DIMS = {
    "voyage-3.5": 1024,
    "voyage-3.5-lite": 1024,
    "voyage-3-large": 1024,
}
_VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"
# Voyage's embeddings API accepts up to 1,000 texts per request (confirmed
# against docs.voyageai.com/docs/embeddings, 2026-07); the brief's draft
# value of 128 was overly conservative and has been corrected here.
_VOYAGE_BATCH = 1000
# Per-request TOKEN caps also apply and are model-specific (docs.voyageai.com
# /docs/embeddings, 2026-07): 320K tokens for voyage-3.5, 120K for
# voyage-3-large, 1M for voyage-3.5-lite. Default chunks run up to ~800
# tokens, so a full 1,000-text batch can be ~800K tokens - well over the
# tightest (voyage-3-large) cap. 100K keeps every batch under that cap with
# a conservative margin, regardless of which Voyage model is in use.
_VOYAGE_TOKEN_BUDGET = 100_000


def _voyage_batches(texts: list[str]) -> list[list[str]]:
    """Greedily group ``texts`` into Voyage request batches, flushing the
    current batch when the next text would push it over the 1,000-text
    count cap OR the estimated-token budget (whichever comes first). A
    single text whose own estimate already exceeds the token budget is
    sent alone rather than blocking forever. Token estimate matches
    ``estimate_embed_cost``: ``len(text) // 4``.
    """
    batches: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for text in texts:
        est = len(text) // 4
        if current and (
            len(current) >= _VOYAGE_BATCH or current_tokens + est > _VOYAGE_TOKEN_BUDGET
        ):
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(text)
        current_tokens += est
    if current:
        batches.append(current)
    return batches


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


def _http_timeout() -> float:
    """Seconds before any provider HTTP call is abandoned (RECALL_HTTP_TIMEOUT
    overrides). Without a bound, one stalled provider connection hangs an
    ingest or CI run forever, which breaks the <5-min CI budget (PRD section 12)."""
    import os

    return float(os.environ.get("RECALL_HTTP_TIMEOUT", "60"))


def _post_json(url: str, body: dict, headers: dict, timeout: float | None = None) -> dict:
    import urllib.request

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout if timeout is not None else _http_timeout()) as resp:
        return json.loads(resp.read().decode("utf-8"))


def openai_request_body(model: str, texts: list[str], dims: int) -> dict:
    return {"model": model, "input": list(texts), "dimensions": int(dims)}


def parse_openai_embeddings(payload: dict) -> list[list[float]]:
    return [d["embedding"] for d in sorted(payload["data"], key=lambda d: d["index"])]


def cohere_request_body(model: str, texts: list[str], input_type: str) -> dict:
    return {"model": model, "texts": list(texts), "input_type": input_type,
            "embedding_types": ["float"]}


def parse_cohere_embeddings(payload: dict) -> list[list[float]]:
    return payload["embeddings"]["float"]


def voyage_request_body(model: str, texts: list[str], input_type: str) -> dict:
    return {"model": model, "input": list(texts), "input_type": input_type}


def parse_voyage_embeddings(payload: dict) -> list[list[float]]:
    return [d["embedding"] for d in sorted(payload["data"], key=lambda d: d["index"])]


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


ST_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_ST_DEFAULT_DIMS = {"all-MiniLM-L6-v2": 384}


class SentenceTransformersProvider(EmbeddingProvider):
    """Local real-semantics embeddings at $0 via sentence-transformers.

    Device defaults to CPU: GPU kernels are nondeterministic and identical
    inputs must produce identical vectors for content-addressing.
    ``revision`` pins the HuggingFace model revision and is recorded in
    ``params`` (hence the manifest); the default "main" is only
    reproducible while the upstream model repo does not move, so pin a
    commit for long-lived baselines.
    """

    def __init__(self, model: str = ST_DEFAULT_MODEL, dims: int | None = None,
                 device: str = "cpu", revision: str | None = None):
        if dims is None:
            if model not in _ST_DEFAULT_DIMS:
                raise ValueError(
                    f"unknown sentence-transformers model {model!r}: pass dims explicitly"
                )
            dims = _ST_DEFAULT_DIMS[model]
        self.provider = "st"
        self.model = model
        self.dims = int(dims)
        self.device = str(device)
        self.revision = str(revision) if revision else "main"
        self.params = {"dims": self.dims, "device": self.device, "revision": self.revision}
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is not installed; run: pip install 'recallops[st]'"
                ) from exc
            model = SentenceTransformer(self.model, device=self.device,
                                        revision=self.revision)
            # get_sentence_embedding_dimension was renamed to
            # get_embedding_dimension in sentence-transformers 5.x; prefer the
            # new name when present and fall back to the old one so both
            # pre- and post-rename releases work without a FutureWarning.
            dims_getter = getattr(model, "get_embedding_dimension", None)
            if dims_getter is None:
                dims_getter = model.get_sentence_embedding_dimension
            got = int(dims_getter())
            if got != self.dims:
                raise ValueError(
                    f"sentence-transformers model {self.model!r} emits {got} dims, "
                    f"config says {self.dims}"
                )
            self._model = model
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        mat = model.encode(list(texts), batch_size=32, convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=False)
        mat = np.asarray(mat, dtype=np.float32).reshape(len(texts), self.dims)
        return _l2_normalize(mat)

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
        rows: list[list[float]] = []
        for start in range(0, len(texts), _OPENAI_BATCH):
            batch = texts[start:start + _OPENAI_BATCH]
            payload = _post_json(
                _OPENAI_URL,
                openai_request_body(self.model, batch, self.dims),
                {"Authorization": f"Bearer {key}"},
            )
            rows.extend(parse_openai_embeddings(payload))
        mat = np.array(rows, dtype=np.float32).reshape(len(texts), self.dims)
        return _l2_normalize(mat)

    def price_per_1k_tokens(self) -> float:
        return OPENAI_PRICE_PER_1K[self.model]


class CohereProvider(EmbeddingProvider):
    """Cohere v2 embeddings over raw HTTP (no SDK dependency).

    Cohere embeds queries and documents differently (``input_type``), so
    ``embed_queries`` is a real override, not the default delegation.
    """

    def __init__(self, model: str = "embed-english-v3.0", dims: int | None = None,
                 api_key: str | None = None):
        if model not in COHERE_PRICE_PER_1K:
            raise ValueError(f"unknown Cohere embedding model: {model!r}")
        expected = _COHERE_DEFAULT_DIMS[model]
        if dims is not None and int(dims) != expected:
            raise ValueError(
                f"Cohere model {model!r} emits {expected} dims, got dims={dims}"
            )
        self.provider = "cohere"
        self.model = model
        self.dims = expected
        self.params = {"dims": self.dims}
        self._api_key = api_key

    def _key(self) -> str:
        import os

        key = self._api_key or os.environ.get("COHERE_API_KEY")
        if not key:
            raise RuntimeError(
                "COHERE_API_KEY is not set; cannot call the Cohere embeddings API"
            )
        return key

    def _embed_as(self, texts: list[str], input_type: str) -> np.ndarray:
        key = self._key()
        rows: list[list[float]] = []
        for start in range(0, len(texts), _COHERE_BATCH):
            batch = texts[start:start + _COHERE_BATCH]
            payload = _post_json(
                _COHERE_URL,
                cohere_request_body(self.model, batch, input_type),
                {"Authorization": f"Bearer {key}"},
            )
            rows.extend(parse_cohere_embeddings(payload))
        mat = np.array(rows, dtype=np.float32).reshape(len(texts), self.dims)
        return _l2_normalize(mat)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._embed_as(texts, "search_document")

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._embed_as(texts, "search_query")

    def price_per_1k_tokens(self) -> float:
        return COHERE_PRICE_PER_1K[self.model]


class VoyageProvider(EmbeddingProvider):
    """Voyage AI embeddings over raw HTTP (no SDK dependency).

    Voyage embeds queries and documents differently (``input_type``), so
    ``embed_queries`` is a real override, not the default delegation.
    """

    def __init__(self, model: str = "voyage-3.5", dims: int | None = None,
                 api_key: str | None = None):
        if model not in VOYAGE_PRICE_PER_1K:
            raise ValueError(f"unknown Voyage embedding model: {model!r}")
        expected = _VOYAGE_DEFAULT_DIMS[model]
        if dims is not None and int(dims) != expected:
            raise ValueError(
                f"Voyage model {model!r} emits {expected} dims, got dims={dims}"
            )
        self.provider = "voyage"
        self.model = model
        self.dims = expected
        self.params = {"dims": self.dims}
        self._api_key = api_key

    def _key(self) -> str:
        import os

        key = self._api_key or os.environ.get("VOYAGE_API_KEY")
        if not key:
            raise RuntimeError(
                "VOYAGE_API_KEY is not set; cannot call the Voyage embeddings API"
            )
        return key

    def _embed_as(self, texts: list[str], input_type: str) -> np.ndarray:
        key = self._key()
        rows: list[list[float]] = []
        for batch in _voyage_batches(list(texts)):
            payload = _post_json(
                _VOYAGE_URL,
                voyage_request_body(self.model, batch, input_type),
                {"Authorization": f"Bearer {key}"},
            )
            rows.extend(parse_voyage_embeddings(payload))
        mat = np.array(rows, dtype=np.float32).reshape(len(texts), self.dims)
        return _l2_normalize(mat)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._embed_as(texts, "document")

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._embed_as(texts, "query")

    def price_per_1k_tokens(self) -> float:
        return VOYAGE_PRICE_PER_1K[self.model]


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
    if provider == "cohere":
        return CohereProvider(model=model, dims=dims)
    if provider == "voyage":
        return VoyageProvider(model=model, dims=dims)
    if provider == "st":
        return SentenceTransformersProvider(
            model=model, dims=dims,
            device=str(spec.get("device", "cpu")),
            revision=spec.get("revision"),
        )
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
