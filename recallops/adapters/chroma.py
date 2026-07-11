"""Chroma adapter (PRD FR-12), embedded via ``chromadb.PersistentClient``.

Chroma's cosine space returns *distance* (1 - similarity); ``query_dense``
converts back so callers always see cosine similarity, higher is better,
matching the built-in local adapter. Chroma metadata values must be
scalars, so the engine payload is stored as one JSON field
(``payload_json``); ``query_dense`` never needs to read it back.

``ensure_collection``/``upsert`` go through ``get_or_create_collection`` (a
missing collection is expected there), but ``query_dense`` fetches with the
strict ``get_collection`` and raises ``KeyError`` on a missing collection,
matching ``LocalIndexAdapter``: silently get-or-creating on a query would
mask a caller bug (querying before ingest) as an empty result instead of
failing loudly.

``hnsw:search_ef`` is pinned high at collection creation so the tiny
collections used in tests and small projects are effectively exact; HNSW
approximation at scale is quarantined by funnel shadow scoring (FR-6.2)
like every other ANN backend. ``chromadb`` is an optional dependency
(extra ``[chroma]``), imported lazily so this module always imports
cleanly.
"""
from __future__ import annotations

import json

import numpy as np

from .base import Capability, VectorAdapter

_COLLECTION_METADATA = {"hnsw:space": "cosine", "hnsw:search_ef": 128}


def distance_to_score(distance: float) -> float:
    """Cosine distance -> cosine similarity."""
    return 1.0 - float(distance)


def payload_metadata(payload: dict) -> dict:
    """Chroma metadata values must be scalars; carry the payload as JSON."""
    return {"payload_json": json.dumps(payload, sort_keys=True)}


def _is_missing_collection_error(exc: Exception) -> bool:
    """True iff ``exc`` signals that a collection does not exist.

    Primary check is the typed ``chromadb.errors.NotFoundError``, imported
    lazily so this module keeps importing cleanly without chromadb
    installed. The substring heuristic is only a fallback for when that
    import itself fails (chromadb absent, or its errors module moved) -
    when chromadb is installed and the typed exception is importable, it is
    the *only* check used, so an unrelated error (e.g. a plain ValueError
    that happens to mention "not found") can never be misclassified as a
    missing collection.
    """
    try:
        from chromadb.errors import NotFoundError
    except ImportError:
        text = str(exc).lower()
        return "does not exist" in text or "not found" in text
    return isinstance(exc, NotFoundError)


class ChromaAdapter(VectorAdapter):
    name = "chroma"

    def __init__(self, path: str):
        if not path:
            raise ValueError("chroma adapter requires a path")
        self.path = str(path)
        self._client = None

    def capabilities(self) -> Capability:
        return Capability(
            name=self.name,
            exposes_dense_scores=True,
            exposes_sparse=False,
            supports_rebuild=False,
        )

    def _connect(self):
        if self._client is None:
            try:
                import chromadb
            except ImportError as exc:
                raise ImportError(
                    "chromadb is not installed; run: pip install 'recallops[chroma]'"
                ) from exc
            self._client = chromadb.PersistentClient(path=self.path)
        return self._client

    def _collection(self, collection: str):
        return self._connect().get_or_create_collection(
            name=collection, metadata=dict(_COLLECTION_METADATA)
        )

    def ensure_collection(self, collection: str, dims: int) -> None:
        self._collection(collection)

    def upsert(self, collection: str, ids: list[str], vectors: np.ndarray,
               payloads: list[dict]) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2-D, got shape {vectors.shape}")
        if len(ids) != vectors.shape[0] or len(ids) != len(payloads):
            raise ValueError(
                f"length mismatch: {len(ids)} ids, {vectors.shape[0]} vectors, "
                f"{len(payloads)} payloads"
            )
        col = self._collection(collection)
        col.upsert(
            ids=list(ids),
            embeddings=vectors.tolist(),
            metadatas=[payload_metadata(p) for p in payloads],
        )

    def query_dense(self, collection: str, vector: np.ndarray,
                    top_k: int) -> list[tuple[str, float]]:
        if top_k <= 0:
            return []
        client = self._connect()
        try:
            col = client.get_collection(name=collection)
        except Exception as exc:
            if _is_missing_collection_error(exc):
                raise KeyError(collection) from exc
            raise
        total = int(col.count())
        if total == 0:
            return []
        query = np.asarray(vector, dtype=np.float32).reshape(-1).tolist()
        res = col.query(query_embeddings=[query], n_results=min(int(top_k), total),
                        include=["distances"])
        return [
            (str(cid), distance_to_score(dist))
            for cid, dist in zip(res["ids"][0], res["distances"][0])
        ]

    def count(self, collection: str) -> int:
        client = self._connect()
        try:
            col = client.get_collection(name=collection)
        except Exception as exc:
            if _is_missing_collection_error(exc):
                return 0
            raise
        return int(col.count())

    def drop(self, collection: str) -> None:
        client = self._connect()
        try:
            client.delete_collection(name=collection)
        except Exception as exc:
            if not _is_missing_collection_error(exc):
                raise
