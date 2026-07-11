"""Qdrant adapter (PRD FR-12).

Runs against a server (``url`` + optional ``api_key``) or the client's
embedded local mode (``path``), which is what CI exercises: no server
required. Collections are created with cosine distance, so ``query_dense``
returns cosine similarity directly (higher is better), matching the
built-in local adapter.

Qdrant point ids must be UUIDs or unsigned ints while chunk ids are strings
(``ch_...``), so ``point_id_for_chunk`` derives a deterministic UUID5 per
chunk id and the chunk id itself rides in the payload. ``qdrant-client`` is
an optional dependency (extra ``[qdrant]``), imported lazily so this module
always imports cleanly.

``supports_rebuild`` is False: local mode is exact (nothing to rebuild) and
server-side HNSW has no cheap deterministic rebuild; calibration falls back
to query re-runs (gating.py) and funnel shadow scoring quarantines ANN
divergence, so nothing is silently lost.
"""
from __future__ import annotations

import uuid

import numpy as np

from .base import Capability, VectorAdapter

_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "recallops.qdrant")
_CHUNK_ID_KEY = "recallops_chunk_id"


def point_id_for_chunk(chunk_id: str) -> str:
    """Deterministic Qdrant point id (UUID5) for a chunk id."""
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


def payload_with_chunk_id(payload: dict, chunk_id: str) -> dict:
    merged = dict(payload)
    merged[_CHUNK_ID_KEY] = chunk_id
    return merged


def chunk_id_from_payload(payload: dict | None) -> str:
    if not payload or _CHUNK_ID_KEY not in payload:
        raise KeyError(f"qdrant point payload missing {_CHUNK_ID_KEY!r}")
    return str(payload[_CHUNK_ID_KEY])


class QdrantAdapter(VectorAdapter):
    name = "qdrant"

    def __init__(self, url: str | None = None, path: str | None = None,
                 api_key: str | None = None):
        if bool(url) == bool(path):
            raise ValueError(
                "qdrant adapter needs exactly one of url= (server) or path= (embedded local mode)"
            )
        self.url = url
        self.path = path
        self.api_key = api_key
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
                from qdrant_client import QdrantClient
            except ImportError as exc:
                raise ImportError(
                    "qdrant-client is not installed; run: pip install 'recallops[qdrant]'"
                ) from exc
            if self.url:
                self._client = QdrantClient(url=self.url, api_key=self.api_key)
            else:
                self._client = QdrantClient(path=self.path)
        return self._client

    def ensure_collection(self, collection: str, dims: int) -> None:
        client = self._connect()
        from qdrant_client import models

        if client.collection_exists(collection):
            return
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(size=int(dims), distance=models.Distance.COSINE),
        )

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
        client = self._connect()
        from qdrant_client import models

        points = [
            models.PointStruct(
                id=point_id_for_chunk(cid),
                vector=[float(x) for x in vec],
                payload=payload_with_chunk_id(payload, cid),
            )
            for cid, vec, payload in zip(ids, vectors, payloads)
        ]
        client.upsert(collection_name=collection, points=points, wait=True)

    def query_dense(self, collection: str, vector: np.ndarray,
                    top_k: int) -> list[tuple[str, float]]:
        if top_k <= 0:
            return []
        client = self._connect()
        query = [float(x) for x in np.asarray(vector, dtype=np.float32).reshape(-1)]
        result = client.query_points(collection_name=collection, query=query,
                                     limit=int(top_k), with_payload=True)
        return [(chunk_id_from_payload(p.payload), float(p.score)) for p in result.points]

    def count(self, collection: str) -> int:
        client = self._connect()
        if not client.collection_exists(collection):
            return 0
        return int(client.count(collection_name=collection, exact=True).count)

    def drop(self, collection: str) -> None:
        client = self._connect()
        if client.collection_exists(collection):
            client.delete_collection(collection_name=collection)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
