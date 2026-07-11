"""LanceDB adapter (PRD FR-12), embedded columnar vector store.

No ANN index is created in v0: LanceDB scans flat (exact) when a table has
no vector index, so results are exact cosine and there is nothing to
rebuild (``supports_rebuild`` False). ``_distance`` is cosine distance;
``query_dense`` converts to cosine similarity, higher is better, matching
the built-in local adapter. Payloads are carried as one JSON column.
``lancedb`` is an optional dependency (extra ``[lancedb]``), imported
lazily so this module always imports cleanly.

``query_dense`` raises ``KeyError`` when the table does not exist yet
(matching ``LocalIndexAdapter``) rather than returning an empty result,
so querying before ingest fails loudly instead of looking like a corpus
with no matches.

Table membership is checked via ``Connection.list_tables()`` rather than
the older ``table_names()`` - the installed lancedb (0.34) emits a
``DeprecationWarning`` from ``table_names()``, and ``list_tables()`` is the
documented replacement (its response wraps the names in a
``.tables`` list, paginated via ``page_token``; a single project's
collection count never approaches the default page size).
"""
from __future__ import annotations

import json

import numpy as np
import pyarrow as pa

from .base import Capability, VectorAdapter


def table_schema(dims: int) -> pa.Schema:
    return pa.schema([
        pa.field("chunk_id", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), int(dims))),
        pa.field("payload_json", pa.string()),
    ])


def distance_to_score(distance: float) -> float:
    """Cosine distance -> cosine similarity."""
    return 1.0 - float(distance)


def rows_for_upsert(ids: list[str], vectors: np.ndarray,
                    payloads: list[dict]) -> list[dict]:
    return [
        {
            "chunk_id": cid,
            "vector": [float(x) for x in vec],
            "payload_json": json.dumps(payload, sort_keys=True),
        }
        for cid, vec, payload in zip(ids, vectors, payloads)
    ]


class LanceDBAdapter(VectorAdapter):
    name = "lancedb"

    def __init__(self, path: str):
        if not path:
            raise ValueError("lancedb adapter requires a path")
        self.path = str(path)
        self._db = None

    def capabilities(self) -> Capability:
        return Capability(
            name=self.name,
            exposes_dense_scores=True,
            exposes_sparse=False,
            supports_rebuild=False,
        )

    def _connect(self):
        if self._db is None:
            try:
                import lancedb
            except ImportError as exc:
                raise ImportError(
                    "lancedb is not installed; run: pip install 'recallops[lancedb]'"
                ) from exc
            self._db = lancedb.connect(self.path)
        return self._db

    def _table_names(self) -> set[str]:
        return set(self._connect().list_tables().tables)

    def ensure_collection(self, collection: str, dims: int) -> None:
        db = self._connect()
        if collection in self._table_names():
            return
        db.create_table(collection, schema=table_schema(dims))

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
        db = self._connect()
        if collection not in self._table_names():
            self.ensure_collection(collection, int(vectors.shape[1]))
        table = db.open_table(collection)
        (table.merge_insert("chunk_id")
             .when_matched_update_all()
             .when_not_matched_insert_all()
             .execute(rows_for_upsert(ids, vectors, payloads)))

    def query_dense(self, collection: str, vector: np.ndarray,
                    top_k: int) -> list[tuple[str, float]]:
        if top_k <= 0:
            return []
        if collection not in self._table_names():
            raise KeyError(collection)
        table = self._connect().open_table(collection)
        query = [float(x) for x in np.asarray(vector, dtype=np.float32).reshape(-1)]
        rows = (table.search(query)
                     .distance_type("cosine")
                     .limit(int(top_k))
                     .to_list())
        return [(str(r["chunk_id"]), distance_to_score(r["_distance"])) for r in rows]

    def count(self, collection: str) -> int:
        if collection not in self._table_names():
            return 0
        return int(self._connect().open_table(collection).count_rows())

    def drop(self, collection: str) -> None:
        self._connect().drop_table(collection, ignore_missing=True)
