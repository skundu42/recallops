"""pgvector adapter (PRD FR-12), first adapter, richest score visibility.

One table per collection under ``schema`` (default ``recallops``):
``(chunk_id text primary key, embedding vector(dims), payload jsonb)``.
Dense retrieval uses pgvector's cosine distance operator ``<=>`` and returns
``1 - distance`` so scores match the built-in local adapter's cosine similarity.

``psycopg`` is an optional dependency (extra ``[pg]``) and is imported lazily
inside the methods that actually talk to the database, the module imports
cleanly and every SQL string is built by connection-free module functions
(``create_table_sql`` / ``upsert_sql`` / ``query_sql`` / ...), so they can be
unit-tested without a server or the driver installed.
"""
from __future__ import annotations

import json
import re

import numpy as np

from .base import Capability, VectorAdapter

SCHEMA_DEFAULT = "recallops"
DEFAULT_LISTS = 100

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _ident(name: str) -> str:
    if not _IDENT.match(name):
        raise ValueError(f"unsafe SQL identifier: {name!r}")
    return f'"{name}"'


def qualified_table(collection: str, schema: str = SCHEMA_DEFAULT) -> str:
    return f"{_ident(schema)}.{_ident(collection)}"


def _index_name(collection: str) -> str:
    return f"{collection}_embedding_ivfflat"


def create_extension_sql() -> str:
    return "CREATE EXTENSION IF NOT EXISTS vector"


def create_schema_sql(schema: str = SCHEMA_DEFAULT) -> str:
    return f"CREATE SCHEMA IF NOT EXISTS {_ident(schema)}"


def create_table_sql(collection: str, dims: int, schema: str = SCHEMA_DEFAULT) -> str:
    return (
        f"CREATE TABLE IF NOT EXISTS {qualified_table(collection, schema)} "
        f"(chunk_id text PRIMARY KEY, embedding vector({int(dims)}), payload jsonb)"
    )


def create_index_sql(collection: str, schema: str = SCHEMA_DEFAULT,
                     lists: int = DEFAULT_LISTS) -> str:
    return (
        f"CREATE INDEX IF NOT EXISTS {_ident(_index_name(collection))} "
        f"ON {qualified_table(collection, schema)} "
        f"USING ivfflat (embedding vector_cosine_ops) WITH (lists = {int(lists)})"
    )


def upsert_sql(collection: str, schema: str = SCHEMA_DEFAULT) -> str:
    return (
        f"INSERT INTO {qualified_table(collection, schema)} "
        f"(chunk_id, embedding, payload) VALUES (%s, %s::vector, %s::jsonb) "
        f"ON CONFLICT (chunk_id) DO UPDATE SET "
        f"embedding = EXCLUDED.embedding, payload = EXCLUDED.payload"
    )


def query_sql(collection: str, schema: str = SCHEMA_DEFAULT) -> str:
    return (
        f"SELECT chunk_id, 1 - (embedding <=> %s::vector) AS score "
        f"FROM {qualified_table(collection, schema)} "
        f"ORDER BY embedding <=> %s::vector LIMIT %s"
    )


def count_sql(collection: str, schema: str = SCHEMA_DEFAULT) -> str:
    return f"SELECT count(*) FROM {qualified_table(collection, schema)}"


def drop_table_sql(collection: str, schema: str = SCHEMA_DEFAULT) -> str:
    return f"DROP TABLE IF EXISTS {qualified_table(collection, schema)}"


def reindex_sql(collection: str, schema: str = SCHEMA_DEFAULT) -> str:
    return f"REINDEX INDEX {_ident(schema)}.{_ident(_index_name(collection))}"


def vector_literal(vec) -> str:
    return "[" + ",".join(repr(float(x)) for x in np.asarray(vec).reshape(-1)) + "]"


class PgVectorAdapter(VectorAdapter):
    name = "pgvector"

    def __init__(self, dsn: str, schema: str = SCHEMA_DEFAULT, probes: int = 1):
        # ``probes`` is the ivfflat recall/latency knob: probes=1 (pgvector's
        # default) searches a single list and can miss the true top-k badly at
        # scale (measured: recall@1 0.98 exact -> ~0.2 live on ~4k vectors);
        # probes toward ``lists`` approaches exact. Phase-0 validation showed the
        # default is not gate-trustworthy, so it is tunable here.
        self.dsn = dsn
        self.schema = schema
        self.probes = int(probes)
        self._conn = None

    def capabilities(self) -> Capability:
        return Capability(
            name=self.name,
            exposes_dense_scores=True,
            exposes_sparse=False,
            supports_rebuild=True,
        )

    def _connection(self):
        if self._conn is None:
            import psycopg

            self._conn = psycopg.connect(self.dsn)
        return self._conn

    def _table_exists(self, cur, collection: str) -> bool:
        cur.execute("SELECT to_regclass(%s)", (f"{self.schema}.{collection}",))
        return cur.fetchone()[0] is not None

    def ensure_collection(self, collection: str, dims: int) -> None:
        conn = self._connection()
        with conn.cursor() as cur:
            cur.execute(create_extension_sql())
            cur.execute(create_schema_sql(self.schema))
            cur.execute(create_table_sql(collection, dims, self.schema))
            cur.execute(create_index_sql(collection, self.schema))
        conn.commit()

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
        rows = [
            (cid, vector_literal(vec), json.dumps(payload))
            for cid, vec, payload in zip(ids, vectors, payloads)
        ]
        conn = self._connection()
        with conn.cursor() as cur:
            cur.executemany(upsert_sql(collection, self.schema), rows)
        conn.commit()

    def query_dense(self, collection: str, vector: np.ndarray,
                    top_k: int) -> list[tuple[str, float]]:
        if top_k <= 0:
            return []
        literal = vector_literal(np.asarray(vector, dtype=np.float32).reshape(-1))
        conn = self._connection()
        with conn.cursor() as cur:
            cur.execute(f"SET LOCAL ivfflat.probes = {int(self.probes)}")
            cur.execute(query_sql(collection, self.schema), (literal, literal, int(top_k)))
            rows = cur.fetchall()
        return [(str(cid), float(score)) for cid, score in rows]

    def count(self, collection: str) -> int:
        conn = self._connection()
        with conn.cursor() as cur:
            if not self._table_exists(cur, collection):
                return 0
            cur.execute(count_sql(collection, self.schema))
            (n,) = cur.fetchone()
        return int(n)

    def drop(self, collection: str) -> None:
        conn = self._connection()
        with conn.cursor() as cur:
            cur.execute(drop_table_sql(collection, self.schema))
        conn.commit()

    def rebuild(self, collection: str, seed: int = 0) -> None:
        conn = self._connection()
        with conn.cursor() as cur:
            cur.execute(reindex_sql(collection, self.schema))
        conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
