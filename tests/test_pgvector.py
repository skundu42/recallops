from __future__ import annotations

import importlib
import os
import sys

import numpy as np
import pytest

from recallops.adapters import pgvector as pg
from recallops.adapters.base import VectorAdapter

DSN = os.environ.get("RECALL_PG_DSN")


# -- always-run unit tests (no connection, psycopg not required) --------------


def test_module_imports_cleanly_without_psycopg() -> None:
    assert not hasattr(pg, "psycopg")


def test_pgvector_adapter_is_a_vector_adapter() -> None:
    assert issubclass(pg.PgVectorAdapter, VectorAdapter)


def test_capabilities_without_connection() -> None:
    adapter = pg.PgVectorAdapter("postgresql://unused")
    caps = adapter.capabilities()
    assert adapter.name == "pgvector"
    assert caps.name == "pgvector"
    assert caps.exposes_dense_scores is True
    assert caps.exposes_sparse is False
    assert caps.supports_rebuild is True


def test_default_schema_is_recallops() -> None:
    adapter = pg.PgVectorAdapter("postgresql://unused")
    assert adapter.schema == "recallops"


def test_probes_defaults_to_one_and_is_configurable() -> None:
    assert pg.PgVectorAdapter("postgresql://unused").probes == 1
    assert pg.PgVectorAdapter("postgresql://unused", probes=25).probes == 25


def test_construct_does_not_connect() -> None:
    adapter = pg.PgVectorAdapter("postgresql://unused")
    assert adapter._conn is None


def test_create_extension_sql() -> None:
    sql = pg.create_extension_sql()
    assert sql == "CREATE EXTENSION IF NOT EXISTS vector"


def test_create_table_sql_has_ddl() -> None:
    sql = pg.create_table_sql("col_abc", 256)
    assert '"recallops"."col_abc"' in sql
    assert "CREATE TABLE IF NOT EXISTS" in sql
    assert "chunk_id text PRIMARY KEY" in sql
    assert "embedding vector(256)" in sql
    assert "payload jsonb" in sql


def test_create_table_sql_dims_are_interpolated() -> None:
    assert "vector(1536)" in pg.create_table_sql("col_x", 1536)
    assert "vector(8)" in pg.create_table_sql("col_x", 8)


def test_create_table_sql_custom_schema() -> None:
    sql = pg.create_table_sql("col_x", 64, schema="myschema")
    assert '"myschema"."col_x"' in sql


def test_create_index_sql_is_ivfflat_cosine() -> None:
    sql = pg.create_index_sql("col_abc")
    assert "CREATE INDEX IF NOT EXISTS" in sql
    assert "USING ivfflat" in sql
    assert "vector_cosine_ops" in sql
    assert '"recallops"."col_abc"' in sql


def test_upsert_sql_uses_on_conflict() -> None:
    sql = pg.upsert_sql("col_abc")
    assert "INSERT INTO" in sql
    assert '"recallops"."col_abc"' in sql
    assert "(chunk_id, embedding, payload)" in sql
    assert "%s::vector" in sql
    assert "%s::jsonb" in sql
    assert "ON CONFLICT (chunk_id) DO UPDATE" in sql
    assert "embedding = EXCLUDED.embedding" in sql
    assert "payload = EXCLUDED.payload" in sql


def test_query_sql_uses_cosine_operator() -> None:
    sql = pg.query_sql("col_abc")
    assert "<=>" in sql
    assert "1 - (embedding <=> %s::vector)" in sql
    assert '"recallops"."col_abc"' in sql
    assert "ORDER BY embedding <=> %s::vector" in sql
    assert "LIMIT %s" in sql


def test_count_and_drop_sql() -> None:
    assert pg.count_sql("col_abc") == 'SELECT count(*) FROM "recallops"."col_abc"'
    assert pg.drop_table_sql("col_abc") == 'DROP TABLE IF EXISTS "recallops"."col_abc"'


def test_reindex_sql_targets_the_ivfflat_index() -> None:
    sql = pg.reindex_sql("col_abc")
    assert "REINDEX INDEX" in sql
    assert '"recallops"."col_abc_embedding_ivfflat"' in sql


def test_identifier_rejects_injection() -> None:
    with pytest.raises(ValueError):
        pg.create_table_sql("col_x; DROP TABLE users", 8)
    with pytest.raises(ValueError):
        pg.qualified_table("col_x", schema="bad schema")
    with pytest.raises(ValueError):
        pg.qualified_table('col"x')


def test_vector_literal_roundtrips() -> None:
    assert pg.vector_literal([1.0, 2.5, -3.0]) == "[1.0,2.5,-3.0]"
    assert pg.vector_literal(np.array([0.0, 1.0], dtype=np.float32)) == "[0.0,1.0]"


def test_module_reloads_with_psycopg_absent() -> None:
    saved = sys.modules.get("psycopg")
    sys.modules["psycopg"] = None  # forces ``import psycopg`` to raise
    try:
        reloaded = importlib.reload(pg)
        assert "vector(8)" in reloaded.create_table_sql("col_x", 8)
        adapter = reloaded.PgVectorAdapter("postgresql://unused")
        assert adapter.capabilities().name == "pgvector"
        with pytest.raises(Exception):
            adapter.count("col_x")
    finally:
        if saved is not None:
            sys.modules["psycopg"] = saved
        else:
            sys.modules.pop("psycopg", None)
        importlib.reload(pg)


# -- real-server contract test (opt-in via RECALL_PG_DSN) ---------------------


@pytest.fixture()
def live_adapter():
    pytest.importorskip("psycopg")
    if not DSN:
        pytest.skip("RECALL_PG_DSN not set; skipping real pgvector contract test")
    adapter = pg.PgVectorAdapter(DSN)
    collection = "recall_ct_" + "abc123"
    adapter.drop(collection)
    try:
        yield adapter, collection
    finally:
        adapter.drop(collection)
        adapter.close()


def test_contract_roundtrip(live_adapter) -> None:
    adapter, collection = live_adapter
    dims = 8
    rng = np.random.default_rng(0)
    ids = [f"ch_{i:02d}" for i in range(6)]
    vectors = rng.normal(size=(6, dims)).astype(np.float32)
    payloads = [{"doc_id": f"doc_{i}", "ordinal": i} for i in range(6)]

    adapter.ensure_collection(collection, dims)
    adapter.ensure_collection(collection, dims)  # idempotent
    adapter.upsert(collection, ids, vectors, payloads)
    assert adapter.count(collection) == len(ids)

    result = adapter.query_dense(collection, vectors[2], top_k=3)
    assert result[0][0] == ids[2]
    assert result[0][1] == pytest.approx(1.0, abs=1e-4)
    assert all(isinstance(c, str) and isinstance(s, float) for c, s in result)

    adapter.rebuild(collection, seed=1)
    assert adapter.count(collection) == len(ids)

    adapter.drop(collection)
    assert adapter.count(collection) == 0
