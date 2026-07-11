from __future__ import annotations

import json

import numpy as np
import pytest
from adapter_contract import AdapterContract

from recallops.adapters import lancedb as ldb
from recallops.adapters.base import VectorAdapter

# -- always-run unit tests (lancedb not required) ------------------------------


def test_module_imports_cleanly_without_client() -> None:
    assert not hasattr(ldb, "lancedb")


def test_is_a_vector_adapter() -> None:
    assert issubclass(ldb.LanceDBAdapter, VectorAdapter)


def test_capabilities_without_connection() -> None:
    caps = ldb.LanceDBAdapter("/tmp/unused").capabilities()
    assert caps.name == "lancedb"
    assert caps.exposes_dense_scores is True
    assert caps.exposes_sparse is False
    assert caps.supports_rebuild is False


def test_requires_a_path() -> None:
    with pytest.raises(ValueError):
        ldb.LanceDBAdapter("")


def test_construct_does_not_connect() -> None:
    assert ldb.LanceDBAdapter("/tmp/unused")._db is None


def test_table_schema_pins_vector_width() -> None:
    schema = ldb.table_schema(16)
    field = schema.field("vector")
    assert field.type.list_size == 16
    assert schema.field("chunk_id").type == "string"
    assert schema.field("payload_json").type == "string"


def test_distance_to_score_inverts_cosine_distance() -> None:
    assert ldb.distance_to_score(0.0) == pytest.approx(1.0)
    assert ldb.distance_to_score(1.0) == pytest.approx(0.0)


def test_rows_for_upsert_shape() -> None:
    rows = ldb.rows_for_upsert(
        ["ch_a"], np.asarray([[1.0, 2.0]], dtype=np.float32), [{"doc_id": "d"}]
    )
    assert rows == [{"chunk_id": "ch_a", "vector": [1.0, 2.0],
                     "payload_json": json.dumps({"doc_id": "d"}, sort_keys=True)}]


# -- query_dense on a missing collection (needs lancedb) -----------------------


def test_query_dense_missing_collection_raises_keyerror(tmp_path) -> None:
    pytest.importorskip("lancedb")
    adapter = ldb.LanceDBAdapter(str(tmp_path / "lancedb"))
    with pytest.raises(KeyError):
        adapter.query_dense("never_created", np.zeros(4, dtype=np.float32), top_k=3)


# -- behavioral contract (embedded; needs lancedb) -----------------------------


class TestLanceDBContract(AdapterContract):
    @pytest.fixture()
    def adapter(self, tmp_path):
        pytest.importorskip("lancedb")
        return ldb.LanceDBAdapter(str(tmp_path / "lancedb"))
