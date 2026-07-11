from __future__ import annotations

import json

import numpy as np
import pytest
from adapter_contract import AdapterContract

from recallops.adapters import chroma as ch
from recallops.adapters.base import VectorAdapter

# -- always-run unit tests (chromadb not required) ----------------------------


def test_module_imports_cleanly_without_client() -> None:
    assert not hasattr(ch, "chromadb")


def test_is_a_vector_adapter() -> None:
    assert issubclass(ch.ChromaAdapter, VectorAdapter)


def test_capabilities_without_connection() -> None:
    caps = ch.ChromaAdapter("/tmp/unused").capabilities()
    assert caps.name == "chroma"
    assert caps.exposes_dense_scores is True
    assert caps.exposes_sparse is False
    assert caps.supports_rebuild is False


def test_requires_a_path() -> None:
    with pytest.raises(ValueError):
        ch.ChromaAdapter("")


def test_construct_does_not_connect() -> None:
    assert ch.ChromaAdapter("/tmp/unused")._client is None


def test_distance_to_score_inverts_cosine_distance() -> None:
    assert ch.distance_to_score(0.0) == pytest.approx(1.0)
    assert ch.distance_to_score(1.0) == pytest.approx(0.0)
    assert ch.distance_to_score(1.7) == pytest.approx(-0.7)


def test_payload_metadata_is_scalar_valued_json() -> None:
    meta = ch.payload_metadata({"doc_id": "d", "ordinal": 3})
    assert set(meta) == {"payload_json"}
    assert json.loads(meta["payload_json"]) == {"doc_id": "d", "ordinal": 3}


# -- missing-collection detection (typed check; needs chromadb) ---------------


def test_missing_collection_error_uses_typed_notfounderror() -> None:
    pytest.importorskip("chromadb")
    from chromadb.errors import NotFoundError

    assert ch._is_missing_collection_error(NotFoundError("Collection x does not exist")) is True


def test_missing_collection_error_rejects_unrelated_valueerror() -> None:
    # Regression: a ValueError whose wording happens to contain "not found"
    # must not be silently swallowed as a missing collection.
    pytest.importorskip("chromadb")
    assert ch._is_missing_collection_error(ValueError("embedding function not found")) is False


def test_missing_collection_error_rejects_unrelated_runtimeerror() -> None:
    pytest.importorskip("chromadb")
    assert ch._is_missing_collection_error(RuntimeError("file is not a database")) is False


# -- query_dense on a missing collection (needs chromadb) ---------------------


def test_query_dense_missing_collection_raises_keyerror(tmp_path) -> None:
    pytest.importorskip("chromadb")
    adapter = ch.ChromaAdapter(str(tmp_path / "chroma"))
    with pytest.raises(KeyError):
        adapter.query_dense("never_created", np.zeros(4, dtype=np.float32), top_k=3)
    # querying must not silently get-or-create the collection
    assert adapter._connect().list_collections() == []


# -- behavioral contract (embedded; needs chromadb) ---------------------------


class TestChromaContract(AdapterContract):
    @pytest.fixture()
    def adapter(self, tmp_path):
        pytest.importorskip("chromadb")
        return ch.ChromaAdapter(str(tmp_path / "chroma"))
