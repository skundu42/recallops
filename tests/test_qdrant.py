from __future__ import annotations

import pytest
from adapter_contract import AdapterContract

from recallops.adapters import qdrant as qd
from recallops.adapters.base import VectorAdapter

# -- always-run unit tests (qdrant-client not required) -----------------------


def test_module_imports_cleanly_without_client() -> None:
    assert not hasattr(qd, "QdrantClient")


def test_is_a_vector_adapter() -> None:
    assert issubclass(qd.QdrantAdapter, VectorAdapter)


def test_capabilities_without_connection() -> None:
    adapter = qd.QdrantAdapter(path="/tmp/unused")
    caps = adapter.capabilities()
    assert caps.name == "qdrant"
    assert caps.exposes_dense_scores is True
    assert caps.exposes_sparse is False
    assert caps.supports_rebuild is False


def test_requires_exactly_one_of_url_or_path() -> None:
    with pytest.raises(ValueError):
        qd.QdrantAdapter()
    with pytest.raises(ValueError):
        qd.QdrantAdapter(url="http://x", path="/tmp/y")


def test_construct_does_not_connect() -> None:
    assert qd.QdrantAdapter(path="/tmp/unused")._client is None


def test_point_id_is_deterministic_uuid() -> None:
    a = qd.point_id_for_chunk("ch_abc")
    assert a == qd.point_id_for_chunk("ch_abc")
    assert a != qd.point_id_for_chunk("ch_abd")
    import uuid

    uuid.UUID(a)  # parses


def test_chunk_id_rides_in_payload() -> None:
    p = qd.payload_with_chunk_id({"doc_id": "d"}, "ch_abc")
    assert qd.chunk_id_from_payload(p) == "ch_abc"
    assert p["doc_id"] == "d"
    with pytest.raises(KeyError):
        qd.chunk_id_from_payload({"doc_id": "d"})
    with pytest.raises(KeyError):
        qd.chunk_id_from_payload(None)


# -- behavioral contract (embedded local mode; needs qdrant-client) ----------


class TestQdrantContract(AdapterContract):
    @pytest.fixture()
    def adapter(self, tmp_path):
        pytest.importorskip("qdrant_client")
        a = qd.QdrantAdapter(path=str(tmp_path / "qdrant"))
        yield a
        a.close()
