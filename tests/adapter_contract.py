"""Shared behavioral contract for VectorAdapter implementations.

Subclass ``AdapterContract`` in a ``test_*.py`` file and override the
``adapter`` fixture. Every adapter must pass the same observable behaviors
so score semantics (cosine similarity, higher is better) never drift
between backends. The parity test is the load-bearing one: it compares each
adapter's ranking and scores against the exact built-in local adapter.
"""
from __future__ import annotations

import numpy as np
import pytest

DIMS = 16


def _unit(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    return vec / np.linalg.norm(vec)


class AdapterContract:
    @pytest.fixture()
    def adapter(self):
        raise NotImplementedError("subclasses provide the adapter fixture")

    @pytest.fixture()
    def collection(self, adapter):
        name = "recall_contract"
        adapter.drop(name)
        yield name
        adapter.drop(name)

    @pytest.fixture()
    def seeded(self, adapter, collection):
        rng = np.random.default_rng(7)
        ids = [f"ch_{i:02d}" for i in range(8)]
        vectors = rng.normal(size=(8, DIMS)).astype(np.float32)
        payloads = [
            {"doc_id": f"doc_{i}", "source_path": f"d/{i}.md", "ordinal": i}
            for i in range(8)
        ]
        adapter.ensure_collection(collection, DIMS)
        adapter.upsert(collection, ids, vectors, payloads)
        return ids, vectors

    def test_ensure_collection_is_idempotent(self, adapter, collection):
        adapter.ensure_collection(collection, DIMS)
        adapter.ensure_collection(collection, DIMS)
        assert adapter.count(collection) == 0

    def test_drop_missing_collection_is_noop(self, adapter):
        adapter.drop("recall_contract_never_created")

    def test_count_missing_collection_is_zero(self, adapter):
        assert adapter.count("recall_contract_never_created") == 0

    def test_upsert_then_count(self, adapter, collection, seeded):
        ids, _ = seeded
        assert adapter.count(collection) == len(ids)

    def test_self_query_ranks_first_with_unit_score(self, adapter, collection, seeded):
        ids, vectors = seeded
        result = adapter.query_dense(collection, vectors[3], top_k=3)
        assert result[0][0] == ids[3]
        assert result[0][1] == pytest.approx(1.0, abs=1e-3)

    def test_scores_are_descending(self, adapter, collection, seeded):
        _, vectors = seeded
        scores = [s for _, s in adapter.query_dense(collection, vectors[0], top_k=8)]
        assert scores == sorted(scores, reverse=True)

    def test_scores_are_cosine_similarity(self, adapter, collection, seeded):
        ids, vectors = seeded
        query = vectors[5]
        result = dict(adapter.query_dense(collection, query, top_k=8))
        qn = _unit(query)
        checked = 0
        for i, cid in enumerate(ids):
            if cid in result:
                expected = float(np.dot(_unit(vectors[i]), qn))
                assert result[cid] == pytest.approx(expected, abs=1e-3)
                checked += 1
        assert checked >= 4

    def test_reupsert_overwrites(self, adapter, collection, seeded):
        ids, vectors = seeded
        replacement = -vectors[0:1]
        adapter.upsert(
            collection, [ids[0]], replacement,
            [{"doc_id": "doc_0", "source_path": "d/0.md", "ordinal": 0}],
        )
        assert adapter.count(collection) == len(ids)
        result = adapter.query_dense(collection, replacement[0], top_k=1)
        assert result[0][0] == ids[0]
        assert result[0][1] == pytest.approx(1.0, abs=1e-3)

    def test_top_k_nonpositive_returns_empty(self, adapter, collection, seeded):
        _, vectors = seeded
        assert adapter.query_dense(collection, vectors[0], top_k=0) == []
        assert adapter.query_dense(collection, vectors[0], top_k=-3) == []

    def test_top_k_beyond_count_returns_all(self, adapter, collection, seeded):
        ids, vectors = seeded
        result = adapter.query_dense(collection, vectors[0], top_k=50)
        assert len(result) == len(ids)
        assert {cid for cid, _ in result} == set(ids)

    def test_upsert_length_mismatch_raises(self, adapter, collection):
        adapter.ensure_collection(collection, DIMS)
        vectors = np.zeros((2, DIMS), dtype=np.float32)
        with pytest.raises(ValueError):
            adapter.upsert(collection, ["ch_00"], vectors, [{}, {}])

    def test_drop_resets_count(self, adapter, collection, seeded):
        adapter.drop(collection)
        assert adapter.count(collection) == 0

    def test_parity_with_exact_local_reference(self, adapter, collection, tmp_path):
        from recallops.adapters.local import LocalIndexAdapter

        rng = np.random.default_rng(42)
        ids = [f"ch_{i:03d}" for i in range(20)]
        vectors = rng.normal(size=(20, DIMS)).astype(np.float32)
        payloads = [
            {"doc_id": f"doc_{i}", "source_path": f"d/{i}.md", "ordinal": i}
            for i in range(20)
        ]
        adapter.ensure_collection(collection, DIMS)
        adapter.upsert(collection, ids, vectors, payloads)
        reference = LocalIndexAdapter(tmp_path / "parity_reference")
        reference.ensure_collection(collection, DIMS)
        reference.upsert(collection, ids, vectors, payloads)
        for qi in (0, 7, 13):
            got = adapter.query_dense(collection, vectors[qi], top_k=5)
            want = reference.query_dense(collection, vectors[qi], top_k=5)
            assert [cid for cid, _ in got] == [cid for cid, _ in want]
            for (_, gs), (_, ws) in zip(got, want):
                assert gs == pytest.approx(ws, abs=1e-3)
