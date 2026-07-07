from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from recallops.adapters.local import LocalIndexAdapter

COL = "snap_test"


def _toy(seed: int = 0, n: int = 6, dims: int = 8):
    rng = np.random.default_rng(seed)
    vectors = (rng.normal(size=(n, dims)) * 3.0).astype(np.float32)
    ids = [f"ch_{i:02d}" for i in range(n)]
    payloads = [{"ordinal": i} for i in range(n)]
    return ids, vectors, payloads


def _numpy_cosine_ranking(ids: list[str], vectors: np.ndarray, query: np.ndarray):
    mat = vectors.astype(np.float64)
    mat = mat / np.linalg.norm(mat, axis=1, keepdims=True)
    q = query.astype(np.float64)
    q = q / np.linalg.norm(q)
    scores = mat @ q
    order = sorted(range(len(ids)), key=lambda i: (-scores[i], ids[i]))
    return [(ids[i], float(scores[i])) for i in order]


def _filled(root: Path, **kwargs) -> LocalIndexAdapter:
    adapter = LocalIndexAdapter(root, **kwargs)
    ids, vectors, payloads = _toy()
    adapter.upsert(COL, ids, vectors, payloads)
    return adapter


def test_capabilities(tmp_path: Path) -> None:
    adapter = LocalIndexAdapter(tmp_path)
    caps = adapter.capabilities()
    assert adapter.name == "local"
    assert caps.name == "local"
    assert caps.exposes_dense_scores is True
    assert caps.exposes_sparse is False
    assert caps.supports_rebuild is True


def test_upsert_query_roundtrip(tmp_path: Path) -> None:
    adapter = _filled(tmp_path)
    ids, vectors, _ = _toy()
    assert adapter.count(COL) == len(ids)
    result = adapter.query_dense(COL, vectors[2], top_k=3)
    assert len(result) == 3
    assert result[0][0] == ids[2]
    assert result[0][1] == pytest.approx(1.0, abs=1e-5)
    assert all(isinstance(c, str) and isinstance(s, float) for c, s in result)


def test_exact_cosine_ranking_matches_numpy(tmp_path: Path) -> None:
    adapter = _filled(tmp_path)
    ids, vectors, _ = _toy()
    rng = np.random.default_rng(42)
    for _ in range(5):
        query = (rng.normal(size=vectors.shape[1]) * 2.0).astype(np.float32)
        expected = _numpy_cosine_ranking(ids, vectors, query)
        got = adapter.query_dense(COL, query, top_k=len(ids))
        assert [c for c, _ in got] == [c for c, _ in expected]
        for (_, gs), (_, es) in zip(got, expected):
            assert gs == pytest.approx(es, abs=1e-5)


def test_vectors_persisted_as_normalized_fp32_npz(tmp_path: Path) -> None:
    _filled(tmp_path)
    path = tmp_path / "collections" / f"{COL}.npz"
    assert path.exists()
    with np.load(path, allow_pickle=False) as data:
        matrix = data["vectors"]
        stored_ids = [str(x) for x in data["ids"]]
    assert matrix.dtype == np.float32
    assert stored_ids == _toy()[0]
    norms = np.linalg.norm(matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_top_k_truncation_and_overflow(tmp_path: Path) -> None:
    adapter = _filled(tmp_path)
    ids, vectors, _ = _toy()
    query = vectors[0]
    assert len(adapter.query_dense(COL, query, top_k=2)) == 2
    assert len(adapter.query_dense(COL, query, top_k=100)) == len(ids)
    assert adapter.query_dense(COL, query, top_k=0) == []


def test_deterministic_tiebreak_by_id(tmp_path: Path) -> None:
    adapter = LocalIndexAdapter(tmp_path)
    same = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    adapter.upsert(COL, ["ch_b", "ch_a"], same, [{}, {}])
    result = adapter.query_dense(COL, np.array([1.0, 0.0], dtype=np.float32), top_k=2)
    assert [c for c, _ in result] == ["ch_a", "ch_b"]


def test_reupsert_replaces_existing_ids(tmp_path: Path) -> None:
    adapter = _filled(tmp_path)
    ids, vectors, _ = _toy()
    replacement = np.zeros((1, vectors.shape[1]), dtype=np.float32)
    replacement[0, 0] = 1.0
    adapter.upsert(COL, [ids[0]], replacement, [{"ordinal": 99}])
    assert adapter.count(COL) == len(ids)
    query = np.zeros(vectors.shape[1], dtype=np.float32)
    query[0] = 1.0
    top = adapter.query_dense(COL, query, top_k=1)
    assert top[0][0] == ids[0]
    assert top[0][1] == pytest.approx(1.0, abs=1e-5)

    new_vec = np.zeros((1, vectors.shape[1]), dtype=np.float32)
    new_vec[0, 1] = 1.0
    adapter.upsert(COL, ["ch_new"], new_vec, [{}])
    assert adapter.count(COL) == len(ids) + 1


def test_duplicate_ids_within_one_upsert_last_wins(tmp_path: Path) -> None:
    adapter = LocalIndexAdapter(tmp_path)
    vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    adapter.upsert(COL, ["ch_x", "ch_x"], vecs, [{"v": 1}, {"v": 2}])
    assert adapter.count(COL) == 1
    result = adapter.query_dense(COL, np.array([0.0, 1.0], dtype=np.float32), top_k=1)
    assert result[0][1] == pytest.approx(1.0, abs=1e-5)


def test_persistence_across_adapter_instances(tmp_path: Path) -> None:
    first = _filled(tmp_path)
    ids, vectors, _ = _toy()
    query = vectors[3]
    expected = first.query_dense(COL, query, top_k=len(ids))

    reopened = LocalIndexAdapter(tmp_path)
    assert reopened.count(COL) == len(ids)
    assert reopened.query_dense(COL, query, top_k=len(ids)) == expected


def test_count_missing_collection_is_zero(tmp_path: Path) -> None:
    assert LocalIndexAdapter(tmp_path).count("nope") == 0


def test_query_missing_collection_raises(tmp_path: Path) -> None:
    adapter = LocalIndexAdapter(tmp_path)
    with pytest.raises(KeyError):
        adapter.query_dense("nope", np.ones(4, dtype=np.float32), top_k=1)


def test_drop_removes_collection_and_is_idempotent(tmp_path: Path) -> None:
    adapter = _filled(tmp_path)
    path = tmp_path / "collections" / f"{COL}.npz"
    assert path.exists()
    adapter.drop(COL)
    assert not path.exists()
    assert adapter.count(COL) == 0
    adapter.drop(COL)
    with pytest.raises(KeyError):
        adapter.query_dense(COL, np.ones(8, dtype=np.float32), top_k=1)


def test_ensure_collection_creates_empty(tmp_path: Path) -> None:
    adapter = LocalIndexAdapter(tmp_path)
    adapter.ensure_collection(COL, dims=4)
    assert adapter.count(COL) == 0
    assert adapter.query_dense(COL, np.ones(4, dtype=np.float32), top_k=5) == []
    assert (tmp_path / "collections" / f"{COL}.npz").exists()
    adapter.ensure_collection(COL, dims=4)
    with pytest.raises(ValueError):
        adapter.ensure_collection(COL, dims=8)


def test_empty_collection_reloads_with_dims(tmp_path: Path) -> None:
    LocalIndexAdapter(tmp_path).ensure_collection(COL, dims=4)
    reopened = LocalIndexAdapter(tmp_path)
    assert reopened.count(COL) == 0
    assert reopened.query_dense(COL, np.ones(4, dtype=np.float32), top_k=3) == []
    with pytest.raises(ValueError):
        reopened.ensure_collection(COL, dims=8)
    reopened.upsert(COL, ["ch_a"], np.ones((1, 4), dtype=np.float32), [{}])
    assert reopened.count(COL) == 1


def test_upsert_dims_mismatch_raises(tmp_path: Path) -> None:
    adapter = LocalIndexAdapter(tmp_path)
    adapter.ensure_collection(COL, dims=4)
    with pytest.raises(ValueError):
        adapter.upsert(COL, ["ch_a"], np.ones((1, 8), dtype=np.float32), [{}])


def test_upsert_length_mismatch_raises(tmp_path: Path) -> None:
    adapter = LocalIndexAdapter(tmp_path)
    with pytest.raises(ValueError):
        adapter.upsert(COL, ["ch_a", "ch_b"], np.ones((1, 4), dtype=np.float32), [{}, {}])
    with pytest.raises(ValueError):
        adapter.upsert(COL, ["ch_a"], np.ones((1, 4), dtype=np.float32), [{}, {}])


def test_ann_same_seed_reproducible(tmp_path: Path) -> None:
    _filled(tmp_path)
    ids, vectors, _ = _toy()
    queries = [vectors[0], vectors[4], vectors[0]]

    def run() -> list:
        adapter = LocalIndexAdapter(tmp_path, ann_mode=True, ann_sigma=0.05, seed=7)
        return [adapter.query_dense(COL, q, top_k=len(ids)) for q in queries]

    assert run() == run()


def test_ann_noise_perturbs_scores_but_stays_near_exact(tmp_path: Path) -> None:
    exact = _filled(tmp_path)
    ids, vectors, _ = _toy()
    query = vectors[1]
    exact_scores = dict(exact.query_dense(COL, query, top_k=len(ids)))

    noisy = LocalIndexAdapter(tmp_path, ann_mode=True, ann_sigma=0.01, seed=0)
    noisy_scores = dict(noisy.query_dense(COL, query, top_k=len(ids)))
    assert noisy_scores != exact_scores
    for cid, score in noisy_scores.items():
        assert abs(score - exact_scores[cid]) < 0.01 * 6


def test_ann_seeds_flip_near_ties_but_not_far_pairs(tmp_path: Path) -> None:
    ids = ["near_a", "near_b", "far"]
    vectors = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.015, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    LocalIndexAdapter(tmp_path).upsert(COL, ids, vectors, [{}, {}, {}])
    query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    exact_order = [c for c, _ in LocalIndexAdapter(tmp_path).query_dense(COL, query, top_k=3)]
    assert exact_order == ["near_a", "near_b", "far"]

    top_pairs = set()
    for seed in range(12):
        adapter = LocalIndexAdapter(tmp_path, ann_mode=True, ann_sigma=0.01, seed=seed)
        order = [c for c, _ in adapter.query_dense(COL, query, top_k=3)]
        assert order[2] == "far"
        top_pairs.add(tuple(order[:2]))
    assert ("near_a", "near_b") in top_pairs
    assert ("near_b", "near_a") in top_pairs


def test_ann_rebuild_reseeds_noise_stream(tmp_path: Path) -> None:
    adapter = _filled(tmp_path, ann_mode=True, ann_sigma=0.05, seed=3)
    ids, vectors, _ = _toy()
    query = vectors[2]
    first = adapter.query_dense(COL, query, top_k=len(ids))
    second = adapter.query_dense(COL, query, top_k=len(ids))
    assert [s for _, s in second] != [s for _, s in first]

    adapter.rebuild(COL, seed=3)
    assert adapter.query_dense(COL, query, top_k=len(ids)) == first

    adapter.rebuild(COL, seed=99)
    reseeded = adapter.query_dense(COL, query, top_k=len(ids))
    assert [s for _, s in reseeded] != [s for _, s in first]


def test_ann_per_collection_noise_streams_independent(tmp_path: Path) -> None:
    ids, vectors, payloads = _toy()
    query = vectors[5]

    solo = LocalIndexAdapter(tmp_path / "solo", ann_mode=True, ann_sigma=0.05, seed=11)
    solo.upsert("two", ids, vectors, payloads)
    expected_two = solo.query_dense("two", query, top_k=len(ids))

    both = LocalIndexAdapter(tmp_path / "both", ann_mode=True, ann_sigma=0.05, seed=11)
    both.upsert("one", ids, vectors, payloads)
    both.upsert("two", ids, vectors, payloads)
    both.query_dense("one", query, top_k=len(ids))
    assert both.query_dense("two", query, top_k=len(ids)) == expected_two


def test_exact_mode_rebuild_is_noop(tmp_path: Path) -> None:
    adapter = _filled(tmp_path)
    ids, vectors, _ = _toy()
    query = vectors[0]
    before = adapter.query_dense(COL, query, top_k=len(ids))
    adapter.rebuild(COL, seed=42)
    assert adapter.query_dense(COL, query, top_k=len(ids)) == before


def test_unnormalized_upsert_and_query_give_cosine(tmp_path: Path) -> None:
    adapter = LocalIndexAdapter(tmp_path)
    vecs = np.array([[10.0, 0.0], [0.0, 0.5]], dtype=np.float32)
    adapter.upsert(COL, ["ch_x", "ch_y"], vecs, [{}, {}])
    result = adapter.query_dense(COL, np.array([200.0, 0.0], dtype=np.float32), top_k=2)
    assert result[0] == ("ch_x", pytest.approx(1.0, abs=1e-5))
    assert result[1] == ("ch_y", pytest.approx(0.0, abs=1e-5))
