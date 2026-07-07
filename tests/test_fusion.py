from __future__ import annotations

import pytest

from recallops.fusion import fuse
from recallops.rerankers import get_reranker


class TestWeightedFusion:
    def test_hand_values(self):
        dense = [("a", 0.9), ("b", 0.5), ("c", 0.1)]
        sparse = [("b", 2.0), ("c", 1.0)]
        fused = fuse(dense, sparse, "weighted", {"bm25_weight": 0.5})
        got = dict(fused)
        assert got["a"] == pytest.approx(0.5 * 1.0 + 0.5 * 0.0, abs=1e-9)
        assert got["b"] == pytest.approx(0.5 * 0.5 + 0.5 * 1.0, abs=1e-9)
        assert got["c"] == pytest.approx(0.5 * 0.0 + 0.5 * 0.0, abs=1e-9)
        assert [c for c, _ in fused] == ["b", "a", "c"]

    def test_absent_candidate_contributes_normalized_zero(self):
        """Documented choice: a candidate absent from a list contributes
        normalized 0 for that list (NOT that list's min score)."""
        dense = [("a", 10.0), ("b", 4.0)]
        sparse = [("z", 7.0), ("b", 3.0)]
        fused = dict(fuse(dense, sparse, "weighted", {"bm25_weight": 0.25}))
        assert fused["z"] == pytest.approx(0.75 * 0.0 + 0.25 * 1.0, abs=1e-9)
        assert fused["a"] == pytest.approx(0.75 * 1.0 + 0.25 * 0.0, abs=1e-9)
        assert fused["b"] == pytest.approx(0.75 * 0.0 + 0.25 * 0.0, abs=1e-9)

    def test_w0_equals_dense_order_same_candidates(self):
        dense = [("a", 0.9), ("b", 0.5), ("c", 0.1)]
        sparse = [("c", 5.0), ("a", 2.0), ("b", 1.0)]
        fused = fuse(dense, sparse, "weighted", {"bm25_weight": 0.0})
        assert [c for c, _ in fused] == ["a", "b", "c"]

    def test_w1_equals_sparse_order_same_candidates(self):
        dense = [("a", 0.9), ("b", 0.5), ("c", 0.1)]
        sparse = [("c", 5.0), ("a", 2.0), ("b", 1.0)]
        fused = fuse(dense, sparse, "weighted", {"bm25_weight": 1.0})
        assert [c for c, _ in fused] == ["c", "a", "b"]

    def test_w0_preserves_dense_relative_order_over_union(self):
        dense = [("a", 0.9), ("b", 0.5)]
        sparse = [("z", 3.0), ("b", 1.0)]
        fused = fuse(dense, sparse, "weighted", {"bm25_weight": 0.0})
        assert [c for c, _ in fused] == ["a", "b", "z"]

    def test_w1_preserves_sparse_relative_order_over_union(self):
        dense = [("a", 0.9), ("b", 0.5)]
        sparse = [("z", 3.0), ("b", 1.0)]
        fused = fuse(dense, sparse, "weighted", {"bm25_weight": 1.0})
        order = [c for c, _ in fused]
        assert order[0] == "z"
        assert [c for c in order if c in {"z", "b"}] == ["z", "b"]

    def test_degenerate_range_normalizes_to_one(self):
        """Documented choice: a list whose scores span zero range (single
        candidate or all-equal) normalizes every present candidate to 1.0,
        so presence still beats absence."""
        dense = [("a", 0.7)]
        sparse = [("b", 1.0), ("c", 0.5)]
        fused = dict(fuse(dense, sparse, "weighted", {"bm25_weight": 0.5}))
        assert fused["a"] == pytest.approx(0.5, abs=1e-9)
        assert fused["b"] == pytest.approx(0.5, abs=1e-9)
        assert fused["c"] == pytest.approx(0.0, abs=1e-9)

    def test_tiebreak_dense_rank_then_sparse_rank_then_id(self):
        dense = [("a", 0.7)]
        sparse = [("b", 1.0), ("c", 0.5)]
        fused = fuse(dense, sparse, "weighted", {"bm25_weight": 0.5})
        assert [c for c, _ in fused] == ["a", "b", "c"]

    def test_empty_lists(self):
        assert fuse([], [], "weighted", {"bm25_weight": 0.5}) == []
        fused = fuse([("a", 1.0), ("b", 0.5)], [], "weighted", {"bm25_weight": 0.5})
        assert [c for c, _ in fused] == ["a", "b"]


class TestRRF:
    def test_known_values_default_k0(self):
        dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        sparse = [("b", 5.0), ("d", 4.0)]
        fused = fuse(dense, sparse, "rrf", {})
        got = dict(fused)
        assert got["a"] == pytest.approx(1.0 / 61.0, abs=1e-12)
        assert got["b"] == pytest.approx(1.0 / 62.0 + 1.0 / 61.0, abs=1e-12)
        assert got["c"] == pytest.approx(1.0 / 63.0, abs=1e-12)
        assert got["d"] == pytest.approx(1.0 / 62.0, abs=1e-12)
        assert [c for c, _ in fused] == ["b", "a", "d", "c"]

    def test_custom_k0(self):
        fused = dict(fuse([("a", 1.0)], [("a", 1.0)], "rrf", {"k0": 1}))
        assert fused["a"] == pytest.approx(1.0, abs=1e-12)

    def test_tiebreak_by_dense_rank(self):
        dense = [("a", 1.0), ("b", 0.9)]
        sparse = [("b", 1.0), ("a", 0.9)]
        fused = fuse(dense, sparse, "rrf", {})
        assert [c for c, _ in fused] == ["a", "b"]

    def test_rank_based_not_score_based(self):
        low = fuse([("a", 0.001), ("b", 0.0005)], [], "rrf", {})
        high = fuse([("a", 100.0), ("b", 50.0)], [], "rrf", {})
        assert low == high


class TestFuseGeneral:
    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            fuse([("a", 1.0)], [("a", 1.0)], "borda", {})

    def test_weighted_requires_bm25_weight(self):
        with pytest.raises(KeyError):
            fuse([("a", 1.0)], [("a", 1.0)], "weighted", {})

    def test_deterministic(self):
        dense = [("a", 0.9), ("b", 0.5), ("c", 0.1)]
        sparse = [("c", 5.0), ("d", 2.0)]
        for method, params in [("weighted", {"bm25_weight": 0.4}), ("rrf", {})]:
            assert fuse(dense, sparse, method, params) == fuse(dense, sparse, method, params)


class TestOverlapReranker:
    def test_exact_overlap_chunk_first(self):
        rerank = get_reranker("recall.rerankers.overlap", {})
        out = rerank(
            "refund policy window",
            [
                ("c1", "invoices are emailed monthly"),
                ("c2", "the refund policy window is fourteen days"),
                ("c3", "refund requests go to support"),
            ],
        )
        assert [c for c, _ in out] == ["c2", "c3", "c1"]
        got = dict(out)
        assert got["c2"] == pytest.approx(1.0, abs=1e-9)
        assert got["c3"] == pytest.approx(1.0 / 3.0, abs=1e-9)
        assert got["c1"] == pytest.approx(0.0, abs=1e-9)

    def test_overlap_over_unique_query_tokens(self):
        rerank = get_reranker("recall.rerankers.overlap", {})
        out = dict(rerank("cat cat dog", [("c1", "the cat sat")]))
        assert out["c1"] == pytest.approx(0.5, abs=1e-9)

    def test_stable_tiebreak_by_prior_order(self):
        rerank = get_reranker("recall.rerankers.overlap", {})
        out = rerank(
            "gateway timeout",
            [("c2", "gateway routing"), ("c1", "gateway affinity"), ("c3", "storage quota")],
        )
        assert [c for c, _ in out] == ["c2", "c1", "c3"]

    def test_empty_query_all_zero_prior_order(self):
        rerank = get_reranker("recall.rerankers.overlap", {})
        out = rerank("...", [("b", "beta"), ("a", "alpha")])
        assert out == [("b", 0.0), ("a", 0.0)]

    def test_unknown_tool_raises(self):
        with pytest.raises(ValueError):
            get_reranker("recall.rerankers.cross_encoder", {})

    def test_deterministic(self):
        rerank = get_reranker("recall.rerankers.overlap", {})
        cands = [("c1", "alpha beta"), ("c2", "beta gamma")]
        assert rerank("beta gamma", cands) == rerank("beta gamma", cands)
