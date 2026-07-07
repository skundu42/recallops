from __future__ import annotations

import math

import pytest

from recallops.bm25 import BM25Index, tokenize

TOY = {
    "d1": "cat sat mat",
    "d2": "cat cat dog",
    "d3": "dog bird",
}


def _idf(n: int, df: int) -> float:
    return math.log((n - df + 0.5) / (df + 0.5) + 1.0)


def _tf_comp(tf: int, k1: float, b: float, dl: int, avgdl: float) -> float:
    return tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * dl / avgdl))


class TestTokenize:
    def test_lowercase_alnum(self):
        assert tokenize("Hello, World-2!") == ["hello", "world", "2"]

    def test_underscore_splits(self):
        assert tokenize("foo_bar baz") == ["foo", "bar", "baz"]

    def test_no_stopword_removal(self):
        assert tokenize("The cat and the hat") == ["the", "cat", "and", "the", "hat"]

    def test_empty(self):
        assert tokenize("") == []
        assert tokenize("!!! ---") == []


class TestBM25HandComputed:
    """3-doc toy corpus, every score derived by hand from the Lucene-style formula.

    N = 3, avgdl = (3 + 3 + 2) / 3 = 8/3, k1 = 1.5, b = 0.75.
    df: cat=2, dog=2, sat=1, mat=1, bird=1.
    idf(t) = ln((N - df + 0.5) / (df + 0.5) + 1)  -- always positive.
    score(q, d) = sum over query token OCCURRENCES of
        idf(t) * tf * (k1+1) / (tf + k1 * (1 - b + b * dl/avgdl))
    """

    def test_query_cat_dog_scores(self):
        idx = BM25Index(TOY)
        n, avgdl, k1, b = 3, 8.0 / 3.0, 1.5, 0.75
        idf_cat = _idf(n, 2)
        idf_dog = _idf(n, 2)

        exp_d1 = idf_cat * _tf_comp(1, k1, b, 3, avgdl)
        exp_d2 = idf_cat * _tf_comp(2, k1, b, 3, avgdl) + idf_dog * _tf_comp(1, k1, b, 3, avgdl)
        exp_d3 = idf_dog * _tf_comp(1, k1, b, 2, avgdl)

        got = idx.scores("cat dog")
        assert set(got) == {"d1", "d2", "d3"}
        assert got["d1"] == pytest.approx(exp_d1, abs=1e-6)
        assert got["d2"] == pytest.approx(exp_d2, abs=1e-6)
        assert got["d3"] == pytest.approx(exp_d3, abs=1e-6)

    def test_query_sat_single_doc(self):
        idx = BM25Index(TOY)
        n, avgdl, k1, b = 3, 8.0 / 3.0, 1.5, 0.75
        exp = _idf(n, 1) * _tf_comp(1, k1, b, 3, avgdl)
        got = idx.scores("sat")
        assert got == {"d1": pytest.approx(exp, abs=1e-6)}

    def test_idf_always_positive_even_when_term_in_all_docs(self):
        idx = BM25Index({"a": "common x", "b": "common y", "c": "common z"})
        got = idx.scores("common")
        assert set(got) == {"a", "b", "c"}
        assert all(s > 0 for s in got.values())

    def test_repeated_query_token_counts_per_occurrence(self):
        idx = BM25Index(TOY)
        single = idx.scores("cat")["d1"]
        double = idx.scores("cat cat")["d1"]
        assert double == pytest.approx(2.0 * single, abs=1e-9)

    def test_custom_k1_b(self):
        idx = BM25Index(TOY, k1=1.2, b=0.0)
        n = 3
        exp = _idf(n, 2) * (1 * 2.2 / (1 + 1.2))
        assert idx.scores("cat")["d1"] == pytest.approx(exp, abs=1e-6)


class TestScoresAndTop:
    def test_scores_only_positive_docs(self):
        idx = BM25Index(TOY)
        got = idx.scores("bird")
        assert set(got) == {"d3"}

    def test_scores_unknown_term_empty(self):
        idx = BM25Index(TOY)
        assert idx.scores("zebra") == {}
        assert idx.scores("") == {}

    def test_top_order_and_truncation(self):
        idx = BM25Index(TOY)
        top_all = idx.top("cat dog", 10)
        assert [c for c, _ in top_all] == ["d2", "d3", "d1"]
        assert idx.top("cat dog", 2) == top_all[:2]
        scores = [s for _, s in top_all]
        assert scores == sorted(scores, reverse=True)

    def test_top_tiebreak_corpus_insertion_order(self):
        idx = BM25Index({"x2": "same words", "x1": "same words"})
        assert [c for c, _ in idx.top("same", 5)] == ["x2", "x1"]

    def test_empty_corpus(self):
        idx = BM25Index({})
        assert idx.scores("anything") == {}
        assert idx.top("anything", 5) == []
        assert idx.rank_of("anything", "d1") is None


class TestRankOf:
    def test_rank_of_matches_top_positions(self):
        idx = BM25Index(TOY)
        top = idx.top("cat dog", 10)
        for i, (cid, _) in enumerate(top):
            assert idx.rank_of("cat dog", cid) == i + 1

    def test_rank_of_none_for_zero_score(self):
        idx = BM25Index(TOY)
        assert idx.rank_of("bird", "d1") is None

    def test_rank_of_none_for_unknown_doc(self):
        idx = BM25Index(TOY)
        assert idx.rank_of("cat", "nope") is None


class TestDeterminism:
    def test_identical_inputs_identical_outputs(self):
        a = BM25Index(TOY)
        b = BM25Index(dict(TOY))
        assert a.scores("cat dog bird") == b.scores("cat dog bird")
        assert a.top("cat dog bird", 3) == b.top("cat dog bird", 3)
