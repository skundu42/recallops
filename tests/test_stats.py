from __future__ import annotations

import math
import random

import pytest

from recallops.stats import bh_fdr, bootstrap_ci, derive_epsilon, mcnemar_exact_p


class TestMcnemarExactP:
    def test_hand_computed_b5_c0(self):
        assert mcnemar_exact_p(5, 0) == pytest.approx(2 * 0.5**5)
        assert mcnemar_exact_p(5, 0) == pytest.approx(0.0625)

    def test_zero_zero_is_one(self):
        assert mcnemar_exact_p(0, 0) == 1.0

    def test_symmetric(self):
        for b, c in [(5, 0), (7, 2), (3, 3), (10, 1), (0, 4)]:
            assert mcnemar_exact_p(b, c) == mcnemar_exact_p(c, b)

    def test_hand_computed_b8_c2(self):
        expected = 2 * (math.comb(10, 0) + math.comb(10, 1) + math.comb(10, 2)) * 0.5**10
        assert mcnemar_exact_p(8, 2) == pytest.approx(expected)
        assert mcnemar_exact_p(8, 2) == pytest.approx(0.109375)

    def test_balanced_counts_capped_at_one(self):
        assert mcnemar_exact_p(1, 1) == 1.0
        assert mcnemar_exact_p(6, 6) == 1.0

    def test_one_sided_extreme_small(self):
        assert mcnemar_exact_p(20, 0) == pytest.approx(2 * 0.5**20)

    def test_bounds(self):
        for b, c in [(0, 0), (1, 0), (5, 3), (12, 12), (30, 1)]:
            p = mcnemar_exact_p(b, c)
            assert 0.0 < p <= 1.0


class TestBootstrapCi:
    def test_deterministic_given_seed(self):
        values = [0.1, 0.4, 0.35, 0.8, 0.2, 0.9, 0.5]
        assert bootstrap_ci(values, seed=7) == bootstrap_ci(values, seed=7)

    def test_different_seed_differs(self):
        values = [0.1, 0.4, 0.35, 0.8, 0.2, 0.9, 0.5]
        assert bootstrap_ci(values, seed=0) != bootstrap_ci(values, seed=1)

    def test_covers_true_mean_on_seeded_normal(self):
        rng = random.Random(42)
        values = [rng.gauss(5.0, 1.0) for _ in range(200)]
        lo, hi = bootstrap_ci(values, seed=0)
        assert lo < 5.0 < hi
        sample_mean = sum(values) / len(values)
        assert lo < sample_mean < hi

    def test_interval_ordering_and_range(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        lo, hi = bootstrap_ci(values, seed=3)
        assert lo <= hi
        assert min(values) <= lo and hi <= max(values)

    def test_empty_degenerate(self):
        assert bootstrap_ci([]) == (0.0, 0.0)

    def test_singleton_degenerate(self):
        assert bootstrap_ci([3.5]) == (3.5, 3.5)

    def test_constant_values_zero_width(self):
        lo, hi = bootstrap_ci([2.0] * 10, seed=0)
        assert lo == hi == 2.0

    def test_alpha_widens_interval(self):
        rng = random.Random(1)
        values = [rng.gauss(0.0, 1.0) for _ in range(100)]
        lo95, hi95 = bootstrap_ci(values, alpha=0.05, seed=0)
        lo50, hi50 = bootstrap_ci(values, alpha=0.50, seed=0)
        assert lo95 <= lo50 and hi50 <= hi95


class TestBhFdr:
    def test_textbook_benjamini_hochberg_1995(self):
        pvals = [0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298,
                 0.0344, 0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.0]
        flags = bh_fdr(pvals, q=0.05)
        assert flags == [True] * 4 + [False] * 11

    def test_step_up_rescues_earlier_pvalue(self):
        assert bh_fdr([0.04, 0.049], q=0.05) == [True, True]

    def test_flags_in_input_order(self):
        pvals = [0.5719, 0.0001, 0.3240, 0.0095, 0.0201]
        flags = bh_fdr(pvals, q=0.05)
        assert flags == [False, True, False, True, True]

    def test_flags_permutation_invariant(self):
        pvals = [0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298,
                 0.0344, 0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.0]
        base = bh_fdr(pvals, q=0.05)
        perm = list(range(len(pvals)))
        random.Random(0).shuffle(perm)
        shuffled_flags = bh_fdr([pvals[i] for i in perm], q=0.05)
        assert shuffled_flags == [base[i] for i in perm]

    def test_none_rejected(self):
        assert bh_fdr([0.5, 0.6, 0.9], q=0.05) == [False, False, False]

    def test_all_rejected(self):
        assert bh_fdr([0.001, 0.002, 0.003], q=0.05) == [True, True, True]

    def test_empty(self):
        assert bh_fdr([]) == []

    def test_single_pvalue(self):
        assert bh_fdr([0.01], q=0.05) == [True]
        assert bh_fdr([0.06], q=0.05) == [False]

    def test_ties_at_cutoff(self):
        assert bh_fdr([0.025, 0.025], q=0.05) == [True, True]


class TestDeriveEpsilon:
    def test_empty(self):
        assert derive_epsilon([]) == 0.0

    def test_exact_grid_percentile(self):
        samples = [i / 20 for i in range(21)]
        assert derive_epsilon(samples) == pytest.approx(0.95)

    def test_linear_interpolation(self):
        assert derive_epsilon([1.0, 2.0, 3.0, 4.0]) == pytest.approx(3.85)

    def test_absolute_values_used(self):
        assert derive_epsilon([-1.0, -2.0, -3.0, -4.0]) == pytest.approx(3.85)
        assert derive_epsilon([-1.0, 2.0, -3.0, 4.0]) == pytest.approx(3.85)

    def test_single_sample(self):
        assert derive_epsilon([-0.3]) == pytest.approx(0.3)

    def test_constant_samples(self):
        assert derive_epsilon([0.5] * 8) == pytest.approx(0.5)

    def test_order_invariant(self):
        samples = [0.02, -0.5, 0.13, 0.4, -0.01]
        assert derive_epsilon(samples) == derive_epsilon(sorted(samples))
