"""Statistical primitives for gating and calibration (PRD FR-9.2 to FR-9.4).

Pure Python + ``math``; every stochastic step takes an explicit seed so
identical inputs always produce identical outputs.
"""
from __future__ import annotations

import math
import random


def _percentile(sorted_vals: list[float], p: float) -> float:
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    idx = p * (n - 1)
    lo = math.floor(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


def bootstrap_ci(values: list[float], n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean; degenerate interval for n <= 1."""
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        v = float(values[0])
        return (v, v)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(math.fsum(rng.choices(values, k=n)) / n for _ in range(n_boot))
    return (_percentile(means, alpha / 2), _percentile(means, 1 - alpha / 2))


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact binomial McNemar p-value for discordant counts b, c."""
    if b < 0 or c < 0:
        raise ValueError("discordant counts must be non-negative")
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def bh_fdr(pvals: list[float], q: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg step-up; rejection flags in input order."""
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    k_max = 0
    for rank, i in enumerate(order, start=1):
        if pvals[i] <= rank * q / m:
            k_max = rank
    flags = [False] * m
    for rank, i in enumerate(order, start=1):
        if rank > k_max:
            break
        flags[i] = True
    return flags


def derive_epsilon(gap_samples: list[float]) -> float:
    """Near-tie threshold: 95th percentile of |score fluctuation| samples (FR-9.3)."""
    if not gap_samples:
        return 0.0
    return _percentile(sorted(abs(x) for x in gap_samples), 0.95)
