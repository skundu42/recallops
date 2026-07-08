"""Noise-floor calibration and statistical gating (PRD FR-9.2-FR-9.4).

Calibration re-runs an identical snapshot ``n_runs`` times against a serving
adapter, rebuilding the index between runs where the adapter supports it, and
records how much the aggregate metrics and the per-query target-vs-kth score
gap fluctuate under pure serving noise (FR-9.2). The near-tie threshold
``epsilon`` is the 95th percentile of those gap fluctuations (FR-9.3).

``evaluate_gate`` turns a snapshot diff into a pass/fail decision. Statistical
mode (the default, FR-9.4) requires a calibration record and combines three
signals: a bootstrap CI on the stable-query primary-metric delta, a McNemar
test on stable-query hit flips, and per-tag McNemar tests corrected with
Benjamini-Hochberg FDR. Near-tie (``unstable``) queries are excluded from every
red signal — the CI and both flip counts alike — so pure serving noise can
never turn a gate red, the FR-9 never-flaky invariant (risk R1). The cost of
that guarantee is that a regression small enough to appear only as near-ties is
below the calibrated noise floor and deliberately not flagged; the near-tie
count is surfaced in the result details so a reviewer can see how much was
excluded. Raw mode applies a single ``metric op threshold`` condition to
snapshot B's aggregate and is offered only as an escape hatch; docs steer
callers to statistical mode.
"""
from __future__ import annotations

import math
import re

from .adapters.base import VectorAdapter
from .evalrunner import _case_eval
from .ingest import collection_name
from .models import (
    CalibrationRecord,
    DiffResult,
    GateResult,
    GoldenDataset,
    QueryDiff,
    QueryEval,
    SnapshotManifest,
)
from .retrieval import RetrievalEngine
from .stats import bh_fdr, bootstrap_ci, derive_epsilon, mcnemar_exact_p
from .store import ProjectStore

__all__ = ["calibrate", "evaluate_gate", "parse_fail_if", "GateNotCalibrated",
           "NEAR_TIE_K", "OVERALL"]

NEAR_TIE_K = 5  # default near-tie boundary; calibrate() uses the gated metric's k
DEFAULT_K = 10
OVERALL = "__overall__"
CALIBRATION_K_VALUES: tuple[int, ...] = (1, 5, 10)

_FAIL_IF = re.compile(r"^\s*([A-Za-z0-9_@]+)\s*(<=|>=|<|>)\s*([+-]?\d+(?:\.\d+)?)\s*$")
_COMPARATORS = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
}


class GateNotCalibrated(Exception):
    """Raised when statistical gating runs without a calibration record (FR-9.2)."""


def _metric_keys(k_values: tuple[int, ...]) -> list[str]:
    keys: list[str] = []
    for k in k_values:
        keys.extend((f"recall@{k}", f"hit_rate@{k}", f"ndcg@{k}"))
    keys.append("mrr")
    return keys


def _metric_k(metric: str) -> int:
    _, _, suffix = metric.partition("@")
    return int(suffix) if suffix.isdigit() else DEFAULT_K


def _population_std(values: list[float]) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    mean = math.fsum(values) / n
    return math.sqrt(math.fsum((v - mean) ** 2 for v in values) / n)


def _target_gap(qe: QueryEval, k: int) -> float | None:
    """Signed target-vs-kth score gap, or None when there is no k-th anchor."""
    scores = [score for _, score in qe.ranked_chunks]
    if not scores:
        return None

    def at(position: int) -> float:
        return scores[min(max(position, 1), len(scores)) - 1]

    if qe.target_rank is not None:
        return at(qe.target_rank) - at(k)
    # Target absent from the served list is a definite drop, not a near-tie (#2);
    # no k/k+1 proxy. None => not a near-tie anchor, so it stays stable.
    return None


def calibrate(store: ProjectStore, manifest: SnapshotManifest, dataset: GoldenDataset,
              adapter: VectorAdapter, n_runs: int = 3, seed: int = 0,
              primary_metric: str = "recall@5") -> CalibrationRecord:
    """Run ``n_runs`` live evals of the same snapshot and record serving noise.

    The index is rebuilt with ``seed + i`` before run ``i`` when the adapter
    supports rebuild (FR-9.2); the built-in exact adapter's rebuild is a no-op,
    so every run is identical and ``epsilon`` collapses to 0. Per-metric std is
    the population std of the ``n_runs`` aggregate metric values; ``epsilon`` is
    ``derive_epsilon`` over every run-pair fluctuation of the target-vs-kth score
    gap and ``per_query_flip_rate`` the fraction of run-pairs whose hit@k
    disagrees, both taken at the gated metric's k (``primary_metric``), so the
    near-tie boundary matches the boundary the gate scores flips at (#4).
    Persisted under ``("calibration", snapshot_id)``.
    """
    near_tie_k = _metric_k(primary_metric)
    k_values = tuple(sorted(set(CALIBRATION_K_VALUES) | {near_tie_k}))
    collection = collection_name(manifest, store.project)
    supports_rebuild = adapter.capabilities().supports_rebuild
    cases = list(dataset.cases)

    runs: list[dict[str, QueryEval]] = []
    for i in range(n_runs):
        if supports_rebuild:
            adapter.rebuild(collection, seed=seed + i)
        engine = RetrievalEngine(store, manifest, adapter=adapter)
        doc_map = engine.chunk_doc_paths()
        per_query: dict[str, QueryEval] = {}
        for case in cases:
            run = engine.run_query(case.id, case.question)
            per_query[case.id] = _case_eval(case, run, doc_map, k_values)
        runs.append(per_query)

    n_cases = len(cases)
    per_metric_std: dict[str, float] = {}
    for key in _metric_keys(CALIBRATION_K_VALUES):
        aggregates = [
            (math.fsum(pq[case.id].metrics[key] for case in cases) / n_cases) if n_cases else 0.0
            for pq in runs
        ]
        per_metric_std[key] = _population_std(aggregates)

    samples: list[float] = []
    for case in cases:
        gaps = [_target_gap(runs[i][case.id], near_tie_k) for i in range(n_runs)]
        for i in range(n_runs):
            for j in range(i + 1, n_runs):
                if gaps[i] is not None and gaps[j] is not None:
                    samples.append(abs(gaps[i] - gaps[j]))
    epsilon = derive_epsilon(samples)

    total_pairs = n_runs * (n_runs - 1) // 2
    per_query_flip_rate: dict[str, float] = {}
    for case in cases:
        if total_pairs == 0:
            per_query_flip_rate[case.id] = 0.0
            continue
        flips = sum(
            1
            for i in range(n_runs)
            for j in range(i + 1, n_runs)
            if runs[i][case.id].hit_at.get(near_tie_k) != runs[j][case.id].hit_at.get(near_tie_k)
        )
        per_query_flip_rate[case.id] = flips / total_pairs

    record = CalibrationRecord(
        snapshot_id=manifest.snapshot_id,
        n_runs=n_runs,
        per_metric_std=per_metric_std,
        epsilon=epsilon,
        per_query_flip_rate=per_query_flip_rate,
        created_at="",
    )
    store.save_json("calibration", manifest.snapshot_id, record.to_dict())
    return record


def parse_fail_if(expr: str) -> tuple[str, str, float]:
    """Parse a raw-threshold expression like ``"recall@5<0.85"``.

    Returns ``(metric, op, threshold)`` with ``op`` in ``{<, <=, >, >=}``.
    Raises ``ValueError`` on anything that is not exactly one such comparison.
    """
    match = _FAIL_IF.match(expr or "")
    if match is None:
        raise ValueError(f"invalid fail-if expression: {expr!r}")
    return match.group(1), match.group(2), float(match.group(3))


def _primary_delta(qd: QueryDiff, primary_metric: str) -> float:
    if primary_metric in qd.metric_delta:
        return qd.metric_delta[primary_metric]
    return qd.after.metrics.get(primary_metric, 0.0) - qd.before.metrics.get(primary_metric, 0.0)


def _flip_counts(qdiffs: list[QueryDiff], k: int) -> tuple[int, int]:
    b = c = 0
    for qd in qdiffs:
        before = qd.before.hit_at.get(k, False)
        after = qd.after.hit_at.get(k, False)
        if before and not after:
            b += 1
        elif after and not before:
            c += 1
    return b, c


def _unstable_ids(diffres: DiffResult) -> list[str]:
    return sorted(qid for qid, qd in diffres.queries.items() if qd.stability == "unstable")


def _raw_gate(diffres: DiffResult, fail_if: str | None) -> GateResult:
    if not fail_if:
        raise ValueError("raw gate mode requires a fail_if expression")
    metric, op, threshold = parse_fail_if(fail_if)
    values = [qd.after.metrics[metric] for qd in diffres.queries.values()
              if metric in qd.after.metrics]
    after_value = math.fsum(values) / len(values) if values else 0.0
    condition_met = _COMPARATORS[op](after_value, threshold)
    reasons = ([] if not condition_met
               else [f"{metric}={after_value:.4f} {op} {threshold} (fail-if condition met)"])
    details = {
        "metric": metric,
        "op": op,
        "threshold": threshold,
        "after_value": after_value,
        "condition_met": condition_met,
    }
    return GateResult(
        passed=not condition_met,
        mode="raw",
        reasons=reasons,
        significant_regression=condition_met,
        unstable_query_ids=_unstable_ids(diffres),
        details=details,
    )


def evaluate_gate(diffres: DiffResult, calibration: CalibrationRecord | None,
                  mode: str = "statistical", fail_if: str | None = None,
                  primary_metric: str = "recall@5", q: float = 0.05, seed: int = 0,
                  dataset: GoldenDataset | None = None) -> GateResult:
    """Decide whether a diff passes the release gate (FR-9.4).

    Statistical mode (requires ``calibration``): a bootstrap 95% CI on the
    stable queries' primary-metric deltas (near-ties excluded) flags a
    significant regression when its upper bound is below 0; McNemar's exact test
    on stable-query hit flips plus per-tag McNemar tests corrected with
    Benjamini-Hochberg FDR flag a one-way flip excess. The gate fails on either
    signal. Near-ties feed neither, so serving noise never reddens the gate.
    Raw mode (selected by ``mode="raw"`` or by passing ``fail_if``) applies a
    single threshold to snapshot B's aggregate and needs no calibration.

    ``dataset`` is optional; when supplied, its per-case tags drive the per-tag
    McNemar/FDR pass (a documented extension of the plan signature).
    """
    if mode == "raw" or fail_if is not None:
        return _raw_gate(diffres, fail_if)
    if mode != "statistical":
        raise ValueError(f"unknown gate mode {mode!r}; expected 'statistical' or 'raw'")
    if calibration is None:
        raise GateNotCalibrated(
            "statistical gating requires a calibration record; run `calibrate` first (FR-9.2)"
        )

    k = _metric_k(primary_metric)
    unstable_ids = _unstable_ids(diffres)
    stable = [qd for qd in diffres.queries.values() if qd.stability == "stable"]
    unstable = [qd for qd in diffres.queries.values() if qd.stability == "unstable"]

    # Near-tie (unstable) queries are quarantined from EVERY red signal — the CI
    # and both flip counts alike. A single near-tie flip carries a full ±1.0
    # metric delta yet is, by calibration, within serving noise; letting it feed
    # any signal (even a directional one) gives pure noise a nonzero chance to
    # redden the gate. Never-flaky (FR-9, risk R1) is paramount, so a systematic
    # sub-epsilon regression that manifests only as near-ties is deliberately not
    # caught here — it is below the calibrated noise floor by definition. The
    # near-tie count is surfaced in ``details`` for visibility instead.
    deltas = [_primary_delta(qd, primary_metric) for qd in stable]
    ci = bootstrap_ci(deltas, seed=seed)
    significant_regression = ci[1] < 0.0

    b, c = _flip_counts(stable, k)
    p_overall = mcnemar_exact_p(b, c)

    per_tag: dict[str, dict[str, float]] = {}
    if dataset is not None:
        tags_by_id = {case.id: list(case.tags) for case in dataset.cases}
        grouped: dict[str, list[QueryDiff]] = {}
        for qd in stable:
            for tag in tags_by_id.get(qd.query_id, []):
                grouped.setdefault(tag, []).append(qd)
        for tag in sorted(grouped):
            tb, tc = _flip_counts(grouped[tag], k)
            per_tag[tag] = {"p": mcnemar_exact_p(tb, tc), "b": float(tb), "c": float(tc)}

    # The overall McNemar is an INDEPENDENT gate signal judged on its own q
    # threshold; BH-FDR corrects only the per-tag subgroup family (FR-9.4). Folding
    # the overall test into the per-tag family would raise its rejection bar as
    # more tag categories are added, so a global regression would get *harder* to
    # detect the more finely a dataset is tagged (finding #3).
    per_tag_labels = [(tag, per_tag[tag]["p"], int(per_tag[tag]["b"]), int(per_tag[tag]["c"]))
                      for tag in sorted(per_tag)]
    tag_rejected = bh_fdr([p for _, p, _, _ in per_tag_labels], q=q)
    labels = [(OVERALL, p_overall, b, c), *per_tag_labels]
    rejected = [p_overall <= q, *tag_rejected]
    bh_rejected = {label: flag for (label, *_), flag in zip(labels, rejected)}
    mcnemar_regression = any(
        flag and lb > lc for flag, (_, _, lb, lc) in zip(rejected, labels)
    )

    passed = not (significant_regression or mcnemar_regression)

    reasons: list[str] = []
    if significant_regression:
        reasons.append(
            f"stable-query {primary_metric} regressed: 95% CI upper bound {ci[1]:.4f} < 0 "
            f"(over {len(stable)} stable queries)"
        )
    if mcnemar_regression:
        for flag, (label, p, lb, lc) in zip(rejected, labels):
            if flag and lb > lc:
                name = "overall" if label == OVERALL else f"tag '{label}'"
                reasons.append(
                    f"significant {name} hit-flip regression (b={lb}, c={lc}, p={p:.4f})"
                )

    details = {
        "primary_metric": primary_metric,
        "ci": [ci[0], ci[1]],
        "b": b,
        "c": c,
        "p_overall": p_overall,
        "per_tag": {tag: per_tag[tag]["p"] for tag in per_tag},
        "per_tag_detail": per_tag,
        "bh_rejected": bh_rejected,
        "mcnemar_significant": mcnemar_regression,
        "near_tie_excluded": len(unstable_ids),
        "n_stable": len(stable),
        "n_unstable": len(unstable),
    }
    return GateResult(
        passed=passed,
        mode="statistical",
        reasons=reasons,
        significant_regression=significant_regression,
        unstable_query_ids=unstable_ids,
        details=details,
    )
