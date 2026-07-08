"""Confirmation rule and attribution assembly (PRD FR-8).

The confirmation rule is the product's core invariant (FR-8.1): a changed factor
is emitted as a *verified* cause for a regressed query iff the arm that reverts
that factor while holding every other factor at B recovers the query, the
target document returns within its original rank (``recovery_threshold="rank1"``,
default) or within ``top_k`` (``"topk"``). Factors the funnel implicated but
whose revert arm did not recover are emitted as *hypotheses* with the funnel
evidence that suggested them (FR-8.2); no factor is ever both. Narratives are
left empty here and filled by the renderer (Task 15).
"""
from __future__ import annotations

from .funnel import failing_stage, implicated_factors
from .models import (
    Arm,
    ArmResult,
    AttributionReport,
    ChunkFate,
    DiffResult,
    FunnelReport,
    GoldenDataset,
    Hypothesis,
    VerifiedCause,
)

__all__ = ["confirm_causes", "fidelity_check", "RECOVERY_THRESHOLDS", "PRIMARY_METRIC"]

RECOVERY_THRESHOLDS = ("rank1", "topk")
PRIMARY_METRIC = "recall@5"

_STAGE_EVIDENCE = {
    "dense": "dense rank drop co-occurred",
    "sparse": "sparse rank drop co-occurred",
    "fused": "fused rank drop co-occurred",
    "index": "target absent from index after change",
    "rerank": "rerank dropped target below top-k",
    "ann": "live dense rank diverged from exact shadow",
}


def _all_factor_names(arms: list[Arm]) -> list[str]:
    names: set[str] = set()
    for arm in arms:
        names.update(arm.assignment.keys())
    return sorted(names)


def _single_revert_arm_id(factor: str, all_names: list[str]) -> str:
    assignment = {n: ("A" if n == factor else "B") for n in all_names}
    return Arm.build(assignment).arm_id


def _recovered(recovered_rank: int | None, before_rank: int | None,
               recovery_threshold: str, top_k: int) -> bool:
    if recovered_rank is None:
        return False
    if recovery_threshold == "topk":
        return recovered_rank <= top_k
    bound = (before_rank + 1) if before_rank is not None else top_k
    return recovered_rank <= bound


def _metric_restored(before_eval, arm_eval, primary_metric: str) -> bool:
    """Whether the arm restores the regression metric to at least its pre-change
    value, the honest, multi-source-correct definition of "recovers the query"
    (FR-8.1). A single-source ``target_rank`` check cannot see a second expected
    source being evicted, so metric restoration is a *necessary* condition for a
    verified cause. Deferred (returns True) only when the metric is unavailable
    on either side, in which case the rank signal governs alone."""
    before_val = before_eval.metrics.get(primary_metric)
    arm_val = arm_eval.metrics.get(primary_metric)
    if before_val is None or arm_val is None:
        return True
    return arm_val >= before_val


def confirm_causes(diffres: DiffResult, dataset: GoldenDataset,
                   arm_results: dict[str, ArmResult], arms: list[Arm],
                   funnel_reports: dict[str, FunnelReport],
                   alignment: dict[str, ChunkFate],
                   recovery_threshold: str = "rank1", top_k: int = 5,
                   primary_metric: str = PRIMARY_METRIC) -> dict[str, AttributionReport]:
    """Attribution report per stable regressed query (FR-8.1, FR-8.2).

    For each factor, the single-revert arm (that factor at A, the rest at B) is
    looked up in ``arm_results``; its target rank for the query is compared to
    the before rank under ``recovery_threshold``. Recovered factors become
    ``VerifiedCause``; funnel-implicated factors that did not recover become
    ``Hypothesis``. ``chunk_fate`` is the alignment entry for the query's tracer.
    """
    if recovery_threshold not in RECOVERY_THRESHOLDS:
        raise ValueError(
            f"unknown recovery_threshold {recovery_threshold!r}; expected one of {RECOVERY_THRESHOLDS}")

    all_names = _all_factor_names(arms)
    reports: dict[str, AttributionReport] = {}

    for qdiff in diffres.by_class("regressed", stable_only=True):
        qid = qdiff.query_id
        funnel = funnel_reports.get(qid)
        if funnel is None:
            continue
        before_rank = qdiff.before.target_rank

        verified: list[VerifiedCause] = []
        verified_factors: set[str] = set()
        for factor in all_names:
            arm_id = _single_revert_arm_id(factor, all_names)
            result = arm_results.get(arm_id)
            if result is None:
                continue
            query_eval = result.eval.per_query.get(qid)
            if query_eval is None:
                continue
            recovered_rank = query_eval.target_rank
            if (_recovered(recovered_rank, before_rank, recovery_threshold, top_k)
                    and _metric_restored(qdiff.before, query_eval, primary_metric)):
                verified.append(VerifiedCause(
                    factor=factor, arm_id=arm_id, recovered_rank=recovered_rank, status="verified"))
                verified_factors.add(factor)

        stage = failing_stage(funnel, top_k)
        implicated = implicated_factors(stage, diffres.config_diff)
        hypotheses = [
            Hypothesis(factor=factor, status="unverified",
                       evidence=_STAGE_EVIDENCE.get(stage, f"{stage} stage implicated"))
            for factor in implicated
            if factor not in verified_factors
        ]

        tracer = funnel.target_chunk_before
        chunk_fate = alignment.get(tracer) if tracer else None

        reports[qid] = AttributionReport(
            query_id=qid,
            classification=qdiff.classification,
            stability=qdiff.stability,
            funnel=funnel,
            chunk_fate=chunk_fate,
            verified_causes=verified,
            hypotheses=hypotheses,
            narrative="",
        )
    return reports


def fidelity_check(reports: dict[str, AttributionReport], arm_results: dict[str, ArmResult],
                   arms: list[Arm], dataset: GoldenDataset, top_k: int,
                   diffres: DiffResult | None = None,
                   primary_metric: str = PRIMARY_METRIC) -> float:
    """Fraction of emitted verified causes whose arm eval still supports the
    claim (§13 Fidelity, ~1.0 *by construction*, a release blocker below 0.995).

    This is a structural consistency tripwire, not an independent re-evaluation:
    arms are content-addressed and replayed deterministically, so re-fetching a
    cause's arm eval yields the same numbers ``confirm_causes`` saw. It re-reads
    each cause's arm eval and re-applies the recovery predicate independently —
    the arm must still rank the target within the claimed ``recovered_rank`` and,
    when ``diffres`` is supplied, still restore the primary metric to at least
    its pre-change value. In a healthy pipeline every emitted cause passes (hence
    ~1.0 by construction, exactly as the PRD states); the value is that it drops
    below 1.0 if a future change lets ``confirm_causes`` emit a cause its own
    recovery predicate does not support, or a stored arm eval is corrupted. It is
    deliberately threshold-independent and cannot false-fail a cause legitimately
    verified under the ``topk`` recovery threshold (recovered above ``top_k``),
    which for a release blocker matters more than catching a hypothetical the
    deterministic replay cannot produce."""
    total = 0
    genuine = 0
    for qid, report in reports.items():
        before_eval = diffres.queries[qid].before if diffres and qid in diffres.queries else None
        for cause in report.verified_causes:
            total += 1
            result = arm_results.get(cause.arm_id)
            if result is None:
                continue
            query_eval = result.eval.per_query.get(qid)
            if query_eval is None or query_eval.target_rank is None:
                continue
            if cause.recovered_rank is not None and query_eval.target_rank > cause.recovered_rank:
                continue  # the arm no longer recovers to the emitted rank
            if before_eval is not None and not _metric_restored(before_eval, query_eval, primary_metric):
                continue
            genuine += 1
    return 1.0 if total == 0 else genuine / total
