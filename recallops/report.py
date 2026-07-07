"""Reporting artifacts (PRD FR-13, FR-9.5, FR-10.1, §10.4).

All reports render from stored structured artifacts (``DiffResult``,
``AttributionReport``, ``GateResult``, ``EvalResult``) with no re-execution, so
``recall report --diff <id>`` is fully reproducible. ``attribution_to_dict``
emits the normative §10.4 JSON shape; ``diff_summary_markdown`` is the PR-comment
report (FR-9.5); ``diff_report_html`` is a self-contained static page (FR-13);
``compare_report_markdown`` is the migration artifact (FR-10.1); ``eval_table``
and ``diff_table`` are the CLI ``rich`` tables; ``render_diff_json`` is the
stable ``--format json`` schema.
"""
from __future__ import annotations

import html

from rich.table import Table

from .models import AttributionReport, DiffResult, EvalResult, GateResult, VerifiedCause

__all__ = [
    "attribution_to_dict",
    "diff_summary_markdown",
    "diff_report_html",
    "eval_table",
    "diff_table",
    "compare_report_markdown",
    "render_diff_json",
]

PRIMARY_METRIC = "recall@5"


def attribution_to_dict(rep: AttributionReport) -> dict:
    """The §10.4 attribution-report JSON shape (delegates to ``rep.to_dict()``)."""
    return rep.to_dict()


def _collect_causes(attributions: dict[str, AttributionReport] | None
                    ) -> list[tuple[str, VerifiedCause]]:
    causes: list[tuple[str, VerifiedCause]] = []
    for qid in sorted(attributions or {}):
        for vc in (attributions or {})[qid].verified_causes:
            causes.append((qid, vc))
    return causes


def _stage_desc(spec: dict | None) -> str:
    if spec is None:
        return "(none)"
    if "merkle_root" in spec:
        return str(spec["merkle_root"])
    label = spec.get("tool") or spec.get("adapter") or spec.get("id", "")
    params = spec.get("params")
    return f"{label} {params}".strip() if params else str(label)


def _config_line(stage_id: str, change: dict) -> str:
    before = _stage_desc(change.get("before"))
    after = _stage_desc(change.get("after"))
    return f"**{stage_id}**: {before} → {after}"


def diff_summary_markdown(diffres: DiffResult,
                          attributions: dict[str, AttributionReport] | None,
                          gate: GateResult | None,
                          calibration_ok: bool) -> str:
    """PR-comment Markdown for a diff (FR-9.5): metric deltas with the gate's
    bootstrap CI on the primary metric, regressed/improved/unstable counts, top
    verified causes, and the config diff, with detail behind ``<details>`` folds
    and an explicit calibration-presence line."""
    lines: list[str] = ["## RecallOps retrieval diff", ""]

    if gate is not None:
        verdict = "PASS ✅" if gate.passed else "FAIL ❌"
        lines.append(f"**Gate: {verdict}** (mode: {gate.mode})")
        for reason in gate.reasons:
            lines.append(f"- {reason}")
        lines.append("")

    lines.append(f"Calibration: {'present' if calibration_ok else 'not present'}.")
    lines.append("")

    regressed = diffres.by_class("regressed")
    # Headline excludes near-tie flips so the number matches the "(excluded)"
    # label and the gate's own McNemar counts (FR-9.3, finding #8). The detail
    # fold below still lists every regressed row with its stability column.
    stable_regressed = len(diffres.by_class("regressed", stable_only=True))
    improved = diffres.by_class("improved")
    changed = diffres.by_class("changed-top-k")
    unstable = sum(1 for qd in diffres.queries.values() if qd.stability == "unstable")
    lines.append(
        f"**Queries:** {stable_regressed} regressed · {len(improved)} improved · "
        f"{len(changed)} changed-top-k · {unstable} unstable (excluded)"
    )
    lines.append("")

    details = gate.details if gate is not None else {}
    primary = details.get("primary_metric", PRIMARY_METRIC)
    ci = details.get("ci")
    lines.append("| Metric | Δ | 95% CI |")
    lines.append("|---|---|---|")
    for metric in sorted(diffres.metric_deltas):
        delta = diffres.metric_deltas[metric]
        ci_txt = ""
        if metric == primary and ci is not None:
            ci_txt = f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
        lines.append(f"| {metric} | {delta:+.4f} | {ci_txt} |")
    lines.append("")

    causes = _collect_causes(attributions)
    if causes:
        lines.append("**Top verified causes:**")
        for qid, vc in causes[:5]:
            lines.append(
                f"- `{qid}`: {vc.factor} → rank {vc.recovered_rank} (`{vc.arm_id}`)"
            )
        lines.append("")

    lines.append("<details><summary>Config diff</summary>")
    lines.append("")
    if diffres.config_diff:
        for stage_id in sorted(diffres.config_diff):
            lines.append(f"- {_config_line(stage_id, diffres.config_diff[stage_id])}")
    else:
        lines.append("- no config changes")
    lines.append("")
    lines.append("</details>")
    lines.append("")

    lines.append("<details><summary>Regressed queries</summary>")
    lines.append("")
    lines.append(f"| Query | Δ{primary} | stability |")
    lines.append("|---|---|---|")
    for qd in regressed:
        delta = qd.metric_delta.get(primary, 0.0)
        lines.append(f"| {qd.query_id} | {delta:+.4f} | {qd.stability} |")
    lines.append("")
    lines.append("</details>")

    return "\n".join(lines)


_HTML_STYLE = (
    "body{font-family:system-ui,sans-serif;margin:2rem;color:#1a1a1a;background:#fff}"
    "h1{font-size:1.4rem}table{border-collapse:collapse;width:100%;margin-top:1rem}"
    "th,td{border:1px solid #ccc;padding:6px 10px;text-align:left;font-size:0.9rem}"
    "th{background:#f2f2f2}.gate.pass{color:#0a7d33;font-weight:600}"
    ".gate.fail{color:#b00020;font-weight:600}.regressed{color:#b00020}"
    ".improved{color:#0a7d33}"
)


def _esc(value: object) -> str:
    return html.escape(str(value))


def diff_report_html(diffres: DiffResult,
                     attributions: dict[str, AttributionReport] | None,
                     gate: GateResult | None) -> str:
    """Self-contained static HTML diff report (FR-13): inline ``<style>``, no
    external CSS/JS/urls, a per-query table of classification, stability, primary
    metric delta, and verified causes."""
    attributions = attributions or {}
    rows: list[str] = []
    for qid in sorted(diffres.queries):
        qd = diffres.queries[qid]
        rep = attributions.get(qid)
        causes = ""
        if rep is not None:
            causes = "; ".join(
                f"{_esc(vc.factor)} → rank {_esc(vc.recovered_rank)} "
                f"({_esc(vc.arm_id)})"
                for vc in rep.verified_causes
            )
        delta = qd.metric_delta.get(PRIMARY_METRIC, 0.0)
        rows.append(
            f"<tr><td>{_esc(qid)}</td>"
            f"<td class='{_esc(qd.classification)}'>{_esc(qd.classification)}</td>"
            f"<td>{_esc(qd.stability)}</td>"
            f"<td>{delta:+.4f}</td><td>{causes}</td></tr>"
        )

    gate_html = ""
    if gate is not None:
        cls = "pass" if gate.passed else "fail"
        verdict = "PASS" if gate.passed else "FAIL"
        gate_html = f"<p class='gate {cls}'>Gate: {verdict} (mode: {_esc(gate.mode)})</p>"

    return (
        "<!DOCTYPE html>\n"
        "<html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>RecallOps diff report</title>"
        f"<style>{_HTML_STYLE}</style></head><body>"
        "<h1>RecallOps diff report</h1>"
        f"<p>Snapshot A <code>{_esc(diffres.snapshot_a)}</code> &rarr; "
        f"B <code>{_esc(diffres.snapshot_b)}</code> &middot; dataset "
        f"<code>{_esc(diffres.dataset_id)}</code></p>"
        f"{gate_html}"
        "<table><thead><tr><th>Query</th><th>Classification</th><th>Stability</th>"
        f"<th>&Delta;{_esc(PRIMARY_METRIC)}</th><th>Verified causes</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        "</body></html>"
    )


def eval_table(ev: EvalResult) -> Table:
    """A ``rich`` table of an eval's aggregate metrics."""
    table = Table(title=f"Eval {ev.run_id} · snapshot {ev.snapshot_id} · {ev.mode}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for metric in sorted(ev.aggregate):
        table.add_row(metric, f"{ev.aggregate[metric]:.4f}")
    return table


def diff_table(diffres: DiffResult) -> Table:
    """A ``rich`` table of per-query classification, stability, and primary delta."""
    table = Table(title=f"Diff {diffres.diff_id}")
    table.add_column("Query")
    table.add_column("Classification")
    table.add_column("Stability")
    table.add_column(f"Δ{PRIMARY_METRIC}", justify="right")
    for qid in sorted(diffres.queries):
        qd = diffres.queries[qid]
        delta = qd.metric_delta.get(PRIMARY_METRIC, 0.0)
        table.add_row(qid, qd.classification, qd.stability, f"{delta:+.4f}")
    return table


def compare_report_markdown(tag_deltas: dict[str, dict[str, float]],
                            overall: dict[str, float],
                            recommendation: dict,
                            cost: dict) -> str:
    """Migration comparison Markdown (FR-10.1): per-tag metric-delta table, an
    overall row, a recommendation block, and the printed cost."""
    metrics = sorted({m for deltas in tag_deltas.values() for m in deltas} | set(overall))
    lines: list[str] = ["# Embedding comparison report", ""]

    lines.append("| Tag | " + " | ".join(metrics) + " |")
    lines.append("|---|" + "|".join(["---"] * len(metrics)) + "|")
    for tag in sorted(tag_deltas):
        cells = " | ".join(f"{tag_deltas[tag].get(m, 0.0):+.4f}" for m in metrics)
        lines.append(f"| {tag} | {cells} |")
    overall_cells = " | ".join(f"{overall.get(m, 0.0):+.4f}" for m in metrics)
    lines.append(f"| **overall** | {overall_cells} |")
    lines.append("")

    lines.append("## Recommendation")
    lines.append("")
    if recommendation:
        for key in sorted(recommendation):
            lines.append(f"- **{key}**: {recommendation[key]}")
    else:
        lines.append("- no recommendation")
    lines.append("")

    lines.append("## Cost")
    lines.append("")
    usd = cost.get("usd")
    if usd is not None:
        lines.append(f"- Estimated spend: ${usd:.2f}")
    wall = cost.get("wall_s")
    if wall is not None:
        lines.append(f"- Wall time: {wall:.0f}s")
    for key in sorted(cost):
        if key not in ("usd", "wall_s"):
            lines.append(f"- {key}: {cost[key]}")

    return "\n".join(lines)


def render_diff_json(diffres: DiffResult,
                     attributions: dict[str, AttributionReport] | None) -> dict:
    """Stable ``--format json`` diff schema (FR-13); plain JSON types only."""
    attributions = attributions or {}
    return {
        "diff_id": diffres.diff_id,
        "snapshot_a": diffres.snapshot_a,
        "snapshot_b": diffres.snapshot_b,
        "dataset_id": diffres.dataset_id,
        "corpus_changed": diffres.corpus_changed,
        "parser_changed": diffres.parser_changed,
        "alignment_available": diffres.alignment_available,
        "config_diff": diffres.config_diff,
        "metric_deltas": diffres.metric_deltas,
        "queries": {
            qid: {
                "classification": qd.classification,
                "stability": qd.stability,
                "metric_delta": qd.metric_delta,
            }
            for qid, qd in diffres.queries.items()
        },
        "attributions": {
            qid: attribution_to_dict(rep) for qid, rep in attributions.items()
        },
    }
