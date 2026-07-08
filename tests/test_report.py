from __future__ import annotations

import json

from rich.console import Console

from recallops.models import (
    AttributionReport,
    ChunkFate,
    DiffResult,
    EvalResult,
    FunnelReport,
    GateResult,
    Hypothesis,
    QueryDiff,
    QueryEval,
    StageSpec,
    VerifiedCause,
)
from recallops.report import (
    attribution_to_dict,
    compare_report_markdown,
    diff_report_html,
    diff_summary_markdown,
    diff_table,
    eval_table,
    render_diff_json,
)

# ---- §10.4 example, built by hand ------------------------------------------

def _example_report() -> AttributionReport:
    funnel = FunnelReport(
        target_chunk_before="ch_5f2e",
        target_in_index_after=True,
        dense={"rank_before": 1, "rank_after": 9, "shadow_exact_rank_after": 9},
        sparse={"rank_before": 2, "rank_after": 14},
        fused={"rank_before": 1, "rank_after": 11},
        rerank={"in_candidates_after": False},
        ann_divergence=False,
    )
    return AttributionReport(
        query_id="q_014",
        classification="regressed",
        stability="stable",
        funnel=funnel,
        chunk_fate=ChunkFate("split", 0.92, "ch_5f2e", ["ch_a01", "ch_a02"]),
        verified_causes=[VerifiedCause("chunker", "arm_B_with_chunker_A", 1, "verified")],
        hypotheses=[Hypothesis("bm25_weight", "unverified", "sparse rank drop co-occurred")],
        narrative=(
            "Target section was split across two chunks by the fixed-token chunker; "
            "each fragment scores lower on both dense and BM25 (heading text no longer "
            "in-chunk). Reverting the chunker alone restores rank 1 "
            "(verified: arm_B_with_chunker_A)."
        ),
    )


def _example_dict() -> dict:
    return {
        "query_id": "q_014",
        "classification": "regressed",
        "stability": "stable",
        "funnel": {
            "target_chunk_before": "ch_5f2e",
            "target_in_index_after": True,
            "dense": {"rank_before": 1, "rank_after": 9, "shadow_exact_rank_after": 9},
            "sparse": {"rank_before": 2, "rank_after": 14},
            "fused": {"rank_before": 1, "rank_after": 11},
            "rerank": {"in_candidates_after": False},
            "ann_divergence": False,
        },
        "chunk_fate": {
            "class": "split",
            "alignment_score": 0.92,
            "old_chunk": "ch_5f2e",
            "new_chunks": ["ch_a01", "ch_a02"],
        },
        "verified_causes": [
            {"factor": "chunker", "arm_id": "arm_B_with_chunker_A",
             "recovered_rank": 1, "status": "verified"}
        ],
        "hypotheses": [
            {"factor": "bm25_weight", "status": "unverified",
             "evidence": "sparse rank drop co-occurred"}
        ],
        "narrative": (
            "Target section was split across two chunks by the fixed-token chunker; "
            "each fragment scores lower on both dense and BM25 (heading text no longer "
            "in-chunk). Reverting the chunker alone restores rank 1 "
            "(verified: arm_B_with_chunker_A)."
        ),
    }


def test_attribution_to_dict_matches_prd_10_4_key_for_key():
    rep = _example_report()
    assert attribution_to_dict(rep) == _example_dict()


def test_attribution_to_dict_round_trips_through_json():
    rep = _example_report()
    d = attribution_to_dict(rep)
    assert json.loads(json.dumps(d)) == d


# ---- Diff / gate fixtures ---------------------------------------------------

def _qe(qid: str, docs: list[str], target_rank, recall5: float) -> QueryEval:
    ranked = [(f"ch_{qid}_{i}", 1.0 - 0.1 * i) for i in range(len(docs))]
    return QueryEval(
        query_id=qid,
        ranked_chunks=ranked,
        ranked_docs=docs,
        target_rank=target_rank,
        hit_at={1: recall5 >= 1.0, 5: recall5 > 0.0},
        metrics={"recall@1": recall5 if target_rank == 1 else 0.0,
                 "recall@5": recall5, "mrr": 0.5, "ndcg@5": recall5},
        run=None,
    )


def _diffres() -> DiffResult:
    before = {
        "q1": _qe("q1", ["billing/refunds.md"], 1, 1.0),
        "q2": _qe("q2", ["api/auth.md"], 2, 1.0),
        "q3": _qe("q3", ["security/sso.md"], 1, 1.0),
    }
    after = {
        "q1": _qe("q1", ["support/tickets.md", "billing/refunds.md"], 8, 0.0),
        "q2": _qe("q2", ["api/auth.md"], 1, 1.0),
        "q3": _qe("q3", ["security/sso.md"], 1, 1.0),
    }
    queries = {
        "q1": QueryDiff("q1", "regressed", "stable",
                        {"recall@5": -1.0, "recall@1": -1.0, "mrr": -0.4},
                        before["q1"], after["q1"]),
        "q2": QueryDiff("q2", "improved", "stable",
                        {"recall@5": 0.0, "recall@1": 1.0, "mrr": 0.5},
                        before["q2"], after["q2"]),
        "q3": QueryDiff("q3", "unchanged", "unstable",
                        {"recall@5": 0.0, "recall@1": 0.0, "mrr": 0.0},
                        before["q3"], after["q3"]),
    }
    chunk_before = StageSpec("chunk", "recall.chunkers.markdown_heading", "1",
                             {"max_tokens": 800, "overlap": 120})
    chunk_after = StageSpec("chunk", "recall.chunkers.fixed_token", "1",
                            {"max_tokens": 60, "overlap": 0})
    config_diff = {
        "chunk": {"before": chunk_before.to_dict(), "after": chunk_after.to_dict()},
        "corpus": {"before": {"merkle_root": "mr_aaa"}, "after": {"merkle_root": "mr_bbb"}},
    }
    return DiffResult(
        diff_id="diff_test1",
        snapshot_a="snap_a",
        snapshot_b="snap_b",
        dataset_id="golden-v1",
        config_diff=config_diff,
        corpus_changed=True,
        metric_deltas={"recall@5": -0.3333, "recall@1": -0.3333, "mrr": -0.1},
        queries=queries,
        alignment={"ch_5f2e": ChunkFate("split", 0.92, "ch_5f2e", ["ch_a01", "ch_a02"])},
        parser_changed=False,
        alignment_available=True,
    )


def _gate() -> GateResult:
    return GateResult(
        passed=False,
        mode="statistical",
        reasons=["aggregate recall@5 regressed: 95% CI upper bound -0.0600 < 0"],
        significant_regression=True,
        unstable_query_ids=["q3"],
        details={
            "primary_metric": "recall@5",
            "ci": [-0.2000, -0.0600],
            "b": 1,
            "c": 0,
            "p_overall": 0.5,
            "n_stable": 2,
            "n_unstable": 1,
        },
    )


def _attributions() -> dict[str, AttributionReport]:
    rep = _example_report()
    rep.query_id = "q1"
    return {"q1": rep}


def test_diff_summary_markdown_has_ci_numbers_and_details_folds():
    md = diff_summary_markdown(_diffres(), _attributions(), _gate(), calibration_ok=True)
    assert "[-0.2000, -0.0600]" in md
    assert "<details>" in md
    assert "</details>" in md
    assert "Calibration: present" in md
    assert "1 regressed" in md
    assert "chunker" in md  # top verified cause


def test_diff_summary_ci_column_labeled_stable_only():
    # Fix 4 makes the CI stable-only while the Δ column is over ALL queries.
    # The table must say so, or a reader pairs an all-query Δ with a
    # stable-only CI and reads a self-contradictory row.
    md = diff_summary_markdown(_diffres(), _attributions(), _gate(), calibration_ok=True)
    header = next(line for line in md.splitlines() if "95% CI" in line and "|" in line)
    assert "stable" in header.lower()


def test_diff_summary_markdown_without_gate_reports_calibration_absent():
    md = diff_summary_markdown(_diffres(), None, None, calibration_ok=False)
    assert "Calibration: not present" in md
    assert "<details>" in md


def test_headline_excludes_unstable_regressed_from_count():
    # Finding #8: a near-tie (unstable) regressed query must not be counted in the
    # headline "regressed" tally that the same line reports as "(excluded)".
    before = {"q1": _qe("q1", ["a.md"], 1, 1.0)}
    after = {"q1": _qe("q1", ["a.md"], 8, 0.0)}
    queries = {"q1": QueryDiff("q1", "regressed", "unstable",
                               {"recall@5": -1.0}, before["q1"], after["q1"])}
    dr = DiffResult("d", "a", "b", "golden-v1", {}, False,
                    {"recall@5": -1.0}, queries, {}, False, True)
    md = diff_summary_markdown(dr, None, None, calibration_ok=True)
    assert "0 regressed" in md
    assert "1 unstable (excluded)" in md


def test_diff_report_html_is_self_contained():
    html_out = diff_report_html(_diffres(), _attributions(), _gate())
    assert "http" not in html_out
    assert "<style>" in html_out
    assert "<!DOCTYPE html>" in html_out
    assert "q1" in html_out
    assert "regressed" in html_out
    assert "arm_B_with_chunker_A" in html_out


def test_eval_table_lists_every_aggregate_metric():
    ev = EvalResult(
        run_id="ev_1", snapshot_id="snap_a", dataset_id="golden-v1",
        mode="replay", adapter="none", created_at="", k_values=(1, 5),
        per_query={}, aggregate={"recall@1": 0.8, "recall@5": 0.9, "mrr": 0.75},
    )
    table = eval_table(ev)
    assert table.row_count == 3
    console = Console(record=True, width=100)
    console.print(table)
    text = console.export_text()
    assert "recall@5" in text
    assert "0.9000" in text


def test_diff_table_lists_every_query():
    diffres = _diffres()
    table = diff_table(diffres)
    assert table.row_count == len(diffres.queries)
    console = Console(record=True, width=120)
    console.print(table)
    text = console.export_text()
    assert "q1" in text
    assert "regressed" in text


def test_compare_report_markdown_has_recommendation_and_cost():
    tag_deltas = {
        "exact-term": {"recall@5": -0.07, "mrr": -0.05},
        "paraphrase": {"recall@5": 0.11, "mrr": 0.09},
    }
    overall = {"recall@5": 0.02, "mrr": 0.02}
    recommendation = {"action": "adopt hybrid", "hybrid_weight": 0.35}
    cost = {"usd": 14.20, "wall_s": 7560.0, "embed_calls": 4200}
    md = compare_report_markdown(tag_deltas, overall, recommendation, cost)
    assert "adopt hybrid" in md
    assert "0.35" in md
    assert "$14.20" in md
    assert "exact-term" in md
    assert "paraphrase" in md
    assert "overall" in md
    assert "embed_calls" in md


def test_render_diff_json_round_trips():
    diffres = _diffres()
    payload = render_diff_json(diffres, _attributions())
    assert json.loads(json.dumps(payload)) == payload
    assert payload["diff_id"] == "diff_test1"
    assert payload["corpus_changed"] is True
    assert set(payload["queries"]) == {"q1", "q2", "q3"}
    assert payload["attributions"]["q1"]["verified_causes"][0]["factor"] == "chunker"


def test_render_diff_json_handles_no_attributions():
    payload = render_diff_json(_diffres(), None)
    assert payload["attributions"] == {}
    assert json.loads(json.dumps(payload)) == payload
