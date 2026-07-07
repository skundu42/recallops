"""The Phase-0 go/no-go, encoded as a test (PRD §13).

``run_scorecard`` builds four known-cause scenarios on the example corpus and
measures the five attribution-engine acceptance metrics. ``test_scorecard_gates``
asserts every §13 gate passes; the other tests pin the scenario shape so a
regression in any engine layer surfaces as a specific failure rather than a bare
gate miss.
"""
from __future__ import annotations

import pytest

from recallops.scorecard import GATES, ScorecardResult, run_scorecard


@pytest.fixture(scope="module")
def result(tmp_path_factory) -> ScorecardResult:
    return run_scorecard(tmp_path_factory.mktemp("scorecard"))


# -- the go/no-go -------------------------------------------------------------


def test_scorecard_gates(result: ScorecardResult):
    """All five §13 gates pass, the Phase-0 go/no-go for verified attribution."""
    assert result.coverage >= GATES["coverage"], f"coverage {result.coverage}"
    assert result.fidelity >= GATES["fidelity"], f"fidelity {result.fidelity}"
    assert result.noise_floor <= GATES["noise_floor"], f"noise_floor {result.noise_floor}"
    assert result.stage_accuracy >= GATES["stage_accuracy"], f"stage_accuracy {result.stage_accuracy}"
    assert result.narrative_violations == GATES["narrative_violations"]
    assert result.passes_gates() is True


# -- per-metric detail --------------------------------------------------------


def test_coverage_every_stable_regression_verified(result: ScorecardResult):
    totals = result.details["totals"]
    assert totals["regressed"] > 0
    assert totals["covered"] == totals["regressed"]
    assert result.coverage == 1.0


def test_fidelity_is_one_by_construction(result: ScorecardResult):
    assert result.fidelity == 1.0
    assert all(s["fidelity"] == 1.0 for s in result.details["scenarios"].values())


def test_stage_accuracy_names_true_factor(result: ScorecardResult):
    totals = result.details["totals"]
    assert totals["stage_ok"] == totals["regressed"]
    assert result.stage_accuracy == 1.0


def test_noise_floor_under_gate(result: ScorecardResult):
    noise = result.details["noise"]
    assert noise["epsilon"] > 0.0  # ANN noise really did produce near-ties to exclude
    assert result.noise_floor <= GATES["noise_floor"]
    assert noise["mean_rate"] <= GATES["noise_floor"]


def test_no_narrative_violations(result: ScorecardResult):
    assert result.narrative_violations == 0


# -- scenario shape -----------------------------------------------------------


def test_all_four_scenarios_regress(result: ScorecardResult):
    scenarios = result.details["scenarios"]
    assert set(scenarios) == {"s1_chunk", "s2_retrieve", "s3_embed", "s4_corpus"}
    for name, s in scenarios.items():
        assert s["regressed"] >= 1, f"{name} produced no stable regressions"


def test_scenario_true_factors(result: ScorecardResult):
    scenarios = result.details["scenarios"]
    assert scenarios["s1_chunk"]["true_factor"] == "chunk"
    assert scenarios["s2_retrieve"]["true_factor"] == "retrieve"
    assert scenarios["s3_embed"]["true_factor"] == "embed"
    assert scenarios["s4_corpus"]["true_factor"] == "corpus"


def test_single_factor_scenarios_isolate_the_change(result: ScorecardResult):
    scenarios = result.details["scenarios"]
    assert scenarios["s1_chunk"]["config_diff"] == ["chunk"]
    assert scenarios["s2_retrieve"]["config_diff"] == ["retrieve"]
    assert scenarios["s3_embed"]["config_diff"] == ["embed"]
    # S4 changes the corpus AND a no-op config factor together (FR-7.6 / FR-11.1)
    assert scenarios["s4_corpus"]["config_diff"] == ["corpus", "index"]


def test_verified_factor_matches_true_factor(result: ScorecardResult):
    for name, s in result.details["scenarios"].items():
        for qid, pq in s["per_query"].items():
            if pq["verified"]:
                assert s["true_factor"] in pq["verified_factors"], f"{name}/{qid}"


# -- determinism & serialization ----------------------------------------------


def test_deterministic(tmp_path):
    a = run_scorecard(tmp_path / "a")
    b = run_scorecard(tmp_path / "b")
    assert a.coverage == b.coverage
    assert a.fidelity == b.fidelity
    assert a.noise_floor == b.noise_floor
    assert a.stage_accuracy == b.stage_accuracy
    assert a.narrative_violations == b.narrative_violations


def test_result_to_dict_shape(result: ScorecardResult):
    d = result.to_dict()
    assert set(d) >= {"coverage", "fidelity", "noise_floor", "stage_accuracy",
                      "narrative_violations", "details"}
