from __future__ import annotations

from types import SimpleNamespace

import pytest

from recallops.adapters.local import LocalIndexAdapter
from recallops.dataset import generate
from recallops.diffing import diff
from recallops.evalrunner import evaluate
from recallops.gating import (
    GateNotCalibrated,
    calibrate,
    evaluate_gate,
    parse_fail_if,
)
from recallops.ingest import build_pipeline, collection_name, ingest
from recallops.models import CalibrationRecord, GoldenCase, GoldenDataset
from recallops.pipeline import chunkers
from recallops.store import ProjectStore

# 12 handcrafted cases over the example corpus: the first 7 lose their rank-1 doc
# under fixed_token(60,0) (hit@1 True -> False), the last 5 stay hit@1 in both
# snapshots. Verified deterministic: b=7, c=0 at k=1.
REG_CASES = [
    ("q1", "What does the documentation say about larger increases?", ["api/rate-limits.md"]),
    ("q2", "What does the documentation say about reach impact?", ["product/roadmap.md"]),
    ("q3", "What does the documentation say about rate limits?", ["api/rate-limits.md"]),
    ("q4", "What does the documentation say about payment method?", ["billing/refunds.md"]),
    ("q5", "How is incident runbook handled?", ["ops/incident-runbook.md"]),
    ("q6", "What does the documentation say about top ten?", ["product/roadmap.md"]),
    ("q7", "How is invoicing handled?", ["billing/invoices.md"]),
    ("q8", "What does the documentation say about single sign?", ["security/sso.md"]),
    ("q9", "What does the documentation say about row level?", ["eng/architecture.md"]),
    ("q10", "What does the documentation say about premises distribution?", ["product/roadmap.md"]),
    ("q11", "How is first week covers environment handled?", ["hr/onboarding.md"]),
    ("q12", "How is false alarm costs minutes handled?", ["ops/incident-runbook.md"]),
]


def _reg_dataset() -> GoldenDataset:
    return GoldenDataset("reg-v1", [
        GoldenCase(id=cid, question=q, expected_sources=list(src),
                   tags=[src[0].split("/")[0]])
        for cid, q, src in REG_CASES
    ])


@pytest.fixture(scope="module")
def nochange_env(tmp_path_factory, corpus_dir):
    root = tmp_path_factory.mktemp("gate_nochange")
    store = ProjectStore(root)
    adapter = LocalIndexAdapter(root / "idx")
    m = ingest(store, corpus_dir, build_pipeline({}), adapter).manifest
    ds = _reg_dataset()
    calibration = calibrate(store, m, ds, adapter, n_runs=3)
    eval_a = evaluate(store, m, ds)
    eval_b = evaluate(store, m, ds)
    diffres = diff(store, m, m, ds, eval_a, eval_b, epsilon=calibration.epsilon)
    return SimpleNamespace(store=store, adapter=adapter, manifest=m, dataset=ds,
                           calibration=calibration, diffres=diffres)


@pytest.fixture(scope="module")
def regression_env(tmp_path_factory, corpus_dir):
    root = tmp_path_factory.mktemp("gate_regress")
    store = ProjectStore(root)
    adapter = LocalIndexAdapter(root / "idx")
    mA = ingest(store, corpus_dir, build_pipeline({}), adapter).manifest
    mB = ingest(store, corpus_dir, build_pipeline({
        "chunker": {"tool": chunkers.FIXED_TOKEN, "params": {"max_tokens": 60, "overlap": 0}},
    }), None).manifest
    ds = _reg_dataset()
    calibration = calibrate(store, mA, ds, adapter, n_runs=3)
    eval_a = evaluate(store, mA, ds)
    eval_b = evaluate(store, mB, ds)
    diffres = diff(store, mA, mB, ds, eval_a, eval_b, epsilon=calibration.epsilon)
    return SimpleNamespace(store=store, manifest_a=mA, manifest_b=mB, dataset=ds,
                           calibration=calibration, diffres=diffres,
                           eval_a=eval_a, eval_b=eval_b)


@pytest.fixture(scope="module")
def ann_env(tmp_path_factory, corpus_dir):
    root = tmp_path_factory.mktemp("gate_ann")
    store = ProjectStore(root)
    adapter = LocalIndexAdapter(root / "idx", ann_mode=True, ann_sigma=0.02, seed=0)
    m = ingest(store, corpus_dir, build_pipeline({}), adapter).manifest
    ds = generate(store, m, n=40, seed=0)
    collection = collection_name(m)
    calibration = calibrate(store, m, ds, adapter, n_runs=3)
    adapter.rebuild(collection, seed=0)
    eval_a = evaluate(store, m, ds, adapter=adapter, mode="live")
    adapter.rebuild(collection, seed=1)
    eval_b = evaluate(store, m, ds, adapter=adapter, mode="live")
    diffres = diff(store, m, m, ds, eval_a, eval_b, epsilon=calibration.epsilon)
    diff0 = diff(store, m, m, ds, eval_a, eval_b, epsilon=0.0)
    return SimpleNamespace(store=store, manifest=m, dataset=ds, calibration=calibration,
                           diffres=diffres, diff0=diff0)


class TestCalibrate:
    def test_exact_adapter_zero_epsilon(self, nochange_env):
        cal = nochange_env.calibration
        assert cal.n_runs == 3
        assert cal.epsilon == 0.0

    def test_exact_adapter_zero_variance(self, nochange_env):
        cal = nochange_env.calibration
        assert set(cal.per_metric_std) >= {"recall@5", "hit_rate@5", "ndcg@10", "mrr"}
        assert all(v == 0.0 for v in cal.per_metric_std.values())
        assert all(v == 0.0 for v in cal.per_query_flip_rate.values())

    def test_snapshot_id_matches(self, nochange_env):
        assert nochange_env.calibration.snapshot_id == nochange_env.manifest.snapshot_id

    def test_persisted_and_roundtrips(self, nochange_env):
        stored = nochange_env.store.get_json("calibration", nochange_env.manifest.snapshot_id)
        assert stored == nochange_env.calibration.to_dict()
        assert CalibrationRecord.from_dict(stored).to_dict() == nochange_env.calibration.to_dict()

    def test_ann_positive_epsilon(self, ann_env):
        assert ann_env.calibration.epsilon > 0.0

    def test_ann_deterministic(self, ann_env):
        again = calibrate(ann_env.store, ann_env.manifest, ann_env.dataset,
                          LocalIndexAdapter(ann_env.store.root / "idx", ann_mode=True,
                                            ann_sigma=0.02, seed=0),
                          n_runs=3)
        assert again.epsilon == ann_env.calibration.epsilon
        assert again.per_metric_std == ann_env.calibration.per_metric_std

    def test_near_tie_k_follows_primary_metric(self, ann_env):
        # Finding #4: the near-tie boundary must track the gated metric's k. The
        # default calibration is at recall@5 (epsilon > 0); calibrating at
        # recall@10 derives the epsilon / flip-rate at the k=10 boundary, which is
        # a different signal, proving primary_metric is threaded through.
        cal10 = calibrate(ann_env.store, ann_env.manifest, ann_env.dataset,
                          LocalIndexAdapter(ann_env.store.root / "idx", ann_mode=True,
                                            ann_sigma=0.02, seed=0),
                          n_runs=3, primary_metric="recall@10")
        cal5 = ann_env.calibration
        assert cal5.epsilon > 0.0
        assert (cal10.epsilon, cal10.per_query_flip_rate) != (cal5.epsilon, cal5.per_query_flip_rate)
        # deterministic given the seed
        cal10b = calibrate(ann_env.store, ann_env.manifest, ann_env.dataset,
                           LocalIndexAdapter(ann_env.store.root / "idx", ann_mode=True,
                                             ann_sigma=0.02, seed=0),
                           n_runs=3, primary_metric="recall@10")
        assert cal10b.epsilon == cal10.epsilon


class TestStatisticalGateNoChange:
    def test_passes(self, nochange_env):
        gate = evaluate_gate(nochange_env.diffres, nochange_env.calibration,
                             dataset=nochange_env.dataset)
        assert gate.passed is True
        assert gate.mode == "statistical"

    def test_no_significant_regression(self, nochange_env):
        gate = evaluate_gate(nochange_env.diffres, nochange_env.calibration,
                             dataset=nochange_env.dataset)
        assert gate.significant_regression is False
        assert gate.details["b"] == 0
        assert gate.details["c"] == 0
        assert gate.details["p_overall"] == 1.0
        assert gate.details["near_tie_excluded"] == 0

    def test_zero_stable_regressed(self, nochange_env):
        assert nochange_env.diffres.by_class("regressed", stable_only=True) == []
        gate = evaluate_gate(nochange_env.diffres, nochange_env.calibration,
                             dataset=nochange_env.dataset)
        assert gate.unstable_query_ids == []


class TestStatisticalGateRegression:
    def test_fails(self, regression_env):
        gate = evaluate_gate(regression_env.diffres, regression_env.calibration,
                             primary_metric="recall@1", dataset=regression_env.dataset)
        assert gate.passed is False
        assert gate.reasons

    def test_significant_regression(self, regression_env):
        gate = evaluate_gate(regression_env.diffres, regression_env.calibration,
                             primary_metric="recall@1", dataset=regression_env.dataset)
        assert gate.significant_regression is True
        assert gate.details["ci"][1] < 0.0

    def test_mcnemar_flip_excess(self, regression_env):
        gate = evaluate_gate(regression_env.diffres, regression_env.calibration,
                             primary_metric="recall@1", dataset=regression_env.dataset)
        assert gate.details["b"] >= 6
        assert gate.details["c"] == 0
        assert gate.details["p_overall"] < 0.05

    def test_all_stable_no_near_ties(self, regression_env):
        gate = evaluate_gate(regression_env.diffres, regression_env.calibration,
                             primary_metric="recall@1", dataset=regression_env.dataset)
        assert gate.details["near_tie_excluded"] == 0
        assert gate.unstable_query_ids == []


class TestMcNemarNotDilutedByTags:
    """Finding #3: a genuine one-way flip excess (b=6, c=0) must fail the gate
    regardless of how many tag categories the dataset has. The overall McNemar is
    judged on its own q threshold; only the per-tag family is BH-corrected."""

    def _env(self):
        from recallops.models import DiffResult, QueryDiff, QueryEval

        def qe(recall5, hit5, target_rank):
            return QueryEval(query_id="", ranked_chunks=[("c", 1.0)], ranked_docs=["d"],
                             target_rank=target_rank, hit_at={5: hit5},
                             metrics={"recall@5": recall5}, run=None)

        queries: dict[str, QueryDiff] = {}
        cases = []
        # 6 hard regressors: hit@5 True -> False (b += 6), each its own tag.
        for i in range(6):
            qid = f"reg{i}"
            queries[qid] = QueryDiff(qid, "regressed", "stable", {"recall@5": -1.0},
                                     qe(1.0, True, 1), qe(0.0, False, None))
            cases.append(GoldenCase(qid, "q", ["d"], tags=[f"cat{i}"]))
        # 14 graded improvers: hit@5 stays True (no flip, c stays 0) but the graded
        # metric rises, pulling the aggregate CI above 0 so McNemar is the sole signal.
        for i in range(14):
            qid = f"imp{i}"
            queries[qid] = QueryDiff(qid, "improved", "stable", {"recall@5": 0.5},
                                     qe(0.5, True, 1), qe(1.0, True, 1))
            cases.append(GoldenCase(qid, "q", ["d"], tags=[f"cat{i}"]))
        diffres = DiffResult("d", "a", "b", "reg-v1", {}, False, {"recall@5": 1.0 / 20},
                             queries, {}, False, True)
        calibration = CalibrationRecord("a", 3, {"recall@5": 0.0}, 0.0, {}, "")
        dataset = GoldenDataset("reg-v1", cases)
        return diffres, calibration, dataset

    def test_overall_mcnemar_fires_despite_many_tags(self):
        diffres, calibration, dataset = self._env()
        with_tags = evaluate_gate(diffres, calibration, dataset=dataset)
        assert with_tags.details["b"] == 6 and with_tags.details["c"] == 0
        assert with_tags.details["p_overall"] < 0.05
        assert with_tags.significant_regression is False  # aggregate CI straddles/above 0
        assert with_tags.details["mcnemar_significant"] is True  # overall signal survives
        assert with_tags.passed is False

    def test_matches_no_dataset_path(self):
        diffres, calibration, dataset = self._env()
        without = evaluate_gate(diffres, calibration, dataset=None)
        assert without.details["mcnemar_significant"] is True
        assert without.passed is False


class TestAbsentTargetRegressionsNotExcluded:
    """Finding #2: a target that fell below the served list (target_rank=None)
    is a severe regression, not a near-tie, even with a large epsilon and a tiny
    k/k+1 boundary gap. It must stay 'stable' and remain in the McNemar count."""

    def test_absent_target_flips_stay_in_gate(self):
        from recallops.diffing import classify_query
        from recallops.models import DiffResult, QueryDiff, QueryEval

        # Served list has a tiny gap at the k/k+1 boundary; target absent after.
        chunks = [("c1", 1.0), ("c2", 0.9), ("c3", 0.8), ("c4", 0.7),
                  ("c5", 0.6), ("c6", 0.599)]

        def qe(recall5, hit5, ranked, target_rank):
            return QueryEval("", ranked, ["d"], target_rank, {5: hit5},
                             {"recall@5": recall5}, None)

        queries = {}
        cases = []
        for i in range(6):
            qid = f"reg{i}"
            before = qe(1.0, True, chunks, 1)
            after = qe(0.0, False, chunks, None)  # target dropped below served top-k
            _, stability = classify_query(before, after, epsilon=0.5)
            assert stability == "stable"  # not a near-tie despite the 0.001 k/k+1 gap
            queries[qid] = QueryDiff(qid, "regressed", stability, {"recall@5": -1.0}, before, after)
            cases.append(GoldenCase(qid, "q", ["d"], tags=["t"]))
        for i in range(14):
            qid = f"imp{i}"
            before = qe(0.5, True, chunks, 1)
            after = qe(1.0, True, chunks, 1)
            queries[qid] = QueryDiff(qid, "improved", "stable", {"recall@5": 0.5}, before, after)
            cases.append(GoldenCase(qid, "q", ["d"], tags=["t"]))
        diffres = DiffResult("d", "a", "b", "reg-v1", {}, False, {"recall@5": 0.05},
                             queries, {}, False, True)
        calibration = CalibrationRecord("a", 3, {"recall@5": 0.0}, 0.5, {}, "")  # epsilon 0.5
        gate = evaluate_gate(diffres, calibration, dataset=GoldenDataset("reg-v1", cases))
        assert gate.details["near_tie_excluded"] == 0
        assert gate.details["b"] == 6 and gate.details["c"] == 0
        assert gate.passed is False


class TestGateNotCalibrated:
    def test_statistical_requires_calibration(self, regression_env):
        with pytest.raises(GateNotCalibrated):
            evaluate_gate(regression_env.diffres, None, mode="statistical")


class TestAnnNeverFlaky:
    def test_gate_passes_with_calibrated_epsilon(self, ann_env):
        gate = evaluate_gate(ann_env.diffres, ann_env.calibration, dataset=ann_env.dataset)
        assert gate.passed is True
        assert gate.significant_regression is False

    def test_near_ties_flagged_and_excluded(self, ann_env):
        gate = evaluate_gate(ann_env.diffres, ann_env.calibration, dataset=ann_env.dataset)
        assert gate.details["near_tie_excluded"] > 0
        assert len(gate.unstable_query_ids) == gate.details["near_tie_excluded"]

    def test_uncalibrated_diff_flags_no_near_ties_but_still_passes(self, ann_env):
        gate0 = evaluate_gate(ann_env.diff0, ann_env.calibration, dataset=ann_env.dataset)
        assert gate0.details["near_tie_excluded"] == 0
        assert gate0.passed is True


class TestParseFailIf:
    @pytest.mark.parametrize("expr,expected", [
        ("recall@5<0.85", ("recall@5", "<", 0.85)),
        ("recall@5<=0.9", ("recall@5", "<=", 0.9)),
        ("mrr>0.5", ("mrr", ">", 0.5)),
        ("hit_rate@10>=0.75", ("hit_rate@10", ">=", 0.75)),
        (" recall@1 < 0.2 ", ("recall@1", "<", 0.2)),
    ])
    def test_valid(self, expr, expected):
        assert parse_fail_if(expr) == expected

    @pytest.mark.parametrize("expr", ["recall@5", "recall@5!0.5", "recall@5<abc", "", "<0.5"])
    def test_invalid(self, expr):
        with pytest.raises(ValueError):
            parse_fail_if(expr)


class TestRawGate:
    def test_raw_fail(self, regression_env):
        gate = evaluate_gate(regression_env.diffres, None, mode="raw",
                             fail_if="recall@1<0.85")
        assert gate.mode == "raw"
        assert gate.passed is False
        assert gate.significant_regression is True
        assert gate.details["after_value"] == pytest.approx(5 / 12)
        assert gate.reasons

    def test_raw_pass(self, regression_env):
        gate = evaluate_gate(regression_env.diffres, None, mode="raw",
                             fail_if="recall@1<0.2")
        assert gate.passed is True
        assert gate.significant_regression is False
        assert gate.reasons == []

    def test_fail_if_selects_raw_without_calibration(self, regression_env):
        gate = evaluate_gate(regression_env.diffres, None, fail_if="recall@1<0.2")
        assert gate.mode == "raw"
        assert gate.passed is True

    def test_raw_requires_expression(self, regression_env):
        with pytest.raises(ValueError):
            evaluate_gate(regression_env.diffres, None, mode="raw")
