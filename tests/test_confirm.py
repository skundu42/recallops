from __future__ import annotations

from pathlib import Path

import pytest

from recallops.ablation import build_arms, enumerate_factors, run_arms
from recallops.confirm import confirm_causes, fidelity_check
from recallops.diffing import diff
from recallops.evalrunner import evaluate
from recallops.funnel import funnel_for_query
from recallops.ingest import build_pipeline, ingest
from recallops.models import (
    Arm,
    ArmResult,
    DiffResult,
    EvalResult,
    Factor,
    FunnelReport,
    GoldenCase,
    GoldenDataset,
    QueryDiff,
    QueryEval,
)
from recallops.pipeline import chunkers
from recallops.retrieval import RetrievalEngine
from recallops.store import ProjectStore

CORPUS = Path(__file__).resolve().parent.parent / "examples" / "corpus"

CASES = [
    ("q1", "How are annual plan refunds prorated after the 30-day window closes?",
     ["billing/refunds.md"]),
    ("q2", "Which single sign-on protocols like SAML and OIDC are supported?",
     ["security/sso.md"]),
    ("q3", "What happens when an engineer declares an incident with the incident bot?",
     ["ops/incident-runbook.md"]),
    ("q4", "When is production access granted to new engineers during onboarding?",
     ["hr/onboarding.md"]),
    ("q5", "How long is the key rotation overlap window for API keys by default?",
     ["api/auth.md"]),
    ("q6", "How do refunds relate to invoices and payment methods?",
     ["billing/refunds.md", "billing/invoices.md"]),
    ("q7", "How do pricing plans and the product roadmap cover SSO?",
     ["sales/pricing.md", "product/roadmap.md", "security/sso.md"]),
    ("q8", "What notice periods apply to price changes and subprocessor onboarding?",
     ["sales/pricing.md", "legal/dpa.md"]),
]


def _hand_dataset() -> GoldenDataset:
    return GoldenDataset("hand-v1", [
        GoldenCase(id=c, question=q, expected_sources=list(s), tags=[]) for c, q, s in CASES])


def _funnels(store, man_a, man_b, diffres, dataset) -> dict[str, FunnelReport]:
    engine_a = RetrievalEngine(store, man_a)
    engine_b = RetrievalEngine(store, man_b)
    out: dict[str, FunnelReport] = {}
    for qdiff in diffres.by_class("regressed", stable_only=True):
        case = dataset.case(qdiff.query_id)
        out[qdiff.query_id] = funnel_for_query(engine_a, engine_b, qdiff, case, diffres.alignment)
    return out


# ------------------------------------------------- unit tests on the rule ----

def _qe(target_rank: int | None) -> QueryEval:
    return QueryEval(query_id="q", ranked_chunks=[], ranked_docs=[],
                     target_rank=target_rank, hit_at={}, metrics={}, run=None)


def _eval_with(target_rank: int | None) -> EvalResult:
    return EvalResult(run_id="r", snapshot_id="s", dataset_id="ds", mode="replay",
                      adapter="none", created_at="", k_values=(1, 5, 10),
                      per_query={"q": _qe(target_rank)}, aggregate={})


def _hand_diff() -> DiffResult:
    qd = QueryDiff(query_id="q", classification="regressed", stability="stable",
                   metric_delta={}, before=_qe(3), after=_qe(9))
    return DiffResult(
        diff_id="d", snapshot_a="a", snapshot_b="b", dataset_id="ds",
        config_diff={"chunk": {}, "embed": {}}, corpus_changed=False, metric_deltas={},
        queries={"q": qd}, alignment={}, parser_changed=False, alignment_available=True,
    )


def _hand_funnel() -> FunnelReport:
    return FunnelReport(
        target_chunk_before="ch_tracer", target_in_index_after=True,
        dense={"rank_before": 3, "rank_after": 9, "shadow_exact_rank_after": 9},
        sparse={"rank_before": 3, "rank_after": 3},
        fused={"rank_before": 3, "rank_after": 9},
        rerank={"in_candidates_after": True}, ann_divergence=False,
    )


def _hand_setup(chunk_rank, embed_rank):
    arms = build_arms([Factor("chunk", "stage"), Factor("embed", "stage")])
    chunk_arm = Arm.build({"chunk": "A", "embed": "B"}).arm_id
    embed_arm = Arm.build({"chunk": "B", "embed": "A"}).arm_id
    arm_results = {
        chunk_arm: ArmResult(chunk_arm, _eval_with(chunk_rank), 0),
        embed_arm: ArmResult(embed_arm, _eval_with(embed_rank), 0),
    }
    return arms, arm_results, chunk_arm, embed_arm


class TestConfirmationRuleUnit:
    def test_recovered_factor_verified_other_is_hypothesis(self):
        arms, arm_results, chunk_arm, _ = _hand_setup(chunk_rank=3, embed_rank=8)
        reports = confirm_causes(_hand_diff(), _hand_dataset(), arm_results, arms,
                                 {"q": _hand_funnel()}, {}, recovery_threshold="rank1", top_k=5)
        rep = reports["q"]
        assert [(c.factor, c.arm_id) for c in rep.verified_causes] == [("chunk", chunk_arm)]
        assert rep.verified_causes[0].recovered_rank == 3
        assert {h.factor for h in rep.hypotheses} == {"embed"}
        assert all(h.status == "unverified" and h.evidence for h in rep.hypotheses)

    def test_factor_never_both_verified_and_hypothesis(self):
        arms, arm_results, _, _ = _hand_setup(chunk_rank=3, embed_rank=8)
        reports = confirm_causes(_hand_diff(), _hand_dataset(), arm_results, arms,
                                 {"q": _hand_funnel()}, {})
        rep = reports["q"]
        verified = {c.factor for c in rep.verified_causes}
        hypotheses = {h.factor for h in rep.hypotheses}
        assert verified.isdisjoint(hypotheses)

    def test_rank1_threshold_rejects_beyond_before_plus_one(self):
        # before_rank=3 -> rank1 bound is 4; a recovered_rank of 5 must NOT verify
        arms, arm_results, _, _ = _hand_setup(chunk_rank=5, embed_rank=9)
        reports = confirm_causes(_hand_diff(), _hand_dataset(), arm_results, arms,
                                 {"q": _hand_funnel()}, {}, recovery_threshold="rank1", top_k=5)
        assert {c.factor for c in reports["q"].verified_causes} == set()

    def test_topk_threshold_accepts_within_top_k(self):
        # recovered_rank=5 fails rank1 (bound 4) but passes topk (top_k=5)
        arms, arm_results, chunk_arm, _ = _hand_setup(chunk_rank=5, embed_rank=9)
        reports = confirm_causes(_hand_diff(), _hand_dataset(), arm_results, arms,
                                 {"q": _hand_funnel()}, {}, recovery_threshold="topk", top_k=5)
        assert {c.factor for c in reports["q"].verified_causes} == {"chunk"}

    def test_only_regressed_stable_queries_reported(self):
        arms, arm_results, _, _ = _hand_setup(chunk_rank=3, embed_rank=8)
        diffres = _hand_diff()
        diffres.queries["q"].classification = "improved"
        reports = confirm_causes(diffres, _hand_dataset(), arm_results, arms,
                                 {"q": _hand_funnel()}, {})
        assert reports == {}

    def test_unknown_recovery_threshold_raises(self):
        arms, arm_results, _, _ = _hand_setup(chunk_rank=3, embed_rank=8)
        with pytest.raises(ValueError):
            confirm_causes(_hand_diff(), _hand_dataset(), arm_results, arms,
                           {"q": _hand_funnel()}, {}, recovery_threshold="bogus")

    def test_fidelity_check_is_one_for_genuine_causes(self):
        arms, arm_results, _, _ = _hand_setup(chunk_rank=3, embed_rank=8)
        reports = confirm_causes(_hand_diff(), _hand_dataset(), arm_results, arms,
                                 {"q": _hand_funnel()}, {})
        assert fidelity_check(reports, arm_results, arms, _hand_dataset(), 5) == 1.0

    def test_no_op_factor_not_verified_when_metric_not_restored(self):
        # Multi-source regression (finding #1/#5): the first expected source stays
        # rank 1 in every arm, so target_rank alone cannot tell recovery apart.
        # Only the arm that RESTORES recall@5 may be verified; a no-op factor whose
        # revert leaves recall@5 depressed must stay an unverified hypothesis.
        def qe(target_rank, recall5):
            return QueryEval(query_id="q", ranked_chunks=[], ranked_docs=[],
                             target_rank=target_rank, hit_at={},
                             metrics={"recall@5": recall5}, run=None)

        def eval_with(target_rank, recall5):
            return EvalResult(run_id="r", snapshot_id="s", dataset_id="ds", mode="replay",
                              adapter="none", created_at="", k_values=(1, 5, 10),
                              per_query={"q": qe(target_rank, recall5)}, aggregate={})

        arms = build_arms([Factor("corpus", "corpus"), Factor("index", "stage")])
        corpus_arm = Arm.build({"corpus": "A", "index": "B"}).arm_id  # genuine cause
        index_arm = Arm.build({"corpus": "B", "index": "A"}).arm_id   # replay no-op
        arm_results = {
            corpus_arm: ArmResult(corpus_arm, eval_with(1, 1.0), 0),   # recall@5 restored
            index_arm: ArmResult(index_arm, eval_with(1, 0.5), 0),     # recall@5 still depressed
        }
        qd = QueryDiff(query_id="q", classification="regressed", stability="stable",
                       metric_delta={"recall@5": -0.5},
                       before=qe(1, 1.0), after=qe(1, 0.5))
        diffres = DiffResult(
            diff_id="d", snapshot_a="a", snapshot_b="b", dataset_id="ds",
            config_diff={"index": {}, "corpus": {}}, corpus_changed=True, metric_deltas={},
            queries={"q": qd}, alignment={}, parser_changed=False, alignment_available=True,
        )
        reports = confirm_causes(diffres, _hand_dataset(), arm_results, arms,
                                 {"q": _hand_funnel()}, {}, recovery_threshold="topk", top_k=5)
        verified = {c.factor for c in reports["q"].verified_causes}
        assert verified == {"corpus"}
        assert "index" not in verified
        # fidelity_check with diffres must not be fooled if a false cause were emitted.
        assert fidelity_check(reports, arm_results, arms, _hand_dataset(), 5, diffres=diffres) == 1.0


# ------------------------------------------ J1 chunker acceptance scenario ---

@pytest.fixture(scope="module")
def chunker_scenario(tmp_path_factory):
    root = tmp_path_factory.mktemp("confirm_chunk")
    store = ProjectStore(root)
    man_a = ingest(store, CORPUS, build_pipeline({}), None).manifest
    man_b = ingest(store, CORPUS, build_pipeline({
        "chunker": {"tool": chunkers.FIXED_TOKEN, "params": {"max_tokens": 60, "overlap": 0}},
    }), None).manifest
    ds = _hand_dataset()
    d = diff(store, man_a, man_b, ds, evaluate(store, man_a, ds), evaluate(store, man_b, ds))
    arms = build_arms(enumerate_factors(d))
    arm_results = run_arms(store, arms, man_a, man_b, CORPUS, ds, "confirm-chunk-job")
    funnels = _funnels(store, man_a, man_b, d, ds)
    reports = confirm_causes(d, ds, arm_results, arms, funnels, d.alignment,
                             recovery_threshold="rank1", top_k=5)
    return {"diff": d, "arms": arms, "arm_results": arm_results, "dataset": ds, "reports": reports}


class TestAcceptanceConfirmationRule:
    """J1: A markdown_heading defaults -> B fixed_token(60,0). The 12-doc corpus
    produces genuine stable recall@5 regressions on the multi-source queries
    (q6/q7/q8); reverting the only changed factor (chunk) recovers them."""

    def test_regressed_queries_exist(self, chunker_scenario):
        assert enumerate_factors(chunker_scenario["diff"]) == [Factor("chunk", "stage")]
        assert chunker_scenario["reports"], "expected at least one stable regressed query"

    def test_chunk_is_the_verified_cause(self, chunker_scenario):
        for rep in chunker_scenario["reports"].values():
            assert {c.factor for c in rep.verified_causes} == {"chunk"}

    def test_untouched_factor_never_verified(self, chunker_scenario):
        verified = {
            c.factor
            for rep in chunker_scenario["reports"].values()
            for c in rep.verified_causes
        }
        assert verified == {"chunk"}
        assert "embed" not in verified and "retrieve" not in verified and "parse" not in verified

    def test_verified_cause_cites_the_revert_arm(self, chunker_scenario):
        revert_arm = Arm.build({"chunk": "A"}).arm_id
        for rep in chunker_scenario["reports"].values():
            for cause in rep.verified_causes:
                assert cause.arm_id == revert_arm
                assert cause.status == "verified"

    def test_fidelity_is_one(self, chunker_scenario):
        assert fidelity_check(chunker_scenario["reports"], chunker_scenario["arm_results"],
                              chunker_scenario["arms"], chunker_scenario["dataset"], 5) == 1.0


# ---------------------------------------- corpus-drift control acceptance ----

_P = "quokka ledger epoch synchronization protocol relay"
_QUESTION = "How does the quokka ledger epoch synchronization protocol relay coordinate state?"
_TARGET_PATH = "sync/target.md"

_FILLER = {
    "misc/weather.md": "# Weather Notes\n\nThe afternoon forecast predicts scattered showers and mild "
    "wind across the coastal plain. Gardeners should cover seedlings before dusk.\n",
    "misc/recipes.md": "# Kitchen Log\n\nThe sourdough starter needs feeding twice daily with equal "
    "parts flour and water. Bake at high heat with steam for a crisp crust.\n",
    "misc/travel.md": "# Travel Diary\n\nThe mountain trail climbs past alpine meadows toward a quiet "
    "lake. Bring layers because temperatures drop quickly after sunset.\n",
    "misc/music.md": "# Practice Journal\n\nScales and arpeggios warm up the fingers before tackling the "
    "sonata. Metronome discipline keeps the tempo honest through hard passages.\n",
    "misc/garden.md": "# Garden Plan\n\nRotate tomatoes and beans between raised beds each season to keep "
    "the soil balanced. Compost the trimmings and mulch heavily in autumn.\n",
}
_TARGET = (
    "# Ledger Synchronization Guide\n\n"
    f"The {_P} coordinates distributed ledger state across regional nodes.\n\n"
    "## Operational Detail\n\n"
    "Operators configure the regional coordinator with a rotation calendar, a quorum threshold, "
    "and a fallback contact roster. The runbook documents escalation windows, maintenance freezes, "
    "and the archival policy for superseded snapshots. Audit trails record every operator override "
    "with a signed justification and a rollback checkpoint for later review.\n"
)
_BASE = {
    "sync/mirror-one.md": f"# Mirror One\n\n{_P}. {_P} confirmed.\n",
    "sync/mirror-two.md": f"# Mirror Two\n\n{_P}. {_P} verified.\n",
}
_DISTRACTORS = {
    "sync/dup-alpha.md": f"# Dup Alpha\n\n{_P} {_P}. {_P}.\n",
    "sync/dup-beta.md": f"# Dup Beta\n\n{_P} {_P}. {_P} now.\n",
    "sync/dup-gamma.md": f"# Dup Gamma\n\n{_P} {_P}. {_P} here.\n",
}
_DENSE_ONLY = {"retrieve": {"top_k": 10, "hybrid": None}}


def _write_corpus(root: Path, docs: dict) -> None:
    for rel, text in docs.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)


@pytest.fixture(scope="module")
def drift_scenario(tmp_path_factory):
    root = tmp_path_factory.mktemp("confirm_drift")
    dir_a = root / "a"
    dir_b = root / "b"
    _write_corpus(dir_a, {**_FILLER, **_BASE, _TARGET_PATH: _TARGET})
    _write_corpus(dir_b, {**_FILLER, **_BASE, _TARGET_PATH: _TARGET, **_DISTRACTORS})

    store = ProjectStore(root / "store")
    man_a = ingest(store, dir_a, build_pipeline(_DENSE_ONLY), None).manifest
    # B adds 3 near-duplicate distractor docs AND a no-op index config touch, so
    # both the corpus and a config factor changed together (FR-7.6 / FR-11.1).
    man_b = ingest(store, dir_b, build_pipeline({
        **_DENSE_ONLY, "index": {"adapter": "local", "collection": "b-index"},
    }), None).manifest

    ds = GoldenDataset("drift-v1", [
        GoldenCase(id="qd", question=_QUESTION, expected_sources=[_TARGET_PATH], tags=[])])
    d = diff(store, man_a, man_b, ds, evaluate(store, man_a, ds), evaluate(store, man_b, ds))
    arms = build_arms(enumerate_factors(d))
    arm_results = run_arms(store, arms, man_a, man_b, dir_b, ds, "confirm-drift-job")
    funnels = _funnels(store, man_a, man_b, d, ds)
    reports = confirm_causes(d, ds, arm_results, arms, funnels, d.alignment,
                             recovery_threshold="rank1", top_k=5)
    return {"store": store, "man_a": man_a, "man_b": man_b, "diff": d,
            "arms": arms, "arm_results": arm_results, "dataset": ds, "reports": reports}


class TestAcceptanceCorpusDriftControl:
    """Distractor docs added alongside a no-op config change: only the corpus
    revert restores the target, so corpus is the verified cause and the config
    factor stays an unverified hypothesis (FR-7.6 / FR-11.1)."""

    def test_both_corpus_and_config_factor_changed(self, drift_scenario):
        d = drift_scenario["diff"]
        assert d.corpus_changed is True
        assert enumerate_factors(d) == [Factor("index", "stage"), Factor("corpus", "corpus")]

    def test_query_regressed_out_of_top_k(self, drift_scenario):
        qd = drift_scenario["diff"].queries["qd"]
        assert qd.classification == "regressed"
        assert qd.stability == "stable"
        assert qd.before.target_rank <= 5
        assert qd.after.target_rank > 5

    def test_corpus_is_the_verified_cause(self, drift_scenario):
        rep = drift_scenario["reports"]["qd"]
        assert {c.factor for c in rep.verified_causes} == {"corpus"}

    def test_untouched_config_factor_not_verified(self, drift_scenario):
        rep = drift_scenario["reports"]["qd"]
        verified = {c.factor for c in rep.verified_causes}
        assert "index" not in verified

    def test_implicated_config_factor_is_a_hypothesis(self, drift_scenario):
        rep = drift_scenario["reports"]["qd"]
        hyp = {h.factor for h in rep.hypotheses}
        assert "index" in hyp
        assert "corpus" not in hyp  # verified, so never also a hypothesis

    def test_corpus_revert_arm_recovers_within_original_rank(self, drift_scenario):
        rep = drift_scenario["reports"]["qd"]
        before_rank = drift_scenario["diff"].queries["qd"].before.target_rank
        cause = next(c for c in rep.verified_causes if c.factor == "corpus")
        assert cause.recovered_rank <= before_rank + 1

    def test_fidelity_is_one(self, drift_scenario):
        assert fidelity_check(drift_scenario["reports"], drift_scenario["arm_results"],
                              drift_scenario["arms"], drift_scenario["dataset"], 5) == 1.0

    def test_arms_share_cache_zero_new_embeds_for_reverts(self, drift_scenario):
        # every arm reuses chunks/embeddings already produced by ingest A and B
        results = drift_scenario["arm_results"]
        assert all(r.embed_calls == 0 for r in results.values())
