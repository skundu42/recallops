from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import recallops.ablation as ablation
from recallops.ablation import (
    ArmPlan,
    _materialize_arm,
    build_arms,
    enumerate_factors,
    materialize_arm,
    plan_arms,
    run_arms,
    shapley,
)
from recallops.diffing import diff
from recallops.evalrunner import evaluate
from recallops.ingest import build_pipeline, ingest
from recallops.models import Arm, DiffResult, Factor, GoldenCase, GoldenDataset
from recallops.pipeline import chunkers
from recallops.pipeline.providers import get_provider
from recallops.store import ProjectStore


def _dataset() -> GoldenDataset:
    return GoldenDataset("hand-v1", [
        GoldenCase(id="q1", question="How are annual plan refunds prorated after 30 days?",
                   expected_sources=["billing/refunds.md"], tags=[]),
        GoldenCase(id="q2", question="Which single sign-on protocols like SAML and OIDC are supported?",
                   expected_sources=["security/sso.md"], tags=[]),
        GoldenCase(id="q3", question="How long is the key rotation overlap window for API keys?",
                   expected_sources=["api/auth.md"], tags=[]),
    ])


def _stage_factors(*names: str) -> list[Factor]:
    return [Factor(name=n, kind="stage") for n in names]


def _at_a_sets(arms):
    return {frozenset(a.at_a) for a in arms}


def _bare_diff(config_diff: dict, corpus_changed: bool) -> DiffResult:
    return DiffResult(
        diff_id="diff_x", snapshot_a="a", snapshot_b="b", dataset_id="ds",
        config_diff=config_diff, corpus_changed=corpus_changed, metric_deltas={},
        queries={}, alignment={}, parser_changed="parse" in config_diff,
        alignment_available=True,
    )


# ---------------------------------------------------------------- Shapley ----

class TestShapley:
    def test_additive_two_factors(self):
        metric = {
            frozenset(): 0.0,
            frozenset({"a"}): 1.0,
            frozenset({"b"}): 2.0,
            frozenset({"a", "b"}): 3.0,
        }
        phi = shapley(metric, ["a", "b"])
        assert phi == pytest.approx({"a": 1.0, "b": 2.0})
        assert sum(phi.values()) == pytest.approx(metric[frozenset({"a", "b"})] - metric[frozenset()])

    def test_interaction_split_evenly(self):
        metric = {
            frozenset(): 0.0,
            frozenset({"a"}): 1.0,
            frozenset({"b"}): 1.0,
            frozenset({"a", "b"}): 4.0,
        }
        phi = shapley(metric, ["a", "b"])
        # each factor: half its solo effect plus half the +2 interaction
        assert phi == pytest.approx({"a": 2.0, "b": 2.0})
        assert sum(phi.values()) == pytest.approx(4.0)

    def test_single_factor(self):
        metric = {frozenset(): 0.3, frozenset({"a"}): 0.9}
        assert shapley(metric, ["a"]) == pytest.approx({"a": 0.6})

    def test_empty_factor_list(self):
        assert shapley({frozenset(): 1.0}, []) == {}


# -------------------------------------------------------- enumerate_factors --

class TestEnumerateFactors:
    def test_stage_factors_only(self):
        factors = enumerate_factors(_bare_diff({"chunk": {}, "embed": {}}, corpus_changed=False))
        assert factors == [Factor("chunk", "stage"), Factor("embed", "stage")]

    def test_adds_corpus_factor(self):
        factors = enumerate_factors(_bare_diff({"chunk": {}, "corpus": {}}, corpus_changed=True))
        assert factors == [Factor("chunk", "stage"), Factor("corpus", "corpus")]

    def test_corpus_key_never_a_stage_factor(self):
        factors = enumerate_factors(_bare_diff({"corpus": {}}, corpus_changed=True))
        assert factors == [Factor("corpus", "corpus")]

    def test_sorted_deterministic(self):
        factors = enumerate_factors(_bare_diff({"retrieve": {}, "chunk": {}}, corpus_changed=False))
        assert [f.name for f in factors] == ["chunk", "retrieve"]


# --------------------------------------------------------------- build_arms --

class TestBuildArms:
    def test_k2_full_lattice(self):
        arms = build_arms(_stage_factors("chunk", "embed"))
        assert len(arms) == 4
        assert _at_a_sets(arms) == {
            frozenset(), frozenset({"chunk"}), frozenset({"embed"}), frozenset({"chunk", "embed"})}

    def test_every_arm_covers_all_factors(self):
        arms = build_arms(_stage_factors("chunk", "embed"))
        for arm in arms:
            assert set(arm.assignment) == {"chunk", "embed"}
            assert set(arm.assignment.values()) <= {"A", "B"}

    def test_k3_is_full_lattice(self):
        arms = build_arms(_stage_factors("chunk", "embed", "retrieve"))
        assert len(arms) == 8

    def test_k4_auto_ofat_pairs_endpoints(self):
        arms = build_arms(_stage_factors("chunk", "embed", "retrieve", "index"), mode="auto")
        sets = _at_a_sets(arms)
        # endpoints
        assert frozenset() in sets
        assert frozenset({"chunk", "embed", "retrieve", "index"}) in sets
        # OFAT
        for name in ("chunk", "embed", "retrieve", "index"):
            assert frozenset({name}) in sets
        # registered interaction pairs present in the factor set
        assert frozenset({"embed", "chunk"}) in sets
        assert frozenset({"retrieve", "chunk"}) in sets
        # exactly those: 1 + 4 + 2 + 1
        assert len(arms) == 8

    def test_k4_full_mode_is_full_lattice(self):
        arms = build_arms(_stage_factors("chunk", "embed", "retrieve", "index"), mode="full")
        assert len(arms) == 16

    def test_pruned_to_restricts_flips(self):
        arms = build_arms(_stage_factors("chunk", "embed", "retrieve", "index"),
                          mode="auto", pruned_to=["chunk"])
        for arm in arms:
            assert set(arm.at_a) <= {"chunk"}
        assert _at_a_sets(arms) == {frozenset(), frozenset({"chunk"})}

    def test_single_revert_arm_exists_per_factor(self):
        factors = _stage_factors("chunk", "embed", "retrieve", "index")
        sets = _at_a_sets(build_arms(factors, mode="auto"))
        for f in factors:
            assert frozenset({f.name}) in sets


# ------------------------------------------------------------ scenarios ------

def _build_chunk_embed(root: Path):
    """A: defaults; B: chunk fixed_token(60,0) + embed seed 7. Two stage factors."""
    store = ProjectStore(root)
    man_a = ingest(store, CORPUS, build_pipeline({}), None).manifest
    man_b = ingest(store, CORPUS, build_pipeline({
        "chunker": {"tool": chunkers.FIXED_TOKEN, "params": {"max_tokens": 60, "overlap": 0}},
        "embedding": {"provider": "local", "model": "hash-v1", "dims": 256,
                      "params": {"seed": 7, "ngram": [1, 2]}},
    }), None).manifest
    ds = _dataset()
    d = diff(store, man_a, man_b, ds, evaluate(store, man_a, ds), evaluate(store, man_b, ds))
    return store, man_a, man_b, ds, d


CORPUS = Path(__file__).resolve().parent.parent / "examples" / "corpus"


class TestMaterializeArm:
    def test_all_b_arm_equals_manifest_b_exactly(self, tmp_path):
        store, man_a, man_b, ds, d = _build_chunk_embed(tmp_path / "s")
        arm = Arm.build({"chunk": "B", "embed": "B"})
        manifest = materialize_arm(store, arm, man_a, man_b, CORPUS)
        assert manifest.snapshot_id == man_b.snapshot_id
        assert manifest.to_json() == man_b.to_json()

    def test_all_a_arm_equals_manifest_a_exactly(self, tmp_path):
        store, man_a, man_b, ds, d = _build_chunk_embed(tmp_path / "s")
        arm = Arm.build({"chunk": "A", "embed": "A"})
        manifest = materialize_arm(store, arm, man_a, man_b, CORPUS)
        assert manifest.snapshot_id == man_a.snapshot_id
        assert manifest.to_json() == man_a.to_json()

    def test_cross_arm_is_new_snapshot(self, tmp_path):
        store, man_a, man_b, ds, d = _build_chunk_embed(tmp_path / "s")
        arm = Arm.build({"chunk": "A", "embed": "B"})
        manifest = materialize_arm(store, arm, man_a, man_b, CORPUS)
        assert manifest.snapshot_id not in (man_a.snapshot_id, man_b.snapshot_id)

    def test_memoization_zero_embed_calls_on_reentry(self, tmp_path):
        store, man_a, man_b, ds, d = _build_chunk_embed(tmp_path / "s")
        arm = Arm.build({"chunk": "A", "embed": "B"})
        m1, calls1 = _materialize_arm(store, arm, man_a, man_b, CORPUS)
        m2, calls2 = _materialize_arm(store, arm, man_a, man_b, CORPUS)
        assert calls1 > 0            # genuinely new embeddings the first time
        assert calls2 == 0           # FR-7.1: memoized second time
        assert m1.snapshot_id == m2.snapshot_id


def _corpus_copy(root: Path, extra: dict | None = None) -> Path:
    dest = root / "corpus"
    shutil.copytree(CORPUS, dest)
    for rel, text in (extra or {}).items():
        p = dest / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return dest


class TestMaterializeFromStore:
    """A novel (corpus, chunk) combination forces re-parsing and re-chunking
    documents out of the store; the result must match an independent ingest of
    the same corpus+config byte-for-byte and be memoized on re-entry."""

    def test_rechunk_from_store_matches_ingest(self, tmp_path):
        long_doc = "# New Doc\n\n" + (
            "An added document about ledger reconciliation and settlement windows "
            "for downstream reporting across many regional processors. " * 8) + "\n"
        dir_a = _corpus_copy(tmp_path / "a")
        dir_b = _corpus_copy(tmp_path / "b", extra={"extra/newdoc.md": long_doc})
        store = ProjectStore(tmp_path / "s")
        man_a = ingest(store, dir_a, build_pipeline({}), None).manifest
        man_b = ingest(store, dir_b, build_pipeline({
            "chunker": {"tool": chunkers.FIXED_TOKEN, "params": {"max_tokens": 60, "overlap": 0}},
        }), None).manifest

        # cross arm: A's chunker over B's (larger) corpus -> a chunkset neither
        # ingest produced, so it must be reconstructed from stored raw bytes.
        arm = Arm.build({"chunk": "A", "corpus": "B"})
        cross, calls = _materialize_arm(store, arm, man_a, man_b, dir_b)
        assert calls > 0

        # independent ground truth: ingest B's corpus with A's config directly.
        reference = ingest(store, dir_b, build_pipeline({}), None).manifest
        assert cross.snapshot_id == reference.snapshot_id
        assert cross.to_json() == reference.to_json()

        _, calls2 = _materialize_arm(store, arm, man_a, man_b, dir_b)
        assert calls2 == 0

    def test_rechunk_from_store_with_deleted_doc_matches_ingest(self, tmp_path):
        # Finding #1 (critical): corpus B REMOVED a non-last document. The cross
        # arm holding corpus at B must reconstruct B's true doc set from the store
        # (record_corpus_state), not the empty set the monotone reconstruction
        # would return, else the control arm retrieves nothing and no genuine
        # corpus-drift cause can ever be confirmed (a false negative).
        dir_a = _corpus_copy(tmp_path / "a")
        dir_b = _corpus_copy(tmp_path / "b")
        (dir_b / "api" / "auth.md").unlink()  # delete a doc early in sorted order
        store = ProjectStore(tmp_path / "s")
        man_a = ingest(store, dir_a, build_pipeline({}), None).manifest
        man_b = ingest(store, dir_b, build_pipeline({
            "chunker": {"tool": chunkers.FIXED_TOKEN, "params": {"max_tokens": 60, "overlap": 0}},
        }), None).manifest
        assert man_b.corpus.doc_count == man_a.corpus.doc_count - 1

        arm = Arm.build({"chunk": "A", "corpus": "B"})
        cross, calls = _materialize_arm(store, arm, man_a, man_b, dir_b)
        reference = ingest(store, dir_b, build_pipeline({}), None).manifest
        assert cross.corpus.doc_count == reference.corpus.doc_count == man_b.corpus.doc_count
        assert cross.corpus.chunk_count > 0
        assert cross.snapshot_id == reference.snapshot_id
        assert cross.to_json() == reference.to_json()

    def test_corpus_factor_selects_a_state(self, tmp_path):
        dir_a = _corpus_copy(tmp_path / "a")
        dir_b = _corpus_copy(tmp_path / "b", extra={
            "extra/newdoc.md": "# New Doc\n\nAn added document about settlement windows.\n"})
        store = ProjectStore(tmp_path / "s")
        man_a = ingest(store, dir_a, build_pipeline({}), None).manifest
        man_b = ingest(store, dir_b, build_pipeline({}), None).manifest
        # corpus is the only factor; reverting it must reproduce A exactly.
        arm = Arm.build({"corpus": "A"})
        manifest = materialize_arm(store, arm, man_a, man_b, dir_b)
        assert manifest.snapshot_id == man_a.snapshot_id
        assert manifest.corpus.doc_count == man_a.corpus.doc_count


class TestPlanArms:
    def test_local_provider_zero_cost_and_predicts_embeds(self, tmp_path):
        store, man_a, man_b, ds, d = _build_chunk_embed(tmp_path / "s")
        arms = build_arms(enumerate_factors(d))
        plan = plan_arms(store, man_a, man_b, CORPUS, arms,
                         lambda stage: get_provider(dict(stage.params)))
        assert isinstance(plan, ArmPlan)
        assert plan.est_usd == 0.0
        assert plan.est_wall_s >= 0.0
        # cross arms need genuinely new embeddings; endpoints are already cached
        assert plan.new_embed_texts > 0

    def test_plan_matches_actual_new_embeds(self, tmp_path):
        store, man_a, man_b, ds, d = _build_chunk_embed(tmp_path / "s")
        arms = build_arms(enumerate_factors(d))
        plan = plan_arms(store, man_a, man_b, CORPUS, arms,
                         lambda stage: get_provider(dict(stage.params)))
        results = run_arms(store, arms, man_a, man_b, CORPUS, ds, "job-plan")
        assert plan.new_embed_texts == sum(r.embed_calls for r in results.values())


class TestRunArms:
    def test_resumable_no_rematerialization(self, tmp_path, monkeypatch):
        store, man_a, man_b, ds, d = _build_chunk_embed(tmp_path / "s")
        arms = build_arms(enumerate_factors(d))
        first = run_arms(store, arms, man_a, man_b, CORPUS, ds, "job-1")
        assert sum(r.embed_calls for r in first.values()) > 0

        def _boom(*args, **kwargs):
            raise AssertionError("resume must not re-materialize any arm")

        monkeypatch.setattr(ablation, "_materialize_arm", _boom)
        second = run_arms(store, arms, man_a, man_b, CORPUS, ds, "job-1")
        assert set(second) == set(first)
        assert all(r.embed_calls == 0 for r in second.values())
        for aid in first:
            assert second[aid].eval.run_id == first[aid].eval.run_id

    def test_arm_results_are_replay_evals(self, tmp_path):
        store, man_a, man_b, ds, d = _build_chunk_embed(tmp_path / "s")
        arms = build_arms(enumerate_factors(d))
        results = run_arms(store, arms, man_a, man_b, CORPUS, ds, "job-2")
        assert set(results) == {a.arm_id for a in arms}
        for r in results.values():
            assert r.eval.mode == "replay"
            assert r.eval.adapter == "none"
