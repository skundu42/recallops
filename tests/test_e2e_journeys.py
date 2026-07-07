"""End-to-end user journeys (PRD §11, journeys J1/J3/J4).

These exercise the real engine the way a user would: the CLI surface via
``click.testing.CliRunner`` where a journey is command-shaped, and the library
API directly where a journey needs to inspect verified causes. J1 is the core
regression-and-attribution loop, J3 the embedding migration report, J4 the
corpus-drift attribution.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from recallops.ablation import build_arms, enumerate_factors, run_arms
from recallops.cli import main
from recallops.config import ProjectConfig
from recallops.confirm import confirm_causes, fidelity_check
from recallops.diffing import diff
from recallops.evalrunner import evaluate
from recallops.funnel import funnel_for_query
from recallops.ingest import build_pipeline, ingest
from recallops.models import Arm, Factor, GoldenCase, GoldenDataset
from recallops.retrieval import RetrievalEngine
from recallops.store import ProjectStore

CORPUS = Path(__file__).resolve().parent.parent / "examples" / "corpus"
FIXED_TOKEN = "recall.chunkers.fixed_token"
FT_PARAMS = '{"max_tokens": 30, "overlap": 0}'


def _run(runner: CliRunner, args: list[str], code: int = 0):
    result = runner.invoke(main, args)
    if result.exit_code != code:
        raise AssertionError(
            f"`recall {' '.join(args)}` exited {result.exit_code} (expected {code})\n"
            f"output:\n{result.output}\nexception: {result.exception!r}"
        )
    return result


def _make_dense_only(path: str = "recall.yaml") -> None:
    cfg = ProjectConfig.load(path)
    cfg.pipeline["retrieve"] = {"top_k": 10, "hybrid": None}
    cfg.save(path)


def _latest_snapshot() -> str:
    return ProjectStore(".").list_snapshots()[-1].snapshot_id


# ============================================================ J1: full journey


class TestJ1FullJourney:
    """init -> ingest -> generate -> eval GREEN -> chunker change -> eval RED ->
    diff -> deep attribute -> verified 'chunk' cause -> revert -> GREEN again."""

    def test_j1_full_journey(self, corpus_dir):
        runner = CliRunner()
        with runner.isolated_filesystem():
            # init + baseline ingest (A) + golden dataset
            _run(runner, ["init", "--source", str(corpus_dir)])
            _make_dense_only()
            _run(runner, ["ingest"])
            snap_a = _latest_snapshot()
            _run(runner, ["dataset", "generate", "--n", "20", "--seed", "0", "--name", "gold"])

            # eval A is GREEN
            _run(runner, ["eval", "gold-v1", "--snapshot", snap_a])
            store = ProjectStore(".")
            ev_a = store.find_eval(snap_a, "gold-v1")
            assert ev_a.aggregate["recall@5"] >= 0.8

            # a chunker change (B) turns the eval RED
            _run(runner, ["ingest", "--chunker", FIXED_TOKEN, "--chunk-params", FT_PARAMS])
            snap_b = _latest_snapshot()
            assert snap_b != snap_a
            _run(runner, ["eval", "gold-v1", "--snapshot", snap_b])
            ev_b = ProjectStore(".").find_eval(snap_b, "gold-v1")
            assert ev_b.aggregate["recall@5"] < ev_a.aggregate["recall@5"]

            # diff + deep attribution: the verified cause is 'chunk'
            deep = _run(runner, ["diff", snap_a, snap_b, "--dataset", "gold-v1",
                                 "--attribute", "deep"])
            assert "regressed" in deep.output
            assert "verified" in deep.output
            assert "chunk" in deep.output

            store = ProjectStore(".")
            diff_id = store.list_json("diff")[0]
            attr = store.get_json("attribution", diff_id)
            assert attr, "deep attribution produced no reports"
            verified_factors = {
                vc["factor"]
                for rep in attr.values()
                for vc in rep["verified_causes"]
            }
            assert verified_factors == {"chunk"}
            revert_arm = Arm.build({"chunk": "A"}).arm_id
            for rep in attr.values():
                for vc in rep["verified_causes"]:
                    assert vc["arm_id"] == revert_arm
                    assert vc["status"] == "verified"

            # revert the chunker (config still holds markdown_heading) -> GREEN again.
            # The revert is content-addressed: it reproduces snapshot A exactly, so
            # committing it is idempotent (no new row), assert via the ingest report.
            store = ProjectStore(".")
            cfg = ProjectConfig.load("recall.yaml")
            revert = ingest(store, corpus_dir, build_pipeline(cfg.pipeline), None)
            assert revert.manifest.snapshot_id == snap_a
            ev_c = evaluate(store, revert.manifest, store.get_dataset("gold-v1"), mode="replay")
            assert ev_c.aggregate["recall@5"] == ev_a.aggregate["recall@5"] >= 0.8


# ==================================================== J3: compare embeddings


class TestJ3CompareEmbeddings:
    """Two local embedding specs -> a migration report with per-tag deltas, a
    recommendation, and a zero-cost auto-approval path (local provider is $0)."""

    def test_j3_compare_embeddings(self, corpus_dir):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _run(runner, ["init", "--source", str(corpus_dir)])
            _make_dense_only()
            _run(runner, ["ingest"])
            _run(runner, ["dataset", "generate", "--n", "20", "--seed", "0", "--name", "gold"])

            # zero-cost approval path: no --yes / --max-cost needed for local specs
            result = _run(runner, ["compare-embeddings", "--from", "local:hash-v1:256",
                                   "--to", "local:hash-v1b:256", "--dataset", "gold-v1"])
            out = result.output
            assert "comparison report" in out.lower()
            # per-tag deltas: generated cases always carry exact-term / paraphrase tags
            assert "exact-term" in out and "paraphrase" in out
            assert "overall" in out
            # a recommendation block
            assert "Recommendation" in out
            assert "hybrid_bm25_weight" in out
            # zero-cost path was taken (no approval required)
            assert "$0.00" in out

    def test_j3_nonzero_cost_requires_approval(self, corpus_dir):
        """A billed (openai) target trips the cost gate without --yes/--max-cost,
        proving the zero-cost path above was a genuine auto-approval."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            _run(runner, ["init", "--source", str(corpus_dir)])
            _make_dense_only()
            _run(runner, ["ingest"])
            _run(runner, ["dataset", "generate", "--n", "10", "--seed", "0", "--name", "gold"])
            result = _run(runner, ["compare-embeddings", "--from", "local:hash-v1:256",
                                   "--to", "openai:text-embedding-3-small:1536",
                                   "--dataset", "gold-v1"], code=1)
            assert "cost" in result.output.lower()


# ================================================================== J4: drift

_PHRASE = "quokka ledger epoch synchronization protocol relay"
_QUESTION = ("How does the quokka ledger epoch synchronization protocol relay "
             "coordinate state?")
_TARGET_PATH = "sync/target.md"
_TARGET = (
    "# Ledger Synchronization Guide\n\n"
    f"The {_PHRASE} coordinates distributed ledger state across regional nodes.\n\n"
    "## Operational Detail\n\n"
    "Operators configure the regional coordinator with a rotation calendar, a quorum "
    "threshold, and a fallback contact roster. The runbook documents escalation windows, "
    "maintenance freezes, and the archival policy for superseded snapshots.\n"
)
_FILLER = {
    "misc/weather.md": "# Weather Notes\n\nThe afternoon forecast predicts scattered showers "
    "across the coastal plain. Gardeners should cover seedlings before dusk.\n",
    "misc/recipes.md": "# Kitchen Log\n\nThe sourdough starter needs feeding twice daily with "
    "equal parts flour and water. Bake at high heat with steam for a crisp crust.\n",
    "misc/travel.md": "# Travel Diary\n\nThe mountain trail climbs past alpine meadows toward a "
    "quiet lake. Bring layers because temperatures drop quickly after sunset.\n",
}
_MIRRORS = {
    "sync/mirror-one.md": f"# Mirror One\n\n{_PHRASE}. {_PHRASE} confirmed.\n",
    "sync/mirror-two.md": f"# Mirror Two\n\n{_PHRASE}. {_PHRASE} verified.\n",
}
_DISTRACTORS = {
    "sync/dup-alpha.md": f"# Dup Alpha\n\n{_PHRASE} {_PHRASE}. {_PHRASE}.\n",
    "sync/dup-beta.md": f"# Dup Beta\n\n{_PHRASE} {_PHRASE}. {_PHRASE} now.\n",
    "sync/dup-gamma.md": f"# Dup Gamma\n\n{_PHRASE} {_PHRASE}. {_PHRASE} here.\n",
}
_DENSE_ONLY = {"retrieve": {"top_k": 10, "hybrid": None}}


def _write_corpus(root: Path, docs: dict[str, str]) -> None:
    for rel, text in docs.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


@pytest.fixture(scope="module")
def drift(tmp_path_factory):
    root = tmp_path_factory.mktemp("j4")
    dir_a = root / "a"
    dir_b = root / "b"
    _write_corpus(dir_a, {**_FILLER, **_MIRRORS, _TARGET_PATH: _TARGET})
    _write_corpus(dir_b, {**_FILLER, **_MIRRORS, _TARGET_PATH: _TARGET, **_DISTRACTORS})

    store = ProjectStore(root / "store")
    man_a = ingest(store, dir_a, build_pipeline(_DENSE_ONLY), None).manifest
    # distractors + a no-op index config touch: corpus and a config factor
    # change together (FR-7.6 / FR-11.1)
    man_b = ingest(store, dir_b, build_pipeline({
        **_DENSE_ONLY, "index": {"adapter": "local", "collection": "drift-b"},
    }), None).manifest

    ds = GoldenDataset("drift-v1", [
        GoldenCase(id="qd", question=_QUESTION, expected_sources=[_TARGET_PATH], tags=[])])
    ev_a = evaluate(store, man_a, ds, mode="replay")
    ev_b = evaluate(store, man_b, ds, mode="replay")
    dr = diff(store, man_a, man_b, ds, ev_a, ev_b)

    engine_a = RetrievalEngine(store, man_a)
    engine_b = RetrievalEngine(store, man_b)
    funnels = {qd.query_id: funnel_for_query(engine_a, engine_b, qd, ds.case(qd.query_id),
                                             dr.alignment)
               for qd in dr.by_class("regressed", stable_only=True)}
    arms = build_arms(enumerate_factors(dr))
    arm_results = run_arms(store, arms, man_a, man_b, dir_b, ds, "j4-job")
    reports = confirm_causes(dr, ds, arm_results, arms, funnels, dr.alignment,
                             recovery_threshold="rank1", top_k=5)
    return {"store": store, "engine_b": engine_b, "ds": ds, "diff": dr,
            "arms": arms, "arm_results": arm_results, "reports": reports,
            "ev_a": ev_a, "ev_b": ev_b}


class TestJ4Drift:
    """An injected near-duplicate distractor outranks a golden target under
    corpus drift; the diff surfaces it and the corpus factor is *verified* by the
    corpus-revert arm while the co-changed config factor stays a hypothesis."""

    def test_distractor_outranks_target(self, drift):
        qe_b = drift["ev_b"].per_query["qd"]
        ranked = qe_b.ranked_docs
        assert _TARGET_PATH in ranked
        target_idx = ranked.index(_TARGET_PATH)
        above = ranked[:target_idx]
        assert any(d.startswith("sync/dup-") for d in above), \
            f"expected an injected distractor above the target, got {above}"

    def test_target_evicted_from_top_5(self, drift):
        qd = drift["diff"].queries["qd"]
        assert qd.before.target_rank <= 5
        assert qd.after.target_rank > 5

    def test_drift_diff_surfaces_corpus_change(self, drift):
        dr = drift["diff"]
        assert dr.corpus_changed is True
        assert enumerate_factors(dr) == [Factor("index", "stage"), Factor("corpus", "corpus")]
        assert dr.queries["qd"].classification == "regressed"
        assert dr.queries["qd"].stability == "stable"

    def test_corpus_factor_is_verified(self, drift):
        rep = drift["reports"]["qd"]
        verified = {c.factor for c in rep.verified_causes}
        assert verified == {"corpus"}

    def test_config_factor_stays_a_hypothesis(self, drift):
        rep = drift["reports"]["qd"]
        assert "index" not in {c.factor for c in rep.verified_causes}
        assert "index" in {h.factor for h in rep.hypotheses}

    def test_corpus_revert_recovers_within_original_rank(self, drift):
        rep = drift["reports"]["qd"]
        before_rank = drift["diff"].queries["qd"].before.target_rank
        cause = next(c for c in rep.verified_causes if c.factor == "corpus")
        assert cause.recovered_rank <= before_rank + 1

    def test_fidelity_is_one(self, drift):
        assert fidelity_check(drift["reports"], drift["arm_results"], drift["arms"],
                              drift["ds"], 5) == 1.0

    def test_cli_drift_command_surfaces_the_distractor(self, corpus_dir):
        """The `recall drift` command surfaces the injected distractor in its
        funnel output on a live project."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_corpus(Path("docs"), {**_FILLER, **_MIRRORS, _TARGET_PATH: _TARGET})
            _run(runner, ["init", "--source", "docs"])
            _make_dense_only()
            _run(runner, ["ingest"])
            snap_a = _latest_snapshot()
            Path("ds.json").write_text(json.dumps({"cases": [
                {"id": "qd", "question": _QUESTION, "expected_sources": [_TARGET_PATH],
                 "tags": []}]}), encoding="utf-8")
            _run(runner, ["dataset", "import", "ds.json", "--name", "drift"])
            _write_corpus(Path("docs"), _DISTRACTORS)
            _run(runner, ["ingest"])
            snap_b = _latest_snapshot()
            assert snap_b != snap_a
            result = _run(runner, ["drift", "--against", snap_a, "--dataset", "drift-v1",
                                   "--snapshot", snap_b])
            assert "distractor" in result.output
