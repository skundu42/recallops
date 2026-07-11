from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from recallops.cli import main
from recallops.config import (
    ProjectConfig,
    dump_simple_yaml,
    parse_embedding_spec,
    parse_simple_yaml,
)
from recallops.store import ProjectStore

FIXED_TOKEN = "recall.chunkers.fixed_token"
FT_PARAMS = '{"max_tokens": 30, "overlap": 0}'


def _run(runner: CliRunner, args: list[str], code: int = 0) -> object:
    result = runner.invoke(main, args)
    if result.exit_code != code:
        exc = result.exception
        raise AssertionError(
            f"`recall {' '.join(args)}` exited {result.exit_code} (expected {code})\n"
            f"output:\n{result.output}\nexception: {exc!r}"
        )
    return result


def _make_dense_only(config_path: str = "recall.yaml") -> None:
    cfg = ProjectConfig.load(config_path)
    cfg.pipeline["retrieve"] = {"top_k": 10, "hybrid": None}
    cfg.save(config_path)


def _bootstrap(runner: CliRunner, source: str, *, dense_only: bool = True, n: int = 20) -> str:
    """init + (dense-only) + ingest A + generate; returns dataset id."""
    _run(runner, ["init", "--source", source])
    if dense_only:
        _make_dense_only()
    _run(runner, ["ingest"])
    _run(runner, ["dataset", "generate", "--n", str(n), "--seed", "0", "--name", "gold"])
    return "gold-v1"


def _latest_snapshot() -> str:
    return ProjectStore(".").list_snapshots()[-1].snapshot_id


# -- config: YAML subset ------------------------------------------------------


def test_yaml_roundtrip_scalars_maps_lists():
    data = {
        "project": "demo",
        "source": "docs",
        "adapter": {"type": "local"},
        "pipeline": {
            "chunker": {"tool": "recall.chunkers.markdown_heading",
                        "params": {"max_tokens": 800, "overlap": 120}},
            "embedding": {"provider": "local", "model": "hash-v1", "dims": 256,
                          "params": {"seed": 0, "ngram": [1, 2]}},
            "retrieve": {"top_k": 10, "hybrid": None},
        },
        "gate": {"mode": "statistical", "primary_metric": "recall@5", "q": 0.05},
        "costs": {"max_cost": 0.0},
        "flag": True,
        "off": False,
    }
    assert parse_simple_yaml(dump_simple_yaml(data)) == data


def test_yaml_scalar_types_preserved():
    parsed = parse_simple_yaml(dump_simple_yaml({"i": 7, "f": 0.45, "b": True,
                                                 "n": None, "s": "recall@5"}))
    assert parsed == {"i": 7, "f": 0.45, "b": True, "n": None, "s": "recall@5"}
    assert isinstance(parsed["i"], int) and isinstance(parsed["f"], float)


def test_yaml_dsn_with_colons_roundtrips():
    data = {"adapter": {"type": "pgvector", "dsn": "postgresql://u:p@host:5432/db"}}
    assert parse_simple_yaml(dump_simple_yaml(data)) == data


def test_yaml_numeric_looking_string_is_quoted_and_preserved():
    parsed = parse_simple_yaml(dump_simple_yaml({"model": "256", "n": 256}))
    assert parsed["model"] == "256" and isinstance(parsed["model"], str)
    assert parsed["n"] == 256 and isinstance(parsed["n"], int)


def test_project_config_default_roundtrip():
    cfg = ProjectConfig.default(project="demo", source="corpus")
    with CliRunner().isolated_filesystem():
        cfg.save("recall.yaml")
        loaded = ProjectConfig.load("recall.yaml")
    assert loaded == cfg
    assert loaded.adapter == {"type": "local"}
    assert loaded.pipeline["chunker"]["tool"] == "recall.chunkers.markdown_heading"


def test_parse_embedding_spec_forms():
    assert parse_embedding_spec("local:hash-v1b:256") == {
        "provider": "local", "model": "hash-v1b", "dims": 256,
        "params": {"seed": 0, "ngram": [1, 2]},
    }
    assert parse_embedding_spec("hash-v1")["provider"] == "local"
    assert parse_embedding_spec("openai:text-embedding-3-small:1536") == {
        "provider": "openai", "model": "text-embedding-3-small", "dims": 1536, "params": {},
    }


# -- init ---------------------------------------------------------------------


def test_init_writes_config_and_store(corpus_dir):
    runner = CliRunner()
    with runner.isolated_filesystem():
        _run(runner, ["init", "--source", str(corpus_dir), "--name", "myproj"])
        assert Path("recall.yaml").exists()
        assert Path(".recall").is_dir()
        cfg = ProjectConfig.load("recall.yaml")
        assert cfg.project == "myproj"
        assert cfg.adapter["type"] == "local"
        # re-init without --force fails
        _run(runner, ["init", "--source", str(corpus_dir)], code=1)


# -- ingest + eval (green) ----------------------------------------------------


def test_ingest_then_eval_green(corpus_dir):
    runner = CliRunner()
    with runner.isolated_filesystem():
        ds_id = _bootstrap(runner, str(corpus_dir))
        snap = _latest_snapshot()
        result = _run(runner, ["eval", ds_id, "--snapshot", snap])
        assert "recall@5" in result.output
        ev = ProjectStore(".").find_eval(snap, ds_id)
        assert ev.aggregate["recall@5"] >= 0.8  # green


def test_ingest_reingest_is_zero_embed(corpus_dir):
    runner = CliRunner()
    with runner.isolated_filesystem():
        _run(runner, ["init", "--source", str(corpus_dir)])
        _make_dense_only()
        _run(runner, ["ingest"])
        result = _run(runner, ["ingest"])
        assert "embed_calls=0" in result.output
        assert "reused_chunks=72" in result.output


# -- full attribution journey -------------------------------------------------


def test_full_journey_fast_deep_report(corpus_dir):
    runner = CliRunner()
    with runner.isolated_filesystem():
        ds_id = _bootstrap(runner, str(corpus_dir))
        snap_a = _latest_snapshot()
        _run(runner, ["ingest", "--chunker", FIXED_TOKEN, "--chunk-params", FT_PARAMS])
        snap_b = _latest_snapshot()
        assert snap_b != snap_a

        # fast attribution: regressions + funnel
        fast = _run(runner, ["diff", snap_a, snap_b, "--dataset", ds_id, "--attribute", "fast"])
        assert "regressed" in fast.output
        assert "split" in fast.output
        assert "chunk" in fast.output

        # deep attribution: verified causes
        deep = _run(runner, ["diff", snap_a, snap_b, "--dataset", ds_id, "--attribute", "deep"])
        assert "verified" in deep.output
        assert "chunk" in deep.output

        # diff --format json carries the full diff schema
        as_json = _run(runner, ["diff", snap_a, snap_b, "--dataset", ds_id,
                                "--attribute", "deep", "--format", "json"])
        doc = json.loads(as_json.output)
        assert doc["snapshot_a"] == snap_a and doc["snapshot_b"] == snap_b
        assert "chunk" in doc["config_diff"]
        assert doc["attributions"]

        diff_id = doc["diff_id"]

        # report --format json re-renders from stored artifacts and validates §10.4 keys
        rep = _run(runner, ["report", "--diff", diff_id, "--format", "json"])
        rendered = json.loads(rep.output)
        assert rendered["diff_id"] == diff_id
        attr = next(iter(rendered["attributions"].values()))
        assert set(attr) >= {"query_id", "classification", "stability", "funnel",
                             "chunk_fate", "verified_causes", "hypotheses", "narrative"}
        assert set(attr["funnel"]) >= {"target_chunk_before", "target_in_index_after",
                                       "dense", "sparse", "fused", "rerank", "ann_divergence"}
        vc = attr["verified_causes"][0]
        assert set(vc) >= {"factor", "arm_id", "recovered_rank", "status"}
        assert vc["factor"] == "chunk" and vc["status"] == "verified"


def test_attribute_command_persists_and_prints(corpus_dir):
    runner = CliRunner()
    with runner.isolated_filesystem():
        ds_id = _bootstrap(runner, str(corpus_dir))
        snap_a = _latest_snapshot()
        _run(runner, ["ingest", "--chunker", FIXED_TOKEN, "--chunk-params", FT_PARAMS])
        snap_b = _latest_snapshot()
        # fast diff (no deep) to persist the diff artifact
        _run(runner, ["diff", snap_a, snap_b, "--dataset", ds_id])
        diff_id = ProjectStore(".").list_json("diff")[0]
        result = _run(runner, ["attribute", diff_id])
        assert "verified" in result.output
        assert ProjectStore(".").get_json("attribution", diff_id)


def test_report_html_is_self_contained(corpus_dir):
    runner = CliRunner()
    with runner.isolated_filesystem():
        ds_id = _bootstrap(runner, str(corpus_dir))
        snap_a = _latest_snapshot()
        _run(runner, ["ingest", "--chunker", FIXED_TOKEN, "--chunk-params", FT_PARAMS])
        snap_b = _latest_snapshot()
        _run(runner, ["diff", snap_a, snap_b, "--dataset", ds_id, "--attribute", "deep"])
        diff_id = ProjectStore(".").list_json("diff")[0]
        _run(runner, ["report", "--diff", diff_id, "--format", "html", "-o", "out.html"])
        html = Path("out.html").read_text()
        assert "<!DOCTYPE html>" in html
        assert "http://" not in html and "https://" not in html


# -- gates --------------------------------------------------------------------


def test_eval_fail_if_returns_exit_1(corpus_dir):
    runner = CliRunner()
    with runner.isolated_filesystem():
        ds_id = _bootstrap(runner, str(corpus_dir))
        snap = _latest_snapshot()
        result = _run(runner, ["eval", ds_id, "--snapshot", snap,
                               "--fail-if", "recall@5<1.0"], code=1)
        assert "FAIL" in result.output


def test_eval_fail_if_passes_when_condition_false(corpus_dir):
    runner = CliRunner()
    with runner.isolated_filesystem():
        ds_id = _bootstrap(runner, str(corpus_dir))
        result = _run(runner, ["eval", ds_id, "--fail-if", "recall@5<0.0"])
        assert "PASS" in result.output


def test_eval_gate_statistical_without_calibration_errors(corpus_dir):
    runner = CliRunner()
    with runner.isolated_filesystem():
        ds_id = _bootstrap(runner, str(corpus_dir))
        result = _run(runner, ["eval", ds_id, "--gate", "statistical"], code=1)
        assert "calibrate" in result.output.lower()


def test_calibrate_smoke(corpus_dir):
    runner = CliRunner()
    with runner.isolated_filesystem():
        ds_id = _bootstrap(runner, str(corpus_dir))
        result = _run(runner, ["calibrate", "--dataset", ds_id, "--runs", "2"])
        assert "epsilon" in result.output


# -- ci -----------------------------------------------------------------------


def test_ci_writes_report(corpus_dir):
    runner = CliRunner()
    with runner.isolated_filesystem():
        ds_id = _bootstrap(runner, str(corpus_dir))
        # ingest the changed pipeline so ci can diff current vs parent
        _run(runner, ["ingest", "--chunker", FIXED_TOKEN, "--chunk-params", FT_PARAMS])
        # point the config at the changed chunker so ci re-ingests the current (B) state
        cfg = ProjectConfig.load("recall.yaml")
        cfg.pipeline["chunker"] = {"tool": FIXED_TOKEN, "params": {"max_tokens": 30, "overlap": 0}}
        cfg.save("recall.yaml")
        result = _run(runner, ["ci", "--dataset", ds_id])
        assert Path("recall-report.md").exists()
        report = Path("recall-report.md").read_text()
        assert "RecallOps retrieval diff" in report
        assert "Phase 2 deep attribution" in result.output


def test_ci_uses_baseline_calibration_not_fresh_snapshot(corpus_dir):
    # Finding #3: the statistical gate must activate in CI using the BASELINE's
    # calibration. ci re-ingests the PR config into a fresh snapshot that was never
    # calibrated, so looking calibration up there would silently skip the gate.
    runner = CliRunner()
    with runner.isolated_filesystem():
        ds_id = _bootstrap(runner, str(corpus_dir))
        # Calibrate the baseline (A is latest right after bootstrap).
        _run(runner, ["calibrate", "--dataset", ds_id, "--runs", "2"])
        # A PR changes the chunker; point the config at it so ci re-ingests B.
        _run(runner, ["ingest", "--chunker", FIXED_TOKEN, "--chunk-params", FT_PARAMS])
        cfg = ProjectConfig.load("recall.yaml")
        cfg.pipeline["chunker"] = {"tool": FIXED_TOKEN, "params": {"max_tokens": 30, "overlap": 0}}
        cfg.save("recall.yaml")
        _run(runner, ["ci", "--dataset", ds_id])
        report = Path("recall-report.md").read_text()
        assert "Calibration: present" in report
        assert "Calibration: not present" not in report


def test_gate_works_for_non_default_primary_metric(corpus_dir):
    # Round-3 finding: threading primary_metric into diff() made classify_query
    # read metrics[primary_metric] unconditionally; the eval must compute that k
    # even when it's outside (1,5,10), else `ci`/`eval --gate` crash with KeyError.
    runner = CliRunner()
    with runner.isolated_filesystem():
        ds_id = _bootstrap(runner, str(corpus_dir))
        cfg = ProjectConfig.load("recall.yaml")
        cfg.gate["primary_metric"] = "recall@7"  # k=7 is not in the default (1,5,10)
        cfg.save("recall.yaml")
        _run(runner, ["calibrate", "--dataset", ds_id, "--runs", "2"])
        # eval --gate statistical must not KeyError
        _run(runner, ["eval", ds_id, "--gate", "statistical"])
        # ci must not KeyError and must produce a gate verdict
        _run(runner, ["ingest", "--chunker", FIXED_TOKEN, "--chunk-params", FT_PARAMS])
        cfg = ProjectConfig.load("recall.yaml")
        cfg.pipeline["chunker"] = {"tool": FIXED_TOKEN, "params": {"max_tokens": 30, "overlap": 0}}
        cfg.save("recall.yaml")
        result = runner.invoke(main, ["ci", "--dataset", ds_id])
        assert result.exit_code in (0, 1)  # a real verdict, not a crash
        assert "KeyError" not in result.output
        assert Path("recall-report.md").exists()


def test_ci_first_ingest_no_base(corpus_dir):
    runner = CliRunner()
    with runner.isolated_filesystem():
        _run(runner, ["init", "--source", str(corpus_dir)])
        _make_dense_only()
        _run(runner, ["dataset"], code=2)  # subcommand required
        _run(runner, ["ingest"])
        _run(runner, ["dataset", "generate", "--n", "10", "--name", "gold"])
        result = _run(runner, ["ci", "--dataset", "gold-v1"])
        assert Path("recall-report.md").exists()
        assert "No base snapshot" in result.output


# -- snapshots / datasets / gc ------------------------------------------------


def test_snapshot_list_and_show(corpus_dir):
    runner = CliRunner()
    with runner.isolated_filesystem():
        _bootstrap(runner, str(corpus_dir))
        snap = _latest_snapshot()
        listed = _run(runner, ["snapshot", "list"])
        assert snap[:12] in listed.output
        shown = _run(runner, ["snapshot", "show", snap])
        assert "pipeline" in shown.output


def test_dataset_list_show_curate(corpus_dir):
    runner = CliRunner()
    with runner.isolated_filesystem():
        ds_id = _bootstrap(runner, str(corpus_dir), n=6)
        listed = _run(runner, ["dataset", "list"])
        assert ds_id in listed.output
        _run(runner, ["dataset", "show", ds_id])
        _run(runner, ["dataset", "curate", ds_id, "--reject", "q_000"])
        assert ProjectStore(".").get_dataset(ds_id).case("q_000") is None


def test_dataset_import_and_mine(corpus_dir):
    runner = CliRunner()
    with runner.isolated_filesystem():
        _run(runner, ["init", "--source", str(corpus_dir)])
        Path("imp.jsonl").write_text(
            json.dumps({"question": "What is the refund policy?",
                        "ground_truth": "billing/refunds.md"}) + "\n"
        )
        _run(runner, ["dataset", "import", "imp.jsonl", "--name", "imp"])
        Path("mine.jsonl").write_text(
            json.dumps({"query": "sso setup", "expected_sources": ["security/sso.md"],
                        "trace_id": "tr_1"}) + "\n"
        )
        _run(runner, ["dataset", "mine", "--from", "mine.jsonl", "--name", "prod"])
        names = ProjectStore(".").list_datasets()
        assert "imp-v1" in names and "prod-v1" in names


def test_gc_runs(corpus_dir):
    runner = CliRunner()
    with runner.isolated_filesystem():
        _bootstrap(runner, str(corpus_dir), n=6)
        result = _run(runner, ["gc", "--keep", "5"])
        assert "Removed" in result.output


def test_snapshot_pin_survives_gc(corpus_dir):
    runner = CliRunner()
    with runner.isolated_filesystem():
        _bootstrap(runner, str(corpus_dir))
        first = _latest_snapshot()
        # two more snapshots so keep-last-1 would evict `first` without a pin
        for params in ('{"max_tokens": 40, "overlap": 0}', '{"max_tokens": 50, "overlap": 0}'):
            _run(runner, ["ingest", "--chunker", FIXED_TOKEN, "--chunk-params", params])
        _run(runner, ["snapshot", "pin", first])
        _run(runner, ["gc", "--keep", "1"])
        remaining = {m.snapshot_id for m in ProjectStore(".").list_snapshots()}
        assert first in remaining
        assert len(remaining) == 2  # pinned + latest

        _run(runner, ["snapshot", "unpin", first])
        _run(runner, ["gc", "--keep", "1"])
        remaining = {m.snapshot_id for m in ProjectStore(".").list_snapshots()}
        assert first not in remaining


# -- sweeps / migration / drift ----------------------------------------------


def test_sweep_hybrid_frontier(corpus_dir):
    runner = CliRunner()
    with runner.isolated_filesystem():
        ds_id = _bootstrap(runner, str(corpus_dir), n=10)
        snap = _latest_snapshot()
        result = _run(runner, ["sweep", "hybrid", "--dataset", ds_id,
                               "--snapshot", snap, "--grid", "0,0.5,1.0"])
        assert "Frontier" in result.output


def test_compare_chunkers_reports_fates(corpus_dir):
    runner = CliRunner()
    with runner.isolated_filesystem():
        ds_id = _bootstrap(runner, str(corpus_dir), n=10)
        from_cfg = json.dumps({"tool": "recall.chunkers.markdown_heading",
                               "params": {"max_tokens": 800, "overlap": 120}})
        to_cfg = json.dumps({"tool": FIXED_TOKEN, "params": {"max_tokens": 30, "overlap": 0}})
        result = _run(runner, ["compare-chunkers", "--from", from_cfg, "--to", to_cfg,
                               "--dataset", ds_id])
        assert "comparison report" in result.output.lower()
        assert "split" in result.output  # chunk-fate table


def test_compare_embeddings_local_zero_cost(corpus_dir):
    runner = CliRunner()
    with runner.isolated_filesystem():
        ds_id = _bootstrap(runner, str(corpus_dir), n=10)
        result = _run(runner, ["compare-embeddings", "--from", "local:hash-v1:256",
                               "--to", "local:hash-v1b:256", "--dataset", ds_id])
        assert "comparison report" in result.output.lower()
        assert "$0.00" in result.output


def test_compare_embeddings_gates_combined_cost(corpus_dir):
    # Finding #5: --max-cost must bound the WHOLE dual ingest, not each arm
    # independently. A budget that covers each arm alone but not their sum must be
    # rejected at the combined gate before any (network) embedding call.
    from recallops.cli import _estimate_embedding_cost, _load_config, _store
    runner = CliRunner()
    with runner.isolated_filesystem():
        ds_id = _bootstrap(runner, str(corpus_dir), n=10)
        cfg = _load_config("recall.yaml")
        store = _store()
        small = "openai:text-embedding-3-small:1536"
        large = "openai:text-embedding-3-large:3072"
        est_small = _estimate_embedding_cost(store, cfg, Path(cfg.source), small)
        est_large = _estimate_embedding_cost(store, cfg, Path(cfg.source), large)
        assert est_small > 0.0 and est_large > 0.0
        # budget covers the larger arm alone, but not both together.
        budget = max(est_small, est_large) + min(est_small, est_large) * 0.5
        result = runner.invoke(main, ["compare-embeddings", "--from", small, "--to", large,
                                      "--dataset", ds_id, "--max-cost", str(budget)])
        assert result.exit_code != 0
        out = result.output.lower()
        assert "cost" in out or "budget" in out or "approval" in out
        # blocked at the gate, never reaching an actual embed call
        assert "openai_api_key" not in out


def test_drift_surfaces_corpus_change(corpus_dir):
    runner = CliRunner()
    with runner.isolated_filesystem():
        shutil.copytree(corpus_dir, "docs")
        _run(runner, ["init", "--source", "docs"])
        _make_dense_only()
        _run(runner, ["ingest"])
        snap_a = _latest_snapshot()
        _run(runner, ["dataset", "generate", "--n", "8", "--name", "gold"])
        Path("docs/support/tickets-refund.md").write_text(
            "# Refund support tickets\n\nCustomers open refund support tickets when a "
            "prorated annual refund is delayed. Refund tickets reference the refund "
            "policy and the annual plan billing schedule repeatedly.\n"
        )
        _run(runner, ["ingest"])
        snap_b = _latest_snapshot()
        assert snap_b != snap_a
        result = _run(runner, ["drift", "--against", snap_a, "--dataset", "gold-v1",
                               "--snapshot", snap_b])
        assert "Diff" in result.output


# -- scorecard graceful degradation ------------------------------------------


def test_scorecard_missing_module_is_graceful():
    try:
        import recallops.scorecard  # noqa: F401
        pytest.skip("scorecard module present; absence path not exercisable")
    except ImportError:
        pass
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = _run(runner, ["scorecard"], code=1)
        assert "not available" in result.output


# -- empty-collection warning on live eval -----------------------------------


def test_live_eval_warns_on_empty_collection(corpus_dir):
    from recallops.config import ProjectConfig, build_adapter
    from recallops.ingest import collection_name

    runner = CliRunner()
    with runner.isolated_filesystem():
        ds = _bootstrap(runner, str(corpus_dir))
        store = ProjectStore(".")
        m = store.resolve_snapshot("latest")
        adapter = build_adapter(ProjectConfig.load("recall.yaml"), store)
        name = collection_name(m, store.project)
        dims = int(m.pipeline.stage("embed").params["dims"])
        adapter.drop(name)
        adapter.ensure_collection(name, dims)  # exists but 0 vectors
        adapter.close()

        result = _run(runner, ["eval", ds, "--live"])
        assert "0 vectors" in result.output
        assert "warning" in result.output.lower()


def test_live_eval_no_warning_when_collection_populated(corpus_dir):
    runner = CliRunner()
    with runner.isolated_filesystem():
        ds = _bootstrap(runner, str(corpus_dir))
        result = _run(runner, ["eval", ds, "--live"])
        assert "0 vectors" not in result.output


# -- LLM-backed dataset generation ---------------------------------------------


def test_dataset_generate_llm_path(corpus_dir, monkeypatch):
    from recallops import llm as llm_module

    def fake_post(url, body, headers, timeout=None):
        prompt = body["messages"][0]["content"]
        # unique per prompt so the generator's dedup keeps them all
        return {"choices": [{"message": {"content": f"What is covered by: {prompt[-48:]}?"}}]}

    monkeypatch.setattr(llm_module, "_post_json", fake_post)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    runner = CliRunner()
    with runner.isolated_filesystem():
        _bootstrap(runner, str(corpus_dir), n=5)

        # billed operation without approval: the cost gate must block
        result = runner.invoke(main, ["dataset", "generate", "--n", "3",
                                      "--llm", "openai", "--name", "lgold"])
        assert result.exit_code != 0
        assert "requires approval" in result.output

        result = _run(runner, ["dataset", "generate", "--n", "3", "--llm", "openai",
                               "--name", "lgold", "--yes"])
        assert "lgold-v1" in result.output
        ds = ProjectStore(".").get_dataset("lgold-v1")
        assert len(ds.cases) == 3
        assert all(c.question.startswith("What is covered by:") for c in ds.cases)
        assert all(c.origin == "synthetic" for c in ds.cases)
