"""Structural checks for the composite GitHub Action. PyYAML is not a
runtime dependency; it is present transitively in the dev env (chromadb),
so these tests importorskip it and a bare install skips them."""
from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parent.parent


def _action() -> dict:
    return yaml.safe_load((REPO_ROOT / "action.yml").read_text(encoding="utf-8"))


def test_action_is_composite_with_expected_inputs_and_outputs():
    doc = _action()
    assert doc["runs"]["using"] == "composite"
    for key in ("phase", "python-version", "install", "config", "max-cost",
                "comment", "github-token", "baseline-cache-key", "save-baseline"):
        assert key in doc["inputs"], f"missing input: {key}"
    assert set(doc["outputs"]) >= {"diff-id", "gate"}


def test_every_run_step_declares_bash_and_comment_uses_pr_comment_module():
    doc = _action()
    steps = doc["runs"]["steps"]
    run_steps = [s for s in steps if "run" in s]
    assert run_steps, "composite action has no run steps"
    assert all(s.get("shell") == "bash" for s in run_steps)
    joined = "\n".join(s["run"] for s in run_steps)
    assert "python -m recallops.pr_comment" in joined
    assert "recall ci" in joined
    assert "recall attribute" in joined


def test_example_workflow_uses_the_action():
    doc = yaml.safe_load(
        (REPO_ROOT / "examples" / "github-action" / "recall-ci.yml").read_text(encoding="utf-8")
    )
    jobs = doc["jobs"]
    assert {"gate", "attribute"} <= set(jobs)
    gate_uses = [s.get("uses", "") for s in jobs["gate"]["steps"]]
    assert any(u.startswith("skundu42/recallops@") for u in gate_uses)
    assert jobs["attribute"]["needs"] == "gate"
    assert jobs["attribute"]["continue-on-error"] is True
