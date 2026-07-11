"""Executable pin of the README quickstart snapshot ids.

The README documents exact content-addressed ids for the 5-minute quickstart
(default config over examples/corpus). Any change that moves these ids is a
breaking change to documented behavior and must be intentional.
"""
from __future__ import annotations

from click.testing import CliRunner

from recallops.cli import main
from recallops.store import ProjectStore

BASELINE_SNAPSHOT = "snap_5cdbd0dc0e65ab00"
FIXED_TOKEN_SNAPSHOT = "snap_9c97b0630a2ea8e2"


def _run(runner: CliRunner, args: list[str]):
    result = runner.invoke(main, args)
    assert result.exit_code == 0, (
        f"`recall {' '.join(args)}` exited {result.exit_code}\n{result.output}"
    )
    return result


def test_readme_quickstart_snapshot_ids_are_pinned(corpus_dir):
    runner = CliRunner()
    with runner.isolated_filesystem():
        _run(runner, ["init", "--source", str(corpus_dir)])
        _run(runner, ["ingest"])
        assert ProjectStore(".").list_snapshots()[-1].snapshot_id == BASELINE_SNAPSHOT

        _run(runner, ["ingest", "--chunker", "recall.chunkers.fixed_token",
                      "--chunk-params", '{"max_tokens": 15, "overlap": 0}'])
        assert ProjectStore(".").list_snapshots()[-1].snapshot_id == FIXED_TOKEN_SNAPSHOT
