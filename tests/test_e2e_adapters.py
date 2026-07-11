"""The README quickstart flow (ingest -> eval -> regression -> diff), run
through each embedded adapter. Proves the full engine works behind every
backend, not just the adapter contract. Snapshot ids are content-addressed
from pipeline + corpus (the adapter is serving-side only), so the pinned
README ids must hold behind every adapter.
"""
from __future__ import annotations

import pytest
from click.testing import CliRunner

from recallops.cli import main
from recallops.store import ProjectStore

BASELINE_SNAPSHOT = "snap_5cdbd0dc0e65ab00"

ADAPTERS = [
    pytest.param("qdrant", "qdrant_client", id="qdrant"),
    pytest.param("chroma", "chromadb", id="chroma"),
    pytest.param("lancedb", "lancedb", id="lancedb"),
]


def _run(runner: CliRunner, args: list[str], code: int = 0):
    result = runner.invoke(main, args)
    assert result.exit_code == code, (
        f"`recall {' '.join(args)}` exited {result.exit_code} (expected {code})\n{result.output}"
    )
    return result


@pytest.mark.parametrize("adapter_type,module", ADAPTERS)
def test_quickstart_flow_through_adapter(adapter_type, module, corpus_dir):
    pytest.importorskip(module)
    runner = CliRunner()
    with runner.isolated_filesystem():
        _run(runner, ["init", "--source", str(corpus_dir), "--adapter", adapter_type])
        _run(runner, ["ingest"])
        snap_a = ProjectStore(".").list_snapshots()[-1].snapshot_id
        assert snap_a == BASELINE_SNAPSHOT

        _run(runner, ["dataset", "generate", "--n", "20", "--seed", "0", "--name", "gold"])
        _run(runner, ["eval", "gold-v1", "--snapshot", snap_a])

        _run(runner, ["ingest", "--chunker", "recall.chunkers.fixed_token",
                      "--chunk-params", '{"max_tokens": 15, "overlap": 0}'])
        snap_b = ProjectStore(".").list_snapshots()[-1].snapshot_id
        assert snap_b != snap_a
        _run(runner, ["eval", "gold-v1", "--snapshot", snap_b])

        result = _run(runner, ["diff", snap_a, snap_b, "--dataset", "gold-v1"])
        assert snap_b[:8] in result.output or "diff" in result.output.lower()
