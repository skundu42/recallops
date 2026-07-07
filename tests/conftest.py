from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture(scope="session")
def corpus_dir() -> Path:
    return REPO_ROOT / "examples" / "corpus"


@pytest.fixture
def small_corpus(tmp_path: Path) -> Path:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "alpha.md").write_text(
        "# Alpha Widget\n\n## Setup\n\nInstall the alpha widget with the bootstrap "
        "command. The bootstrap command validates the manifest checksum.\n\n"
        "## Troubleshooting\n\nIf the checksum fails, regenerate the manifest and "
        "retry the bootstrap.\n"
    )
    (root / "beta.md").write_text(
        "# Beta Gateway\n\n## Routing\n\nThe beta gateway routes traffic by header "
        "affinity. Header affinity keeps sessions pinned to one backend.\n\n"
        "## Timeouts\n\nGateway timeouts default to thirty seconds and are "
        "configurable per route.\n"
    )
    (root / "gamma.md").write_text(
        "# Gamma Storage\n\n## Replication\n\nGamma storage replicates each object "
        "three times across zones. Replication lag alerts fire above five seconds.\n\n"
        "## Quotas\n\nStorage quotas are enforced per bucket with soft and hard limits.\n"
    )
    return root
