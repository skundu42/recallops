"""build_adapter dispatch for the reach-round adapter types.

Constructing these adapters performs no client import (lazy), so these
tests run without any extras installed.
"""
from __future__ import annotations

import pytest
from click.testing import CliRunner

from recallops.adapters.chroma import ChromaAdapter
from recallops.adapters.lancedb import LanceDBAdapter
from recallops.adapters.qdrant import QdrantAdapter
from recallops.cli import main
from recallops.config import ProjectConfig, build_adapter
from recallops.store import ProjectStore


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["init", "--source", "docs"])
    assert result.exit_code == 0, result.output
    return ProjectStore(".")


def _cfg(adapter_block: dict) -> ProjectConfig:
    cfg = ProjectConfig.default(project="p")
    cfg.adapter = adapter_block
    return cfg


def test_qdrant_defaults_to_embedded_path_under_store(store, monkeypatch):
    monkeypatch.delenv("RECALL_QDRANT_URL", raising=False)
    adapter = build_adapter(_cfg({"type": "qdrant"}), store)
    assert isinstance(adapter, QdrantAdapter)
    assert adapter.url is None
    assert adapter.path.endswith("index/qdrant") or adapter.path.endswith("index\\qdrant")


def test_qdrant_url_from_config_wins(store):
    adapter = build_adapter(_cfg({"type": "qdrant", "url": "http://localhost:6333"}), store)
    assert adapter.url == "http://localhost:6333"
    assert adapter.path is None


def test_qdrant_url_from_env(store, monkeypatch):
    monkeypatch.setenv("RECALL_QDRANT_URL", "http://qdrant.internal:6333")
    adapter = build_adapter(_cfg({"type": "qdrant"}), store)
    assert adapter.url == "http://qdrant.internal:6333"


def test_chroma_defaults_to_embedded_path_under_store(store):
    adapter = build_adapter(_cfg({"type": "chroma"}), store)
    assert isinstance(adapter, ChromaAdapter)
    assert "index" in adapter.path and "chroma" in adapter.path


def test_lancedb_defaults_to_embedded_path_under_store(store):
    adapter = build_adapter(_cfg({"type": "lancedb"}), store)
    assert isinstance(adapter, LanceDBAdapter)
    assert "index" in adapter.path and "lancedb" in adapter.path


def test_explicit_path_wins(store):
    adapter = build_adapter(_cfg({"type": "chroma", "path": "/x/y"}), store)
    assert adapter.path == "/x/y"


def test_unknown_type_error_lists_all_types(store):
    with pytest.raises(ValueError, match="qdrant"):
        build_adapter(_cfg({"type": "milvus"}), store)
    with pytest.raises(ValueError, match="lancedb"):
        build_adapter(_cfg({"type": "milvus"}), store)


@pytest.mark.parametrize("adapter_type", ["qdrant", "chroma", "lancedb"])
def test_init_accepts_new_adapter_types(tmp_path, monkeypatch, adapter_type):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["init", "--source", "docs", "--adapter", adapter_type])
    assert result.exit_code == 0, result.output
    cfg = ProjectConfig.load("recall.yaml")
    assert cfg.adapter["type"] == adapter_type
