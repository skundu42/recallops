"""RecallOps, retrieval regression testing with verified root-cause attribution.

Public API (PRD §11). The heavier engine surfaces (diffing, funnel, ablation,
confirm, gating, report) are imported from their modules directly.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    # Single source of truth: the installed distribution metadata, which is
    # baked from the git tag at build time (hatch-vcs). See pyproject.toml.
    __version__ = _pkg_version("recallops")
except PackageNotFoundError:  # running from a source tree with no install
    __version__ = "0.0.0+unknown"

from .adapters.base import Capability, VectorAdapter
from .adapters.local import LocalIndexAdapter
from .dataset import generate as dataset_generate
from .evalrunner import evaluate
from .ingest import build_pipeline
from .models import (
    AttributionReport,
    GoldenCase,
    GoldenDataset,
    PipelineDAG,
    SnapshotManifest,
    StageSpec,
)
from .pipeline.providers import EmbeddingProvider, LocalHashProvider, get_provider
from .recorder import Recorder
from .retrieval import RetrievalEngine
from .store import ProjectStore

__all__ = [
    "__version__",
    "Recorder",
    "ProjectStore",
    "RetrievalEngine",
    "LocalIndexAdapter",
    "VectorAdapter",
    "Capability",
    "EmbeddingProvider",
    "LocalHashProvider",
    "get_provider",
    "build_pipeline",
    "evaluate",
    "dataset_generate",
    "PipelineDAG",
    "StageSpec",
    "SnapshotManifest",
    "GoldenCase",
    "GoldenDataset",
    "AttributionReport",
]
