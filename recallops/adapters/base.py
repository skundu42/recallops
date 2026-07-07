"""Vector DB adapter contract (PRD FR-12).

The capability descriptor is load-bearing: it tells the funnel layer what the
adapter can expose so shadow re-scoring (FR-6.2) can fill the gaps.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Capability:
    name: str
    exposes_dense_scores: bool
    exposes_sparse: bool
    supports_rebuild: bool


class VectorAdapter(ABC):
    name: str = "abstract"

    @abstractmethod
    def capabilities(self) -> Capability: ...

    @abstractmethod
    def ensure_collection(self, collection: str, dims: int) -> None: ...

    @abstractmethod
    def upsert(self, collection: str, ids: list[str], vectors: np.ndarray, payloads: list[dict]) -> None: ...

    @abstractmethod
    def query_dense(self, collection: str, vector: np.ndarray, top_k: int) -> list[tuple[str, float]]: ...

    @abstractmethod
    def count(self, collection: str) -> int: ...

    @abstractmethod
    def drop(self, collection: str) -> None: ...

    def rebuild(self, collection: str, seed: int = 0) -> None:
        """Rebuild the index (calibration hook, FR-9.2). Default: no-op."""
