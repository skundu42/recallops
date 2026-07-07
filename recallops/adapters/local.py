"""Built-in exact-KNN vector adapter (PRD FR-12).

Each collection persists as an npz (ids + fp32 L2-normalized matrix) under
``root/collections/``. Queries are exact cosine via ``matrix @ vector``;
``ann_mode`` overlays seeded gaussian score noise to simulate ANN/quantization
variance for calibration (FR-9.2) and divergence detection (FR-6.3).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import hashing
from .base import Capability, VectorAdapter


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return mat / norms


def _noise_seed(seed: int, collection: str) -> int:
    return int(hashing.h(str(seed), collection), 16)


@dataclass
class _Collection:
    ids: list[str]
    matrix: np.ndarray
    payloads: list[dict]

    @property
    def dims(self) -> int:
        return int(self.matrix.shape[1])


class LocalIndexAdapter(VectorAdapter):
    name = "local"

    def __init__(self, root: Path, ann_mode: bool = False, ann_sigma: float = 0.01, seed: int = 0):
        self.root = Path(root)
        self.ann_mode = ann_mode
        self.ann_sigma = ann_sigma
        self.seed = seed
        self._collections_dir = self.root / "collections"
        self._collections_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, _Collection] = {}
        self._noise_seeds: dict[str, int] = {}
        self._rngs: dict[str, np.random.Generator] = {}

    def capabilities(self) -> Capability:
        return Capability(
            name=self.name,
            exposes_dense_scores=True,
            exposes_sparse=False,
            supports_rebuild=True,
        )

    def ensure_collection(self, collection: str, dims: int) -> None:
        col = self._load(collection)
        if col is not None:
            if col.dims != dims:
                raise ValueError(
                    f"collection {collection!r} has dims={col.dims}, requested dims={dims}"
                )
            return
        col = _Collection(ids=[], matrix=np.zeros((0, dims), dtype=np.float32), payloads=[])
        self._cache[collection] = col
        self._save(collection, col)

    def upsert(self, collection: str, ids: list[str], vectors: np.ndarray, payloads: list[dict]) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2-D, got shape {vectors.shape}")
        if len(ids) != vectors.shape[0] or len(ids) != len(payloads):
            raise ValueError(
                f"length mismatch: {len(ids)} ids, {vectors.shape[0]} vectors, {len(payloads)} payloads"
            )
        col = self._load(collection)
        if col is None:
            self.ensure_collection(collection, int(vectors.shape[1]))
            col = self._cache[collection]
        if vectors.shape[1] != col.dims:
            raise ValueError(
                f"collection {collection!r} has dims={col.dims}, got vectors with dims={vectors.shape[1]}"
            )
        normalized = _l2_normalize(vectors)
        merged_ids = list(col.ids)
        rows = [col.matrix[i] for i in range(len(merged_ids))]
        merged_payloads = list(col.payloads)
        index = {cid: i for i, cid in enumerate(merged_ids)}
        for cid, row, payload in zip(ids, normalized, payloads):
            if cid in index:
                pos = index[cid]
                rows[pos] = row
                merged_payloads[pos] = payload
            else:
                index[cid] = len(merged_ids)
                merged_ids.append(cid)
                rows.append(row)
                merged_payloads.append(payload)
        matrix = np.vstack(rows).astype(np.float32) if rows else np.zeros((0, col.dims), dtype=np.float32)
        updated = _Collection(ids=merged_ids, matrix=matrix, payloads=merged_payloads)
        self._cache[collection] = updated
        self._save(collection, updated)

    def query_dense(self, collection: str, vector: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        col = self._require(collection)
        if not col.ids or top_k <= 0:
            return []
        query = _l2_normalize(np.asarray(vector, dtype=np.float32).reshape(-1))
        scores = col.matrix @ query
        if self.ann_mode:
            noise = self._rng(collection).normal(0.0, self.ann_sigma, size=scores.shape)
            scores = scores.astype(np.float64) + noise
        order = sorted(range(len(col.ids)), key=lambda i: (-scores[i], col.ids[i]))
        return [(col.ids[i], float(scores[i])) for i in order[:top_k]]

    def count(self, collection: str) -> int:
        col = self._load(collection)
        return 0 if col is None else len(col.ids)

    def drop(self, collection: str) -> None:
        self._cache.pop(collection, None)
        self._rngs.pop(collection, None)
        self._noise_seeds.pop(collection, None)
        path = self._path(collection)
        if path.exists():
            path.unlink()

    def rebuild(self, collection: str, seed: int = 0) -> None:
        if not self.ann_mode:
            return
        self._noise_seeds[collection] = seed
        self._rngs[collection] = np.random.default_rng(_noise_seed(seed, collection))

    def _rng(self, collection: str) -> np.random.Generator:
        if collection not in self._rngs:
            seed = self._noise_seeds.get(collection, self.seed)
            self._rngs[collection] = np.random.default_rng(_noise_seed(seed, collection))
        return self._rngs[collection]

    def _path(self, collection: str) -> Path:
        if not collection or "/" in collection or "\\" in collection or collection in {".", ".."}:
            raise ValueError(f"invalid collection name: {collection!r}")
        return self._collections_dir / f"{collection}.npz"

    def _require(self, collection: str) -> _Collection:
        col = self._load(collection)
        if col is None:
            raise KeyError(collection)
        return col

    def _load(self, collection: str) -> _Collection | None:
        if collection in self._cache:
            return self._cache[collection]
        path = self._path(collection)
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as data:
            ids = [str(x) for x in data["ids"]]
            matrix = np.asarray(data["vectors"], dtype=np.float32)
            payloads = [json.loads(str(p)) for p in data["payloads"]]
        col = _Collection(ids=ids, matrix=matrix, payloads=payloads)
        self._cache[collection] = col
        return col

    def _save(self, collection: str, col: _Collection) -> None:
        path = self._path(collection)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as fh:
            np.savez(
                fh,
                ids=np.array(col.ids, dtype=np.str_),
                vectors=col.matrix,
                payloads=np.array(
                    [hashing.canonical_json(p).decode("utf-8") for p in col.payloads],
                    dtype=np.str_,
                ),
            )
        os.replace(tmp, path)
