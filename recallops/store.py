"""Content-addressed provenance store (PRD FR-1, FR-13).

SQLite holds metadata and small JSON artifacts; chunk sets and fp16 embeddings
live in Parquet files under ``root/.recall/artifacts``. Everything is keyed by
content hashes so identical inputs always resolve to existing artifacts
(FR-1.5) and snapshot commits are idempotent.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import hashing
from .models import ChunkRecord, EvalResult, GoldenDataset, SnapshotManifest

_SCHEMA = """
CREATE TABLE IF NOT EXISTS docs (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL,
    source_path TEXT NOT NULL,
    parse_ph TEXT NOT NULL,
    parsed_text TEXT NOT NULL,
    raw BLOB NOT NULL,
    UNIQUE (doc_id, source_path, parse_ph)
);
CREATE TABLE IF NOT EXISTS corpus_states (
    merkle TEXT NOT NULL,
    source_path TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    PRIMARY KEY (merkle, source_path, doc_id)
);
CREATE TABLE IF NOT EXISTS chunksets (
    chunkset_key TEXT PRIMARY KEY,
    uri TEXT NOT NULL,
    n INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS embeddings (
    key TEXT PRIMARY KEY,
    model_key TEXT NOT NULL,
    part_uri TEXT NOT NULL,
    dims INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshots (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id TEXT UNIQUE NOT NULL,
    manifest_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version INTEGER NOT NULL,
    body TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evals (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT UNIQUE NOT NULL,
    snapshot_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    body TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS json_blobs (
    kind TEXT NOT NULL,
    key TEXT NOT NULL,
    body TEXT NOT NULL,
    PRIMARY KEY (kind, key)
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_CHUNK_FIELDS = (
    ("chunk_id", pa.string()),
    ("doc_id", pa.string()),
    ("span_start", pa.int64()),
    ("span_end", pa.int64()),
    ("ordinal", pa.int64()),
    ("text", pa.string()),
    ("text_hash", pa.string()),
    ("parse_stage_id", pa.string()),
    ("chunk_stage_id", pa.string()),
)
_CHUNK_SCHEMA = pa.schema(list(_CHUNK_FIELDS))

_SQL_BATCH = 500


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "-", name)


def _write_parquet_atomic(table: pa.Table, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, path)


class ProjectStore:
    def __init__(self, root: Path, project: str = "") -> None:
        self.root = Path(root)
        self.base = self.root / ".recall"
        for sub in ("chunks", "emb", "reports"):
            (self.base / "artifacts" / sub).mkdir(parents=True, exist_ok=True)
        self.db_path = self.base / "db.sqlite"
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.executescript(_SCHEMA)
        # The project namespace qualifies serving-collection names so different
        # projects sharing one vector DB never collide (it does not enter any
        # content hash). Set once (at init); read from meta on every later open.
        if project:
            self.set_meta("project", project)
        self.project = project or self.get_meta("project") or ""
        # In-memory, per-process cache of query embeddings (keyed by
        # embedding_key). Eval/calibration/ablation loops re-query the same
        # golden questions many times; without this a network embedding provider
        # re-embeds each query on every pass, slow and re-billed. fp32, so it
        # never changes replay precision.
        self._query_vectors: dict = {}

    def query_vector_cached(self, key: str):
        return self._query_vectors.get(key)

    def cache_query_vector(self, key: str, vec) -> None:
        self._query_vectors[key] = vec

    def set_meta(self, key: str, value: str) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def get_meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def close(self) -> None:
        self._conn.close()

    def _path(self, uri: str) -> Path:
        return self.base / Path(uri)

    # -- docs -----------------------------------------------------------------

    def put_doc(self, source_path: str, raw: bytes, parsed_text: str, parse_ph: str) -> str:
        did = hashing.doc_id(raw)
        with self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO docs (doc_id, source_path, parse_ph, parsed_text, raw)"
                " VALUES (?, ?, ?, ?, ?)",
                (did, source_path, parse_ph, parsed_text, raw),
            )
        return did

    @staticmethod
    def _doc_dict(row: sqlite3.Row) -> dict:
        return {
            "doc_id": row["doc_id"],
            "source_path": row["source_path"],
            "parsed_text": row["parsed_text"],
            "parse_ph": row["parse_ph"],
        }

    def get_doc(self, doc_id: str) -> dict:
        row = self._conn.execute(
            "SELECT * FROM docs WHERE doc_id = ? ORDER BY seq DESC LIMIT 1", (doc_id,)
        ).fetchone()
        if row is None:
            raise KeyError(doc_id)
        return self._doc_dict(row)

    def doc_by_source(self, source_path: str, parse_ph: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM docs WHERE source_path = ? AND parse_ph = ? ORDER BY seq DESC LIMIT 1",
            (source_path, parse_ph),
        ).fetchone()
        return None if row is None else self._doc_dict(row)

    def docs_for_merkle(self, merkle: str) -> list[dict]:
        pairs = [
            (row["source_path"], row["doc_id"])
            for row in self._conn.execute(
                "SELECT source_path, doc_id FROM corpus_states WHERE merkle = ?"
                " ORDER BY source_path",
                (merkle,),
            )
        ]
        if not pairs:
            pairs = self._resolve_merkle(merkle)
            if pairs is None:
                return []
            with self._conn:
                self._conn.executemany(
                    "INSERT OR IGNORE INTO corpus_states (merkle, source_path, doc_id)"
                    " VALUES (?, ?, ?)",
                    [(merkle, sp, did) for sp, did in pairs],
                )
        return [self._doc_for_pair(sp, did) for sp, did in pairs]

    def record_corpus_state(self, merkle: str, pairs: list[tuple[str, str]]) -> None:
        """Record the exact (source_path, doc_id) set behind a merkle root.

        Ingest and the Recorder already hold these pairs, so recording them
        directly is authoritative, it makes ``docs_for_merkle`` resolve any
        corpus state (including one that *removed* a document), which the
        monotone seq-prefix reconstruction in ``_resolve_merkle`` cannot do.
        """
        with self._conn:
            self._conn.executemany(
                "INSERT OR IGNORE INTO corpus_states (merkle, source_path, doc_id)"
                " VALUES (?, ?, ?)",
                [(merkle, sp, did) for sp, did in pairs],
            )

    def _resolve_merkle(self, merkle: str) -> list[tuple[str, str]] | None:
        # Best-effort fallback for merkles recorded before record_corpus_state
        # existed. Reconstructs monotone add/update states only; a corpus that
        # dropped a document is authoritatively recorded at ingest time instead.
        if merkle == hashing.merkle_root([]):
            return []
        state: dict[str, str] = {}
        for row in self._conn.execute("SELECT source_path, doc_id FROM docs ORDER BY seq"):
            if state.get(row["source_path"]) == row["doc_id"]:
                continue
            state[row["source_path"]] = row["doc_id"]
            if hashing.merkle_root(list(state.items())) == merkle:
                return sorted(state.items())
        return None

    def _doc_for_pair(self, source_path: str, doc_id: str) -> dict:
        row = self._conn.execute(
            "SELECT * FROM docs WHERE source_path = ? AND doc_id = ? ORDER BY seq DESC LIMIT 1",
            (source_path, doc_id),
        ).fetchone()
        out = self._doc_dict(row)
        out["raw"] = row["raw"]
        return out

    # -- chunks ---------------------------------------------------------------

    def put_chunks(self, chunkset_key: str, records: list[ChunkRecord]) -> str:
        row = self._conn.execute(
            "SELECT uri FROM chunksets WHERE chunkset_key = ?", (chunkset_key,)
        ).fetchone()
        if row is not None:
            return row["uri"]
        uri = f"artifacts/chunks/{_safe_name(chunkset_key)}.parquet"
        table = pa.table(
            {name: [getattr(r, name) for r in records] for name, _ in _CHUNK_FIELDS},
            schema=_CHUNK_SCHEMA,
        )
        _write_parquet_atomic(table, self._path(uri))
        with self._conn:
            self._conn.execute(
                "INSERT INTO chunksets (chunkset_key, uri, n) VALUES (?, ?, ?)",
                (chunkset_key, uri, len(records)),
            )
        return uri

    def get_chunks(self, chunkset_key: str) -> list[ChunkRecord]:
        row = self._conn.execute(
            "SELECT uri FROM chunksets WHERE chunkset_key = ?", (chunkset_key,)
        ).fetchone()
        if row is None:
            raise KeyError(chunkset_key)
        table = pq.read_table(self._path(row["uri"]))
        return [ChunkRecord(**rec) for rec in table.to_pylist()]

    def has_chunkset(self, chunkset_key: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM chunksets WHERE chunkset_key = ?", (chunkset_key,)
        ).fetchone()
        return row is not None

    # -- embeddings (fp16 parquet parts + sqlite cache index, FR-1.4/FR-1.5) ---

    def _found_embedding_keys(self, keys: list[str]) -> set[str]:
        found: set[str] = set()
        uniq = list(dict.fromkeys(keys))
        for i in range(0, len(uniq), _SQL_BATCH):
            batch = uniq[i : i + _SQL_BATCH]
            marks = ",".join("?" * len(batch))
            rows = self._conn.execute(
                f"SELECT key FROM embeddings WHERE key IN ({marks})", batch
            ).fetchall()
            found.update(r["key"] for r in rows)
        return found

    def missing_embedding_keys(self, keys: list[str]) -> list[str]:
        found = self._found_embedding_keys(keys)
        return [k for k in dict.fromkeys(keys) if k not in found]

    def put_embeddings(self, model_key: str, embs: dict[str, np.ndarray]) -> None:
        keys = sorted(set(embs) - self._found_embedding_keys(list(embs)))
        if not keys:
            return
        matrix = np.stack([np.asarray(embs[k], dtype=np.float32) for k in keys])
        dims = int(matrix.shape[1])
        part_dir = f"artifacts/emb/{_safe_name(model_key)}"
        (self.base / part_dir).mkdir(parents=True, exist_ok=True)
        uri = f"{part_dir}/part_{hashing.h(*keys)}.parquet"
        flat = pa.array(matrix.astype(np.float16).reshape(-1), type=pa.float16())
        table = pa.table({
            "key": pa.array(keys, type=pa.string()),
            "vector": pa.FixedSizeListArray.from_arrays(flat, dims),
        })
        _write_parquet_atomic(table, self._path(uri))
        with self._conn:
            self._conn.executemany(
                "INSERT OR IGNORE INTO embeddings (key, model_key, part_uri, dims)"
                " VALUES (?, ?, ?, ?)",
                [(k, model_key, uri, dims) for k in keys],
            )

    def get_embeddings(self, keys: list[str]) -> dict[str, np.ndarray]:
        uniq = list(dict.fromkeys(keys))
        location: dict[str, str] = {}
        for i in range(0, len(uniq), _SQL_BATCH):
            batch = uniq[i : i + _SQL_BATCH]
            marks = ",".join("?" * len(batch))
            for row in self._conn.execute(
                f"SELECT key, part_uri FROM embeddings WHERE key IN ({marks})", batch
            ):
                location[row["key"]] = row["part_uri"]
        by_key: dict[str, np.ndarray] = {}
        for uri in sorted(set(location.values())):
            table = pq.read_table(self._path(uri))
            part_keys = table.column("key").to_pylist()
            vectors = table.column("vector").combine_chunks()
            dims = vectors.type.list_size
            matrix = np.asarray(vectors.flatten(), dtype=np.float16).reshape(-1, dims)
            wanted = {k for k, u in location.items() if u == uri}
            for j, k in enumerate(part_keys):
                if k in wanted:
                    by_key[k] = matrix[j].astype(np.float32)
        return {k: by_key[k] for k in uniq if k in by_key}

    # -- snapshots --------------------------------------------------------------

    def commit_snapshot(self, manifest: SnapshotManifest) -> SnapshotManifest:
        manifest.pipeline.validate()
        expected = hashing.snapshot_hash(manifest.core())
        if manifest.snapshot_id != expected:
            raise ValueError(
                f"snapshot_id {manifest.snapshot_id!r} does not match core hash {expected!r}"
            )
        row = self._conn.execute(
            "SELECT manifest_json FROM snapshots WHERE snapshot_id = ?",
            (manifest.snapshot_id,),
        ).fetchone()
        if row is not None:
            return SnapshotManifest.from_dict(json.loads(row["manifest_json"]))
        with self._conn:
            self._conn.execute(
                "INSERT INTO snapshots (snapshot_id, manifest_json) VALUES (?, ?)",
                (manifest.snapshot_id, manifest.to_json()),
            )
        return self.get_snapshot(manifest.snapshot_id)

    def get_snapshot(self, snapshot_id: str) -> SnapshotManifest:
        row = self._conn.execute(
            "SELECT manifest_json FROM snapshots WHERE snapshot_id = ?", (snapshot_id,)
        ).fetchone()
        if row is None:
            raise KeyError(snapshot_id)
        return SnapshotManifest.from_dict(json.loads(row["manifest_json"]))

    def list_snapshots(self) -> list[SnapshotManifest]:
        return [
            SnapshotManifest.from_dict(json.loads(row["manifest_json"]))
            for row in self._conn.execute("SELECT manifest_json FROM snapshots ORDER BY seq")
        ]

    def resolve_snapshot(self, prefix: str) -> SnapshotManifest:
        if prefix == "latest":
            row = self._conn.execute(
                "SELECT manifest_json FROM snapshots ORDER BY seq DESC LIMIT 1"
            ).fetchone()
            if row is None:
                raise KeyError("no snapshots committed yet")
            return SnapshotManifest.from_dict(json.loads(row["manifest_json"]))
        ids = [r["snapshot_id"] for r in self._conn.execute("SELECT snapshot_id FROM snapshots")]
        matches = sorted(
            sid for sid in ids if sid.startswith(prefix) or sid.startswith(f"snap_{prefix}")
        )
        if not matches:
            raise KeyError(prefix)
        if len(matches) > 1:
            raise ValueError(f"ambiguous snapshot prefix {prefix!r}: {matches}")
        return self.get_snapshot(matches[0])

    # -- datasets ---------------------------------------------------------------

    def save_dataset(self, ds: GoldenDataset) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO datasets (dataset_id, name, version, body)"
                " VALUES (?, ?, ?, ?)",
                (ds.dataset_id, ds.name, ds.version,
                 hashing.canonical_json(ds.to_dict()).decode("utf-8")),
            )

    def get_dataset(self, dataset_id: str) -> GoldenDataset:
        row = self._conn.execute(
            "SELECT body FROM datasets WHERE dataset_id = ?", (dataset_id,)
        ).fetchone()
        if row is None:
            row = self._conn.execute(
                "SELECT body FROM datasets WHERE name = ? ORDER BY version DESC LIMIT 1",
                (dataset_id,),
            ).fetchone()
        if row is None:
            raise KeyError(dataset_id)
        return GoldenDataset.from_dict(json.loads(row["body"]))

    def list_datasets(self) -> list[str]:
        return [
            row["dataset_id"]
            for row in self._conn.execute(
                "SELECT dataset_id FROM datasets ORDER BY name, version"
            )
        ]

    # -- evals --------------------------------------------------------------------

    def save_eval(self, ev: EvalResult) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO evals (run_id, snapshot_id, dataset_id, body)"
                " VALUES (?, ?, ?, ?)",
                (ev.run_id, ev.snapshot_id, ev.dataset_id,
                 hashing.canonical_json(ev.to_dict()).decode("utf-8")),
            )

    def get_eval(self, run_id: str) -> EvalResult:
        row = self._conn.execute(
            "SELECT body FROM evals WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise KeyError(run_id)
        return EvalResult.from_dict(json.loads(row["body"]))

    def find_eval(self, snapshot_id: str, dataset_id: str) -> EvalResult | None:
        row = self._conn.execute(
            "SELECT body FROM evals WHERE snapshot_id = ? AND dataset_id = ?"
            " ORDER BY seq DESC LIMIT 1",
            (snapshot_id, dataset_id),
        ).fetchone()
        return None if row is None else EvalResult.from_dict(json.loads(row["body"]))

    # -- generic JSON artifacts (diff/attribution/calibration/job/arm) -------------

    def save_json(self, kind: str, key: str, obj: dict) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO json_blobs (kind, key, body) VALUES (?, ?, ?)",
                (kind, key, hashing.canonical_json(obj).decode("utf-8")),
            )

    def get_json(self, kind: str, key: str) -> dict | None:
        row = self._conn.execute(
            "SELECT body FROM json_blobs WHERE kind = ? AND key = ?", (kind, key)
        ).fetchone()
        return None if row is None else json.loads(row["body"])

    def list_json(self, kind: str) -> list[str]:
        return [
            row["key"]
            for row in self._conn.execute(
                "SELECT key FROM json_blobs WHERE kind = ? ORDER BY key", (kind,)
            )
        ]

    # -- retention (FR-1.8) ---------------------------------------------------------

    def gc(self, keep_last: int, pinned: set[str] = frozenset()) -> dict:
        rows = self._conn.execute(
            "SELECT snapshot_id, manifest_json FROM snapshots ORDER BY seq"
        ).fetchall()
        start = max(0, len(rows) - keep_last)
        kept_ids = {r["snapshot_id"] for r in rows[start:]} if keep_last > 0 else set()
        kept_ids |= {r["snapshot_id"] for r in rows if r["snapshot_id"] in pinned}
        removed_rows = [r for r in rows if r["snapshot_id"] not in kept_ids]

        def artifact_uris(subset) -> set[str]:
            uris: set[str] = set()
            for r in subset:
                uris.update(json.loads(r["manifest_json"]).get("artifacts", {}).values())
            return uris

        kept_uris = artifact_uris(r for r in rows if r["snapshot_id"] in kept_ids)
        victims = artifact_uris(removed_rows) - kept_uris

        chunk_uris = sorted(u for u in victims if u.startswith("artifacts/chunks/"))
        emb_uris = sorted(u for u in victims if u.startswith("artifacts/emb/"))
        emb_part_uris = [p for uri in emb_uris for p in self._emb_part_uris(uri)]

        # Index rows go first, in one transaction: interrupted mid-gc, the
        # worst case is an orphan artifact file. The reverse order leaves rows
        # for deleted files — a snapshot listed without its artifacts, or an
        # embeddings row whose missing parquet makes later ingests skip
        # re-embedding and commit vector-less snapshots.
        with self._conn:
            self._conn.executemany(
                "DELETE FROM snapshots WHERE snapshot_id = ?",
                [(r["snapshot_id"],) for r in removed_rows],
            )
            self._conn.executemany(
                "DELETE FROM chunksets WHERE uri = ?", [(u,) for u in chunk_uris]
            )
            self._conn.executemany(
                "DELETE FROM embeddings WHERE part_uri = ?", [(u,) for u in emb_part_uris]
            )

        removed_chunksets = 0
        for uri in chunk_uris:
            path = self._path(uri)
            if path.exists():
                path.unlink()
                removed_chunksets += 1
        removed_emb_files = 0
        for part_uri in emb_part_uris:
            part_path = self._path(part_uri)
            if part_path.exists():
                part_path.unlink()
                removed_emb_files += 1
        for uri in emb_uris:
            path = self._path(uri)
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()
        return {"removed_chunksets": removed_chunksets, "removed_emb_files": removed_emb_files}

    def _emb_part_uris(self, uri: str) -> list[str]:
        path = self._path(uri)
        if path.is_dir():
            return [f"{uri}/{p.name}" for p in sorted(path.glob("*.parquet"))]
        return [uri] if path.exists() else []
