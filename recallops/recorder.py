"""Recorder SDK (PRD §11, FR-2.2): bring-your-own-pipeline provenance.

The Recorder builds the same content-addressed provenance as managed ingest:
identical corpus + stage specs yield the same doc/chunk ids, the same chunkset
key (and therefore the same ``chunks_uri``), and the same snapshot core, so
SDK snapshots are directly diffable against managed-mode snapshots.

Stage specs are assembled into a linear ``PipelineDAG`` in ``stage()`` call
order; ``log_embeddings`` auto-registers an embed stage when the caller did
not declare one. ``log_retrieval`` buffers QueryRun-shaped candidates and
flushes them on ``commit()`` under ``retrieval_log:{snapshot_id}:{query_id}``
(the snapshot id only exists at commit time).
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import hashing
from .ingest import chunkset_key, embeddings_uri, parse_ph
from .models import (
    ChunkRecord,
    CorpusInfo,
    PipelineDAG,
    QueryRun,
    SnapshotManifest,
    StageCandidates,
    StageSpec,
)
from .pipeline.providers import embed_stage_spec, get_provider
from .store import ProjectStore

_DEFAULT_PARSE = StageSpec(id="parse", tool="external", version="")
_DEFAULT_CHUNK = StageSpec(id="chunk", tool="external", version="")


class Recorder:
    def __init__(self, project: str, store: str | Path):
        self.project = project
        self.store = ProjectStore(Path(store))
        self._stages: list[StageSpec] = []
        self._active: list[StageSpec] = []
        self._pairs: dict[str, str] = {}
        self._chunks: list[ChunkRecord] = []
        self._ordinals: dict[str, int] = {}
        self._chunk_ids_by_doc: dict[str, list[str]] = {}
        self._text_hash_by_chunk: dict[str, str] = {}
        self._emb_model_key: str | None = None
        self._retrieval: dict[str, dict[str, list[tuple[str, float]]]] = {}

    @contextmanager
    def stage(self, stage_id: str, tool: str, version: str = "",
              params: dict | None = None) -> Iterator[StageSpec]:
        spec = self._register(StageSpec(
            id=stage_id, tool=tool, version=version, params=dict(params or {}),
            inputs=(self._stages[-1].id,) if self._stages else (),
        ))
        self._active.append(spec)
        try:
            yield spec
        finally:
            self._active.pop()

    def _register(self, spec: StageSpec) -> StageSpec:
        existing = next((s for s in self._stages if s.id == spec.id), None)
        if existing is None:
            self._stages.append(spec)
            return spec
        if (existing.tool, existing.version, existing.params) != (spec.tool, spec.version, spec.params):
            raise ValueError(
                f"stage {spec.id!r} already recorded with a different tool/version/params"
            )
        return existing

    def _stage_for(self, stage_id: str, default: StageSpec) -> StageSpec:
        if self._active:
            return self._active[-1]
        return self._recorded(stage_id, default)

    def _recorded(self, stage_id: str, default: StageSpec) -> StageSpec:
        return next((s for s in self._stages if s.id == stage_id), default)

    def logged_chunks(self) -> list[ChunkRecord]:
        return list(self._chunks)

    def log_document(self, source_path: str, raw: bytes, parsed_text: str) -> str:
        stage = self._stage_for("parse", _DEFAULT_PARSE)
        doc_id = self.store.put_doc(source_path, raw, parsed_text, parse_ph(stage))
        self._pairs[source_path] = doc_id
        return doc_id

    def log_chunks(self, doc_id: str, chunks: list[dict]) -> list[str]:
        # Byte-identical files share one content-addressed doc_id; managed
        # ingest chunks each document once, so a repeat call for the same
        # doc_id returns the already-logged chunk ids rather than appending
        # duplicate records (which would double-count and desync the shared
        # chunkset from managed-mode content).
        if doc_id in self._chunk_ids_by_doc:
            return list(self._chunk_ids_by_doc[doc_id])
        chunk_stage = self._stage_for("chunk", _DEFAULT_CHUNK)
        parse_stage = self._recorded("parse", _DEFAULT_PARSE)
        chunk_ids: list[str] = []
        for chunk in chunks:
            start, end = (int(x) for x in chunk["span"])
            text = chunk["text"]
            ordinal = self._ordinals.get(doc_id, 0)
            self._ordinals[doc_id] = ordinal + 1
            record = ChunkRecord(
                chunk_id=hashing.chunk_id(doc_id, start, end, text),
                doc_id=doc_id,
                span_start=start,
                span_end=end,
                ordinal=ordinal,
                text=text,
                text_hash=hashing.text_hash(text),
                parse_stage_id=parse_stage.id,
                chunk_stage_id=chunk_stage.id,
            )
            self._chunks.append(record)
            self._text_hash_by_chunk[record.chunk_id] = record.text_hash
            chunk_ids.append(record.chunk_id)
        self._chunk_ids_by_doc[doc_id] = list(chunk_ids)
        return chunk_ids

    def log_embeddings(self, provider: str, model: str, chunk_ids: list[str],
                       vectors: np.ndarray, dims: int | None = None,
                       params: dict | None = None) -> None:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(chunk_ids):
            raise ValueError(
                f"vectors must be ({len(chunk_ids)}, dims), got shape {matrix.shape}"
            )
        dims = int(dims) if dims is not None else int(matrix.shape[1])
        params = dict(params or {})
        # Key the embeddings under the SAME canonical params RetrievalEngine
        # reconstructs (get_provider canonicalizes the param set, e.g. local ->
        # {dims, seed, ngram}). Using the raw params dict here would produce keys
        # that replay/eval can never find (finding #4). If an embed stage was
        # already declared explicitly, canonicalize from its params so ph tracks
        # whatever embed stage lands in the manifest.
        existing = next((s for s in self._stages if s.id == "embed"), None)
        candidate = embed_stage_spec(provider, model, dims, params)
        spec = existing or candidate

        def _canon(stage: StageSpec):
            try:
                return get_provider(dict(stage.params))
            except KeyError as exc:
                raise ValueError(
                    f"embed stage params {stage.params!r} cannot resolve an embedding "
                    f"provider (missing {exc}); declare the stage with provider/model/dims"
                ) from exc

        # Vectors are always keyed under the stage that lands in the manifest.
        canon = _canon(spec)
        # A snapshot records exactly one embed stage. Quietly keying this call's
        # vectors under a DIFFERENT already-recorded stage would drop them (the
        # old keys already exist, so the store writes nothing) while the manifest
        # claims the old model — mirror _register and refuse. Compare each value
        # this call states EXPLICITLY (candidate.params: provider/model/dims plus
        # any params it passed) against the existing stage's CANONICAL identity
        # (defaults filled in, e.g. seed=0/ngram=[1,2] for local), so a stated
        # value that contradicts a defaulted one is caught while a caller that
        # omits a declared stage's params is not flagged.
        existing_identity = {"provider": canon.provider, "model": canon.model, **canon.params}
        if existing is not None:
            conflicts = {
                k: (existing_identity[k], v)
                for k, v in candidate.params.items()
                if k in existing_identity and existing_identity[k] != v
            }
            if conflicts:
                raise ValueError(
                    f"log_embeddings(provider={provider!r}, model={model!r}, dims={dims}) "
                    f"conflicts with the already-recorded embed stage on {sorted(conflicts)} "
                    f"(recorded {existing.params!r})"
                )
        ph = hashing.params_hash(canon.params)
        model_key = canon.model_key
        embs: dict[str, np.ndarray] = {}
        for chunk_id, row in zip(chunk_ids, matrix):
            text_hash = self._text_hash_by_chunk[chunk_id]
            embs[hashing.embedding_key(text_hash, canon.provider, canon.model, canon.dims, ph)] = row
        self.store.put_embeddings(model_key, embs)
        self._emb_model_key = model_key
        if existing is None:
            self._stages.append(replace(
                spec, inputs=(self._stages[-1].id,) if self._stages else (),
            ))

    def log_retrieval(self, query_id: str, stage: str,
                      candidates: list[tuple[str, float]]) -> None:
        stages = self._retrieval.setdefault(query_id, {})
        stages[stage] = [(str(c), float(s)) for c, s in candidates]

    def _query_run(self, query_id: str, stages: dict[str, list[tuple[str, float]]]) -> QueryRun:
        reranked = stages.get("rerank", stages.get("reranked"))
        candidates = StageCandidates(
            dense=stages.get("dense", []),
            sparse=stages.get("sparse", []),
            fused=stages.get("fused", []),
            reranked=reranked,
        )
        if "final" in stages:
            final = stages["final"]
        elif reranked is not None:
            final = reranked
        else:
            final = candidates.fused or candidates.dense
        return QueryRun(query_id=query_id, question="", stages=candidates, final=final)

    def commit(self) -> str:
        dag = PipelineDAG(tuple(self._stages))
        dag.validate()
        pairs = list(self._pairs.items())
        merkle = hashing.merkle_root(pairs)

        parse_stage = self._recorded("parse", _DEFAULT_PARSE)
        chunk_stage = self._recorded("chunk", _DEFAULT_CHUNK)
        key = chunkset_key(merkle, parse_stage, chunk_stage)
        chunks_uri = self.store.put_chunks(key, self._chunks)

        artifacts = {"chunks_uri": chunks_uri}
        if self._emb_model_key is not None:
            artifacts["embeddings_uri"] = embeddings_uri(self._emb_model_key)
        corpus = CorpusInfo(doc_count=len(pairs), chunk_count=len(self._chunks),
                            merkle_root=merkle)
        manifest = SnapshotManifest.build(dag, corpus, artifacts)
        self.store.record_corpus_state(merkle, pairs)
        committed = self.store.commit_snapshot(manifest)

        for query_id, stages in sorted(self._retrieval.items()):
            run = self._query_run(query_id, stages)
            self.store.save_json(
                "retrieval_log", f"{committed.snapshot_id}:{query_id}", run.to_dict()
            )
        self._retrieval.clear()
        return committed.snapshot_id
