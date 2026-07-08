"""Managed ingestion (PRD FR-1 acceptance, FR-2.1).

Provenance keys:

- ``chunkset_key = h(merkle_root, parse_tool, parse_params_hash, chunk_tool,
  chunk_params_hash)``. The parse *tool* is included on top of the planned
  formula because the corpus merkle root hashes raw bytes and the built-in
  parse stages carry empty params, without the tool, ``text-v1`` and
  ``markdown-v2`` ingests of the same corpus would collide on one chunkset key
  while producing different parsed text.
- The per-doc ``parse_ph`` stored with each document hashes the full parse
  stage identity (tool, version, params) for the same reason: the docs table
  is unique on ``(doc_id, source_path, parse_ph)`` and must keep one parsed
  text per parse configuration.
- ``collection_name(manifest)`` derives the serving collection from the corpus
  merkle root plus the parse/chunk/embed/index stage dicts, so every distinct
  chunk set gets its own collection while retrieval-stage changes never force
  an index rebuild.

Manifests are fully deterministic (``created_at`` is left empty) so identical
inputs produce byte-identical manifests on any machine.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import hashing
from .adapters.base import VectorAdapter
from .models import ChunkRecord, CorpusInfo, PipelineDAG, SnapshotManifest, StageSpec
from .pipeline import chunkers, parsers
from .pipeline.providers import EmbeddingProvider, embed_stage_spec, get_provider
from .store import ProjectStore, _safe_name

SOURCE_SUFFIXES = (".md", ".txt")

DEFAULT_CONFIG: dict = {
    "parser": {"tool": "text-v1"},
    "chunker": {"tool": chunkers.MARKDOWN_HEADING, "params": {"max_tokens": 800, "overlap": 120}},
    "embedding": {"provider": "local", "model": "hash-v1", "dims": 256,
                  "params": {"seed": 0, "ngram": [1, 2]}},
    "index": {"adapter": "local"},
    "retrieve": {"top_k": 10,
                 "hybrid": {"sparse": "bm25", "fusion": "weighted", "bm25_weight": 0.45}},
}


@dataclass
class IngestReport:
    manifest: SnapshotManifest
    embed_calls: int
    new_chunks: int
    reused_chunks: int


def parse_ph(parse_stage: StageSpec) -> str:
    return hashing.params_hash({
        "tool": parse_stage.tool,
        "version": parse_stage.version,
        "params": parse_stage.params,
    })


def chunkset_key(merkle: str, parse_stage: StageSpec, chunk_stage: StageSpec) -> str:
    return hashing.h(
        merkle,
        parse_stage.tool,
        parse_stage.params_hash,
        chunk_stage.tool,
        chunk_stage.params_hash,
    )


def collection_name(manifest: SnapshotManifest, namespace: str = "") -> str:
    # The served chunk set is fully determined by the corpus (merkle root),
    # parse+chunk (which chunks exist) and embed (their vectors); index carries
    # the serving params. Two snapshots that differ in any of these must occupy
    # distinct collections so live-mode upserts never mix chunk sets — without
    # the merkle root, re-ingesting an edited corpus under the same pipeline
    # would keep serving chunks of deleted documents (upserts never delete).
    #
    # ``namespace`` (the project id) qualifies the name so that two DIFFERENT
    # projects sharing one vector DB with an identical pipeline never collide in
    # the same table. Empty namespace reproduces the un-namespaced name exactly,
    # so single-project and local-adapter deployments are unaffected.
    stages = [manifest.pipeline.stage(s) for s in ("parse", "chunk", "embed", "index")]
    parts = ([namespace] if namespace else []) + [manifest.corpus.merkle_root] + [
        hashing.canonical_json(s.to_dict() if s else None) for s in stages
    ]
    return "col_" + hashing.h(*parts)


def embeddings_uri(model_key: str) -> str:
    return f"artifacts/emb/{_safe_name(model_key)}"


def _section(config: dict, key: str) -> dict:
    return {**DEFAULT_CONFIG[key], **config.get(key, {})}


def build_pipeline(config: dict) -> PipelineDAG:
    """Build the managed-mode DAG from a config dict.

    Sections and defaults (each section shallow-merged over its default, so
    overriding ``chunker.tool`` means supplying ``chunker.params`` too):

    - ``parser``: ``{"tool": "text-v1"}``
    - ``chunker``: ``{"tool": markdown_heading, "params": {"max_tokens": 800, "overlap": 120}}``
    - ``embedding``: ``{"provider": "local", "model": "hash-v1", "dims": 256,
      "params": {"seed": 0, "ngram": [1, 2]}}``
    - ``index``: ``{"adapter": "local"}`` (optional ``collection``)
    - ``retrieve``: ``{"top_k": 10, "hybrid": {...}}``; ``"hybrid": None`` selects dense-only
    - ``rerank``: absent by default; ``{"tool", "params", "top_n"}`` when given
    """
    parser_cfg = _section(config, "parser")
    chunker_cfg = _section(config, "chunker")
    embed_cfg = _section(config, "embedding")
    index_cfg = _section(config, "index")
    retrieve_cfg = _section(config, "retrieve")

    stages = [
        parsers.parser_stage_spec(parser_cfg["tool"]),
        StageSpec(id="chunk", tool=chunker_cfg["tool"], version="1",
                  params=dict(chunker_cfg.get("params", {})), inputs=("parse",)),
        embed_stage_spec(embed_cfg["provider"], embed_cfg["model"], int(embed_cfg["dims"]),
                         dict(embed_cfg.get("params", {}))),
        StageSpec(id="index", tool=index_cfg["adapter"], version="1",
                  params={k: v for k, v in index_cfg.items() if k != "adapter" and v is not None},
                  inputs=("embed",)),
    ]
    retrieve_params: dict = {"top_k": int(retrieve_cfg.get("top_k", 10))}
    if retrieve_cfg.get("hybrid") is not None:
        retrieve_params["hybrid"] = dict(retrieve_cfg["hybrid"])
    stages.append(StageSpec(id="retrieve", tool="recall.retrieve", version="1",
                            params=retrieve_params, inputs=("index",)))
    rerank_cfg = config.get("rerank")
    if rerank_cfg:
        stages.append(StageSpec(
            id="rerank",
            tool=rerank_cfg.get("tool", "recall.rerankers.overlap"),
            version="1",
            params={"top_n": int(rerank_cfg.get("top_n", 10)), **rerank_cfg.get("params", {})},
            inputs=("retrieve",),
        ))
    dag = PipelineDAG(tuple(stages))
    dag.validate()
    return dag


def _source_files(source_dir: Path) -> list[tuple[str, Path]]:
    return sorted(
        ((p.relative_to(source_dir).as_posix(), p)
         for p in source_dir.rglob("*")
         if p.is_file() and p.suffix in SOURCE_SUFFIXES),
        key=lambda item: item[0],
    )


def _require_stage(pipeline: PipelineDAG, stage_id: str) -> StageSpec:
    stage = pipeline.stage(stage_id)
    if stage is None:
        raise ValueError(f"pipeline is missing required stage {stage_id!r}")
    return stage


def embedding_keys(records: list[ChunkRecord], provider: EmbeddingProvider) -> list[str]:
    ph = hashing.params_hash(provider.params)
    return [
        hashing.embedding_key(r.text_hash, provider.provider, provider.model, provider.dims, ph)
        for r in records
    ]


def ingest(store: ProjectStore, source_dir: Path, pipeline: PipelineDAG,
           adapter: VectorAdapter | None, parent: str | None = None,
           provider: EmbeddingProvider | None = None) -> IngestReport:
    pipeline.validate()
    parse_stage = _require_stage(pipeline, "parse")
    chunk_stage = _require_stage(pipeline, "chunk")
    embed_stage = _require_stage(pipeline, "embed")
    source_dir = Path(source_dir)

    doc_ph = parse_ph(parse_stage)
    pairs: list[tuple[str, str]] = []
    parsed_docs: list[tuple[str, str]] = []
    for rel, path in _source_files(source_dir):
        raw = path.read_bytes()
        parsed = parsers.parse(rel, raw, tool=parse_stage.tool)
        doc_id = store.put_doc(rel, raw, parsed.text, doc_ph)
        pairs.append((rel, doc_id))
        parsed_docs.append((doc_id, parsed.text))
    merkle = hashing.merkle_root(pairs)

    key = chunkset_key(merkle, parse_stage, chunk_stage)
    if store.has_chunkset(key):
        records = store.get_chunks(key)
        new_chunks, reused_chunks = 0, len(records)
    else:
        records = []
        chunked: set[str] = set()
        for doc_id, text in parsed_docs:
            # Byte-identical files share one content-addressed doc_id and would
            # otherwise emit the same chunk records once per path.
            if doc_id in chunked:
                continue
            chunked.add(doc_id)
            records.extend(chunkers.chunk_doc(
                doc_id, text, chunk_stage.tool, chunk_stage.params,
                parse_stage.id, chunk_stage.id,
            ))
        new_chunks, reused_chunks = len(records), 0
    chunks_uri = store.put_chunks(key, records)

    if provider is None:
        provider = get_provider(dict(embed_stage.params))
    keys = embedding_keys(records, provider)
    missing = store.missing_embedding_keys(keys)
    embed_calls = len(missing)
    if missing:
        text_by_key: dict[str, str] = {}
        for record, k in zip(records, keys):
            text_by_key.setdefault(k, record.text)
        vectors = provider.embed([text_by_key[k] for k in missing])
        store.put_embeddings(provider.model_key, {k: vectors[i] for i, k in enumerate(missing)})

    artifacts = {
        "chunks_uri": chunks_uri,
        "embeddings_uri": embeddings_uri(provider.model_key),
    }
    corpus = CorpusInfo(doc_count=len(pairs), chunk_count=len(records), merkle_root=merkle)
    manifest = SnapshotManifest.build(pipeline, corpus, artifacts, parent=parent)

    if adapter is not None:
        collection = collection_name(manifest, store.project)
        adapter.ensure_collection(collection, provider.dims)
        if records:
            found = store.get_embeddings(keys)
            # pairs is sorted by path; keep the first path as the canonical
            # source for documents that exist at several byte-identical paths.
            source_by_doc: dict[str, str] = {}
            for rel, doc_id in pairs:
                source_by_doc.setdefault(doc_id, rel)
            adapter.upsert(
                collection,
                [r.chunk_id for r in records],
                np.stack([found[k] for k in keys]).astype(np.float32),
                [{"doc_id": r.doc_id, "source_path": source_by_doc[r.doc_id],
                  "ordinal": r.ordinal} for r in records],
            )

    store.record_corpus_state(merkle, pairs)
    committed = store.commit_snapshot(manifest)
    return IngestReport(manifest=committed, embed_calls=embed_calls,
                        new_chunks=new_chunks, reused_chunks=reused_chunks)
