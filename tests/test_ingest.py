from __future__ import annotations

from pathlib import Path

import pytest

from recallops import hashing
from recallops.adapters.local import LocalIndexAdapter
from recallops.ingest import (
    IngestReport,
    build_pipeline,
    chunkset_key,
    collection_name,
    ingest,
)
from recallops.models import PipelineDAG
from recallops.pipeline import chunkers
from recallops.store import ProjectStore

CONFIG = {
    "chunker": {"tool": chunkers.MARKDOWN_HEADING, "params": {"max_tokens": 40, "overlap": 8}},
    "embedding": {"provider": "local", "model": "hash-v1", "dims": 64,
                  "params": {"seed": 0, "ngram": [1, 2]}},
}


@pytest.fixture
def store(tmp_path: Path) -> ProjectStore:
    return ProjectStore(tmp_path / "project")


@pytest.fixture
def pipeline() -> PipelineDAG:
    return build_pipeline(CONFIG)


def test_build_pipeline_defaults():
    dag = build_pipeline({})
    dag.validate()
    assert [s.id for s in dag.stages] == ["parse", "chunk", "embed", "index", "retrieve"]
    assert dag.stage("parse").tool == "text-v1"
    assert dag.stage("chunk").tool == chunkers.MARKDOWN_HEADING
    embed = dag.stage("embed")
    assert embed.params["provider"] == "local"
    assert embed.params["dims"] == 256
    retrieve = dag.stage("retrieve")
    assert retrieve.params["top_k"] == 10
    assert retrieve.params["hybrid"]["fusion"] == "weighted"


def test_build_pipeline_overrides_and_rerank():
    dag = build_pipeline({
        **CONFIG,
        "retrieve": {"top_k": 5, "hybrid": None},
        "rerank": {"tool": "recall.rerankers.overlap", "params": {}, "top_n": 5},
    })
    dag.validate()
    assert "hybrid" not in dag.stage("retrieve").params
    assert dag.stage("retrieve").params["top_k"] == 5
    rerank = dag.stage("rerank")
    assert rerank is not None
    assert rerank.params["top_n"] == 5
    assert rerank.inputs == ("retrieve",)


def test_ingest_report_and_artifacts(store: ProjectStore, pipeline: PipelineDAG, small_corpus: Path):
    report = ingest(store, small_corpus, pipeline, adapter=None)
    assert isinstance(report, IngestReport)
    assert report.new_chunks > 0
    assert report.reused_chunks == 0
    assert report.embed_calls > 0
    manifest = report.manifest
    assert manifest.corpus.doc_count == 3
    assert manifest.corpus.chunk_count == report.new_chunks
    assert manifest.corpus.merkle_root.startswith("mr_")
    for uri in manifest.artifacts.values():
        assert not Path(uri).is_absolute()
        assert "\\" not in uri
    assert manifest.artifacts["chunks_uri"].startswith("artifacts/chunks/")
    assert manifest.artifacts["embeddings_uri"].startswith("artifacts/emb/")
    key = chunkset_key(manifest.corpus.merkle_root, pipeline.stage("parse"), pipeline.stage("chunk"))
    records = store.get_chunks(key)
    assert len(records) == report.new_chunks
    keys = [
        hashing.embedding_key(r.text_hash, "local", "hash-v1", 64,
                              hashing.params_hash({"dims": 64, "seed": 0, "ngram": [1, 2]}))
        for r in records
    ]
    assert store.missing_embedding_keys(keys) == []


def test_acceptance_reingest_identical(store: ProjectStore, pipeline: PipelineDAG, small_corpus: Path):
    first = ingest(store, small_corpus, pipeline, adapter=None)
    second = ingest(store, small_corpus, pipeline, adapter=None)
    assert second.manifest.to_json() == first.manifest.to_json()
    assert second.manifest.snapshot_id == first.manifest.snapshot_id
    assert second.embed_calls == 0
    assert second.new_chunks == 0
    assert second.reused_chunks == first.new_chunks


def test_acceptance_fusion_weight_change(store: ProjectStore, pipeline: PipelineDAG, small_corpus: Path):
    first = ingest(store, small_corpus, pipeline, adapter=None)
    changed = build_pipeline({
        **CONFIG,
        "retrieve": {"hybrid": {"sparse": "bm25", "fusion": "weighted", "bm25_weight": 0.9}},
    })
    second = ingest(store, small_corpus, changed, adapter=None)
    assert second.manifest.snapshot_id != first.manifest.snapshot_id
    assert second.embed_calls == 0
    assert second.new_chunks == 0
    assert second.reused_chunks == first.new_chunks
    assert second.manifest.artifacts["chunks_uri"] == first.manifest.artifacts["chunks_uri"]
    assert second.manifest.artifacts["embeddings_uri"] == first.manifest.artifacts["embeddings_uri"]


def test_parent_linkage(store: ProjectStore, pipeline: PipelineDAG, small_corpus: Path):
    first = ingest(store, small_corpus, pipeline, adapter=None)
    changed = build_pipeline({
        **CONFIG,
        "retrieve": {"hybrid": {"sparse": "bm25", "fusion": "rrf"}},
    })
    second = ingest(store, small_corpus, changed, adapter=None,
                    parent=first.manifest.snapshot_id)
    assert second.manifest.parent_snapshot == first.manifest.snapshot_id
    assert store.get_snapshot(second.manifest.snapshot_id).parent_snapshot == first.manifest.snapshot_id


def test_parse_tool_included_in_chunkset_key(store: ProjectStore, small_corpus: Path):
    text_pipeline = build_pipeline(CONFIG)
    md_pipeline = build_pipeline({**CONFIG, "parser": {"tool": "markdown-v2"}})
    first = ingest(store, small_corpus, text_pipeline, adapter=None)
    second = ingest(store, small_corpus, md_pipeline, adapter=None)
    assert second.manifest.corpus.merkle_root == first.manifest.corpus.merkle_root
    assert second.manifest.artifacts["chunks_uri"] != first.manifest.artifacts["chunks_uri"]
    assert second.new_chunks > 0
    key = chunkset_key(second.manifest.corpus.merkle_root,
                       md_pipeline.stage("parse"), md_pipeline.stage("chunk"))
    for record in store.get_chunks(key):
        assert not record.text.startswith("#")


def test_adapter_write_through(tmp_path: Path, store: ProjectStore, pipeline: PipelineDAG,
                               small_corpus: Path):
    adapter = LocalIndexAdapter(tmp_path / "index")
    report = ingest(store, small_corpus, pipeline, adapter=adapter)
    col = collection_name(report.manifest)
    assert col.startswith("col_")
    assert adapter.count(col) == report.manifest.corpus.chunk_count
    key = chunkset_key(report.manifest.corpus.merkle_root,
                       pipeline.stage("parse"), pipeline.stage("chunk"))
    records = store.get_chunks(key)
    chunk_ids = {r.chunk_id for r in records}
    keys = [
        hashing.embedding_key(r.text_hash, "local", "hash-v1", 64,
                              hashing.params_hash({"dims": 64, "seed": 0, "ngram": [1, 2]}))
        for r in records
    ]
    vec = store.get_embeddings(keys[:1])[keys[0]]
    hits = adapter.query_dense(col, vec, top_k=1)
    assert hits and hits[0][0] in chunk_ids


def test_collection_name_tracks_embed_and_index_factors(store: ProjectStore, small_corpus: Path):
    base = build_pipeline(CONFIG)
    reseeded = build_pipeline({
        **CONFIG,
        "embedding": {"provider": "local", "model": "hash-v1", "dims": 64,
                      "params": {"seed": 7, "ngram": [1, 2]}},
    })
    first = ingest(store, small_corpus, base, adapter=None)
    second = ingest(store, small_corpus, reseeded, adapter=None)
    assert collection_name(first.manifest) != collection_name(second.manifest)
    assert collection_name(first.manifest) == collection_name(
        ingest(store, small_corpus, base, adapter=None).manifest
    )


def test_collection_name_tracks_corpus_identity(store: ProjectStore, small_corpus: Path):
    # Two corpus versions under the same pipeline must occupy distinct
    # collections; sharing one lets stale chunks from the old corpus keep
    # serving after a re-ingest (upserts never delete).
    pipeline = build_pipeline(CONFIG)
    first = ingest(store, small_corpus, pipeline, adapter=None)
    (small_corpus / "beta.md").write_text("# Beta Gateway\n\nRewritten from scratch.\n")
    second = ingest(store, small_corpus, pipeline, adapter=None)
    assert second.manifest.corpus.merkle_root != first.manifest.corpus.merkle_root
    assert collection_name(first.manifest) != collection_name(second.manifest)


def test_live_collection_no_stale_chunks_after_corpus_edit(
        tmp_path: Path, store: ProjectStore, small_corpus: Path):
    adapter = LocalIndexAdapter(tmp_path / "index")
    pipeline = build_pipeline(CONFIG)
    ingest(store, small_corpus, pipeline, adapter=adapter)
    (small_corpus / "gamma.md").unlink()
    report = ingest(store, small_corpus, pipeline, adapter=adapter)
    col = collection_name(report.manifest)
    assert adapter.count(col) == report.manifest.corpus.chunk_count


def test_duplicate_content_files_chunked_once(tmp_path: Path, store: ProjectStore,
                                              small_corpus: Path):
    # Byte-identical files share one doc_id; the chunkset must hold each unique
    # chunk once and chunk_count must match what the adapter serves.
    (small_corpus / "zeta.md").write_text((small_corpus / "alpha.md").read_text())
    adapter = LocalIndexAdapter(tmp_path / "index")
    pipeline = build_pipeline(CONFIG)
    report = ingest(store, small_corpus, pipeline, adapter=adapter)

    assert report.manifest.corpus.doc_count == 4
    key = chunkset_key(report.manifest.corpus.merkle_root,
                       pipeline.stage("parse"), pipeline.stage("chunk"))
    ids = [r.chunk_id for r in store.get_chunks(key)]
    assert len(ids) == len(set(ids))
    assert report.manifest.corpus.chunk_count == len(ids)
    assert adapter.count(collection_name(report.manifest)) == report.manifest.corpus.chunk_count


def test_collection_name_namespace_isolates_projects(store: ProjectStore, small_corpus: Path):
    # Finding: two DIFFERENT projects sharing one vector DB with an identical
    # pipeline must get DIFFERENT collections (else their chunks collide in one
    # table). Empty namespace reproduces the un-namespaced name exactly.
    m = ingest(store, small_corpus, build_pipeline(CONFIG), adapter=None).manifest
    bare = collection_name(m)
    assert collection_name(m, "") == bare
    a = collection_name(m, "project-a")
    b = collection_name(m, "project-b")
    assert a != b != bare and a != bare
    assert collection_name(m, "project-a") == a  # deterministic


def test_collection_name_tracks_chunk_factor(store: ProjectStore, small_corpus: Path):
    # Two snapshots differing only in the chunker must occupy distinct serving
    # collections, else live-mode upserts mix their chunk sets.
    base = build_pipeline(CONFIG)
    rechunked = build_pipeline({
        **CONFIG,
        "chunker": {"tool": "recall.chunkers.fixed_token", "params": {"max_tokens": 30, "overlap": 0}},
    })
    a = ingest(store, small_corpus, base, adapter=None)
    b = ingest(store, small_corpus, rechunked, adapter=None)
    assert a.manifest.snapshot_id != b.manifest.snapshot_id
    assert collection_name(a.manifest) != collection_name(b.manifest)


def test_ingest_example_corpus(store: ProjectStore, pipeline: PipelineDAG, corpus_dir: Path):
    report = ingest(store, corpus_dir, pipeline, adapter=None)
    n_files = len(list(corpus_dir.rglob("*.md"))) + len(list(corpus_dir.rglob("*.txt")))
    assert report.manifest.corpus.doc_count == n_files >= 12
    assert report.new_chunks == report.manifest.corpus.chunk_count > n_files
    again = ingest(store, corpus_dir, pipeline, adapter=None)
    assert again.embed_calls == 0
    assert again.manifest.to_json() == report.manifest.to_json()
