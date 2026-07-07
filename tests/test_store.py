from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from recallops import hashing
from recallops.models import (
    ChunkRecord,
    CorpusInfo,
    EvalResult,
    GoldenCase,
    GoldenDataset,
    PipelineDAG,
    QueryEval,
    SnapshotManifest,
    StageSpec,
)
from recallops.store import ProjectStore


def make_pipeline(chunk_params: dict | None = None) -> PipelineDAG:
    return PipelineDAG((
        StageSpec(id="parse", tool="text-v1", version="1"),
        StageSpec(id="chunk", tool="recall.chunkers.fixed_token", version="1",
                  params=chunk_params or {"max_tokens": 60, "overlap": 0}, inputs=("parse",)),
        StageSpec(id="embed", tool="local", version="1",
                  params={"model": "hash-v1", "dims": 8}, inputs=("chunk",)),
    ))


def make_manifest(chunk_params: dict | None = None, artifacts: dict[str, str] | None = None,
                  created_at: str = "2026-07-07T00:00:00Z",
                  parent: str | None = None) -> SnapshotManifest:
    pipeline = make_pipeline(chunk_params)
    corpus = CorpusInfo(doc_count=2, chunk_count=4, merkle_root="mr_0000000000000000")
    return SnapshotManifest.build(pipeline, corpus, artifacts or {}, parent=parent, created_at=created_at)


def make_chunks(doc: str, texts: list[str]) -> list[ChunkRecord]:
    records = []
    pos = 0
    for i, text in enumerate(texts):
        start, end = pos, pos + len(text)
        records.append(ChunkRecord(
            chunk_id=hashing.chunk_id(doc, start, end, text),
            doc_id=doc, span_start=start, span_end=end, ordinal=i,
            text=text, text_hash=hashing.text_hash(text),
            parse_stage_id="parse", chunk_stage_id="chunk",
        ))
        pos = end + 1
    return records


def make_eval(run_id: str, snapshot_id: str, dataset_id: str) -> EvalResult:
    qe = QueryEval(query_id="q1", ranked_chunks=[("ch_a", 0.9)], ranked_docs=["a.md"],
                   target_rank=1, hit_at={5: True}, metrics={"recall@5": 1.0})
    return EvalResult(run_id=run_id, snapshot_id=snapshot_id, dataset_id=dataset_id,
                      mode="replay", adapter="", created_at="2026-07-07T00:00:00Z",
                      k_values=(1, 5), per_query={"q1": qe}, aggregate={"recall@5": 1.0})


@pytest.fixture
def store(tmp_project: Path) -> ProjectStore:
    return ProjectStore(tmp_project)


def test_init_creates_layout(tmp_project: Path):
    ProjectStore(tmp_project)
    base = tmp_project / ".recall"
    assert (base / "db.sqlite").exists()
    for sub in ("chunks", "emb", "reports"):
        assert (base / "artifacts" / sub).is_dir()


def test_put_get_doc_roundtrip(store: ProjectStore):
    raw = b"# Alpha\n\nbody text\n"
    did = store.put_doc("docs/alpha.md", raw, "Alpha\n\nbody text", "ph_p1")
    assert did == hashing.doc_id(raw)
    got = store.get_doc(did)
    assert got == {"doc_id": did, "source_path": "docs/alpha.md",
                   "parsed_text": "Alpha\n\nbody text", "parse_ph": "ph_p1"}


def test_put_doc_idempotent(store: ProjectStore):
    raw = b"same bytes"
    d1 = store.put_doc("a.md", raw, "same bytes", "ph_p1")
    d2 = store.put_doc("a.md", raw, "same bytes", "ph_p1")
    assert d1 == d2


def test_get_doc_missing_raises(store: ProjectStore):
    with pytest.raises(KeyError):
        store.get_doc("doc_ffffffffffffffff")


def test_doc_by_source_hit_and_miss(store: ProjectStore):
    did = store.put_doc("b.md", b"bee", "bee", "ph_p1")
    hit = store.doc_by_source("b.md", "ph_p1")
    assert hit is not None and hit["doc_id"] == did and hit["parsed_text"] == "bee"
    assert store.doc_by_source("b.md", "ph_other") is None
    assert store.doc_by_source("missing.md", "ph_p1") is None


def test_docs_for_merkle_resolves_states(store: ProjectStore):
    a1 = store.put_doc("a.md", b"alpha v1", "alpha v1", "ph_p1")
    b1 = store.put_doc("b.md", b"beta v1", "beta v1", "ph_p1")
    merkle1 = hashing.merkle_root([("a.md", a1), ("b.md", b1)])

    store.put_doc("b.md", b"beta v1", "beta v1", "ph_p1")
    a2 = store.put_doc("a.md", b"alpha v2", "alpha v2", "ph_p1")
    c1 = store.put_doc("c.md", b"gamma v1", "gamma v1", "ph_p1")
    merkle2 = hashing.merkle_root([("a.md", a2), ("b.md", b1), ("c.md", c1)])

    docs1 = store.docs_for_merkle(merkle1)
    assert {(d["source_path"], d["doc_id"]) for d in docs1} == {("a.md", a1), ("b.md", b1)}
    docs2 = store.docs_for_merkle(merkle2)
    assert {(d["source_path"], d["doc_id"]) for d in docs2} == {("a.md", a2), ("b.md", b1), ("c.md", c1)}
    assert [d["source_path"] for d in docs2] == sorted(d["source_path"] for d in docs2)
    assert all("parsed_text" in d and "parse_ph" in d for d in docs1 + docs2)
    assert store.docs_for_merkle(merkle1) == docs1


def test_docs_for_merkle_unknown_returns_empty(store: ProjectStore):
    store.put_doc("a.md", b"alpha", "alpha", "ph_p1")
    assert store.docs_for_merkle("mr_ffffffffffffffff") == []


def test_project_namespace_persists_and_reloads(tmp_project):
    s = ProjectStore(tmp_project, project="support-rag")
    assert s.project == "support-rag"
    s.close()
    # a later open with no project argument reads it back from meta
    reopened = ProjectStore(tmp_project)
    assert reopened.project == "support-rag"
    # default (no project) is the empty namespace
    assert ProjectStore(tmp_project / "other").project == ""


def test_docs_for_merkle_resolves_a_deleted_document(store: ProjectStore):
    # Finding #1 (critical): a corpus that REMOVED a non-last document cannot be
    # reconstructed by monotone seq-prefix replay; record_corpus_state records it
    # authoritatively so docs_for_merkle resolves it.
    a = store.put_doc("aaa.md", b"alpha", "alpha", "ph_p1")
    z = store.put_doc("zzz.md", b"zeta", "zeta", "ph_p1")
    merkle_a = hashing.merkle_root([("aaa.md", a), ("zzz.md", z)])
    merkle_b = hashing.merkle_root([("zzz.md", z)])  # aaa.md deleted (not last-in-seq)
    store.record_corpus_state(merkle_a, [("aaa.md", a), ("zzz.md", z)])
    store.record_corpus_state(merkle_b, [("zzz.md", z)])

    docs_b = store.docs_for_merkle(merkle_b)
    assert {(d["source_path"], d["doc_id"]) for d in docs_b} == {("zzz.md", z)}
    docs_a = store.docs_for_merkle(merkle_a)
    assert {(d["source_path"], d["doc_id"]) for d in docs_a} == {("aaa.md", a), ("zzz.md", z)}


def test_chunks_roundtrip_exact(store: ProjectStore):
    records = make_chunks("doc_aaaaaaaaaaaaaaaa", ["first chunk text", "second chunk", "third"])
    uri = store.put_chunks("cs_key1", records)
    assert store.has_chunkset("cs_key1")
    assert (store.base / uri).exists()
    got = store.get_chunks("cs_key1")
    assert got == records


def test_put_chunks_idempotent(store: ProjectStore):
    records = make_chunks("doc_aaaaaaaaaaaaaaaa", ["one", "two"])
    uri1 = store.put_chunks("cs_key2", records)
    uri2 = store.put_chunks("cs_key2", records)
    assert uri1 == uri2


def test_get_chunks_missing_raises(store: ProjectStore):
    assert not store.has_chunkset("cs_nope")
    with pytest.raises(KeyError):
        store.get_chunks("cs_nope")


def test_empty_chunkset_roundtrip(store: ProjectStore):
    store.put_chunks("cs_empty", [])
    assert store.get_chunks("cs_empty") == []


def _emb_batch(model: str, n: int, dims: int = 8, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(n):
        key = hashing.embedding_key(f"tx_{model}{i:04d}", "local", model, dims, "ph_e1")
        out[key] = rng.uniform(-1.0, 1.0, dims).astype(np.float32)
    return out


def test_acceptance_embedding_cache_lookup(store: ProjectStore):
    embs = _emb_batch("hash-v1", 5)
    keys = list(embs)
    assert store.missing_embedding_keys(keys) == keys
    store.put_embeddings("local_hash-v1_8_ph_e1", embs)
    assert store.missing_embedding_keys(keys) == []


def test_fp16_roundtrip_within_tolerance(store: ProjectStore):
    embs = _emb_batch("hash-v1", 6, dims=16, seed=1)
    store.put_embeddings("local_hash-v1_16_ph_e1", embs)
    got = store.get_embeddings(list(embs))
    assert set(got) == set(embs)
    for key, vec in embs.items():
        out = got[key]
        assert out.dtype == np.float32
        assert out.shape == vec.shape
        assert np.max(np.abs(out - vec)) < 1e-3


def test_get_embeddings_only_found(store: ProjectStore):
    embs = _emb_batch("hash-v1", 3, seed=2)
    store.put_embeddings("local_hash-v1_8_ph_e1", embs)
    keys = list(embs)
    got = store.get_embeddings([keys[0], "emb_ffffffffffffffff", keys[2]])
    assert list(got) == [keys[0], keys[2]]


def test_missing_embedding_keys_preserves_order(store: ProjectStore):
    embs = _emb_batch("hash-v1", 4, seed=3)
    keys = list(embs)
    store.put_embeddings("mk", {keys[1]: embs[keys[1]], keys[3]: embs[keys[3]]})
    assert store.missing_embedding_keys(keys) == [keys[0], keys[2]]


def test_embeddings_across_parts(store: ProjectStore):
    part1 = _emb_batch("hash-v1", 3, seed=4)
    part2 = _emb_batch("hash-v2", 3, seed=5)
    store.put_embeddings("mk1", part1)
    store.put_embeddings("mk1", part2)
    all_keys = list(part1) + list(part2)
    got = store.get_embeddings(all_keys)
    assert list(got) == all_keys


def test_reput_same_keys_is_noop(store: ProjectStore):
    embs = _emb_batch("hash-v1", 3, seed=6)
    store.put_embeddings("mk1", embs)
    before = store.get_embeddings(list(embs))
    mutated = {k: v + 1.0 for k, v in embs.items()}
    store.put_embeddings("mk1", mutated)
    after = store.get_embeddings(list(embs))
    for k in embs:
        assert np.array_equal(before[k], after[k])


def test_embedding_batching_over_sql_limit(store: ProjectStore):
    rng = np.random.default_rng(9)
    embs = {
        hashing.embedding_key(f"tx_{i:04d}", "local", "hash-v1", 4, "ph_e1"):
            rng.uniform(-1.0, 1.0, 4).astype(np.float32)
        for i in range(520)
    }
    keys = list(embs)
    store.put_embeddings("mk_big", embs)
    assert store.missing_embedding_keys(keys) == []
    got = store.get_embeddings(keys)
    assert list(got) == keys


def test_get_embeddings_duplicate_input_keys(store: ProjectStore):
    embs = _emb_batch("hash-v1", 2, seed=8)
    store.put_embeddings("mk", embs)
    keys = list(embs)
    got = store.get_embeddings([keys[0], keys[1], keys[0]])
    assert list(got) == keys


def test_part_file_naming_deterministic(tmp_path: Path):
    embs = _emb_batch("hash-v1", 3, seed=20)
    uris = []
    for sub in ("p1", "p2"):
        s = ProjectStore(tmp_path / sub)
        s.put_embeddings("mk", embs)
        emb_dir = s.base / "artifacts" / "emb" / "mk"
        uris.append(sorted(p.name for p in emb_dir.glob("*.parquet")))
    assert uris[0] == uris[1] and len(uris[0]) == 1


def test_chunks_roundtrip_unicode(store: ProjectStore):
    records = make_chunks("doc_cccccccccccccccc", ["naïve, résumé ✓", "line\none\ttab", "日本語テキスト"])
    store.put_chunks("cs_uni", records)
    assert store.get_chunks("cs_uni") == records


def test_acceptance_idempotent_commit_byte_identical(store: ProjectStore):
    m1 = make_manifest(created_at="2026-07-07T00:00:00Z")
    committed1 = store.commit_snapshot(m1)
    m2 = make_manifest(created_at="2026-07-08T12:34:56Z", parent="snap_other")
    assert m2.snapshot_id == m1.snapshot_id
    committed2 = store.commit_snapshot(m2)
    assert committed2.to_json() == committed1.to_json()
    assert committed2.created_at == "2026-07-07T00:00:00Z"
    assert len(store.list_snapshots()) == 1


def test_commit_snapshot_invalid_id_atomic(store: ProjectStore):
    m = make_manifest()
    bad = SnapshotManifest(snapshot_id="snap_ffffffffffffffff", parent_snapshot=None,
                           created_at=m.created_at, pipeline=m.pipeline,
                           corpus=m.corpus, artifacts=m.artifacts)
    with pytest.raises(ValueError):
        store.commit_snapshot(bad)
    assert store.list_snapshots() == []


def test_commit_snapshot_invalid_pipeline_atomic(store: ProjectStore):
    pipeline = PipelineDAG((StageSpec(id="chunk", tool="t", inputs=("parse",)),))
    corpus = CorpusInfo(1, 1, "mr_0000000000000000")
    bad = SnapshotManifest.build(pipeline, corpus, {})
    with pytest.raises(ValueError):
        store.commit_snapshot(bad)
    assert store.list_snapshots() == []


def test_get_snapshot_roundtrip_and_missing(store: ProjectStore):
    m = store.commit_snapshot(make_manifest())
    assert store.get_snapshot(m.snapshot_id).to_json() == m.to_json()
    with pytest.raises(KeyError):
        store.get_snapshot("snap_ffffffffffffffff")


def test_list_snapshots_insertion_order(store: ProjectStore):
    m1 = store.commit_snapshot(make_manifest(chunk_params={"max_tokens": 60, "overlap": 0}))
    m2 = store.commit_snapshot(make_manifest(chunk_params={"max_tokens": 80, "overlap": 10}))
    assert [s.snapshot_id for s in store.list_snapshots()] == [m1.snapshot_id, m2.snapshot_id]


def test_resolve_snapshot(store: ProjectStore):
    m1 = store.commit_snapshot(make_manifest(chunk_params={"max_tokens": 60, "overlap": 0}))
    m2 = store.commit_snapshot(make_manifest(chunk_params={"max_tokens": 80, "overlap": 10}))
    assert store.resolve_snapshot("latest").snapshot_id == m2.snapshot_id
    assert store.resolve_snapshot(m1.snapshot_id).snapshot_id == m1.snapshot_id
    assert store.resolve_snapshot(m1.snapshot_id[:9]).snapshot_id == m1.snapshot_id
    bare = m2.snapshot_id.removeprefix("snap_")[:6]
    assert store.resolve_snapshot(bare).snapshot_id == m2.snapshot_id
    with pytest.raises(ValueError):
        store.resolve_snapshot("snap_")
    with pytest.raises(KeyError):
        store.resolve_snapshot("snap_zzzz")


def test_resolve_snapshot_latest_empty_raises(store: ProjectStore):
    with pytest.raises(KeyError):
        store.resolve_snapshot("latest")


def _dataset(dataset_id: str) -> GoldenDataset:
    case = GoldenCase(id=f"case_{dataset_id}", question="What is alpha?",
                      expected_sources=["a.md"], tags=["docs"], origin="synthetic")
    return GoldenDataset(dataset_id, [case])


def test_dataset_save_get_exact_and_latest(store: ProjectStore):
    for did in ("gold-v1", "gold-v3", "gold-v2"):
        store.save_dataset(_dataset(did))
    assert store.get_dataset("gold-v2").dataset_id == "gold-v2"
    assert store.get_dataset("gold").dataset_id == "gold-v3"
    assert store.get_dataset("gold").to_dict() == _dataset("gold-v3").to_dict()
    with pytest.raises(KeyError):
        store.get_dataset("silver")


def test_list_datasets(store: ProjectStore):
    store.save_dataset(_dataset("gold-v2"))
    store.save_dataset(_dataset("gold-v1"))
    store.save_dataset(_dataset("base-v1"))
    assert store.list_datasets() == ["base-v1", "gold-v1", "gold-v2"]


def test_eval_save_get_find(store: ProjectStore):
    ev = make_eval("ev_0000000000000001", "snap_a", "gold-v1")
    store.save_eval(ev)
    assert store.get_eval("ev_0000000000000001").to_dict() == ev.to_dict()
    found = store.find_eval("snap_a", "gold-v1")
    assert found is not None and found.run_id == ev.run_id
    ev2 = make_eval("ev_0000000000000002", "snap_a", "gold-v1")
    store.save_eval(ev2)
    assert store.find_eval("snap_a", "gold-v1").run_id == "ev_0000000000000002"


def test_find_eval_missing_none(store: ProjectStore):
    assert store.find_eval("snap_x", "gold-v1") is None


def test_get_eval_missing_raises(store: ProjectStore):
    with pytest.raises(KeyError):
        store.get_eval("ev_ffffffffffffffff")


def test_save_json_get_list(store: ProjectStore):
    store.save_json("diff", "diff_b", {"x": 1})
    store.save_json("diff", "diff_a", {"y": [1, 2]})
    store.save_json("job", "job_1", {"done": []})
    assert store.get_json("diff", "diff_a") == {"y": [1, 2]}
    assert store.get_json("diff", "diff_missing") is None
    assert store.list_json("diff") == ["diff_a", "diff_b"]
    assert store.list_json("calibration") == []
    store.save_json("diff", "diff_a", {"y": "replaced"})
    assert store.get_json("diff", "diff_a") == {"y": "replaced"}


def test_reopen_persistence(tmp_project: Path):
    s1 = ProjectStore(tmp_project)
    records = make_chunks("doc_aaaaaaaaaaaaaaaa", ["persist me"])
    s1.put_chunks("cs_p", records)
    embs = _emb_batch("hash-v1", 2, seed=7)
    s1.put_embeddings("mk", embs)
    m = s1.commit_snapshot(make_manifest())

    s2 = ProjectStore(tmp_project)
    assert s2.get_chunks("cs_p") == records
    assert s2.missing_embedding_keys(list(embs)) == []
    assert s2.get_snapshot(m.snapshot_id).to_json() == m.to_json()


def test_gc_keeps_pinned_and_last_n(store: ProjectStore):
    snaps = []
    for i, model in enumerate(("m1", "m2", "m3")):
        records = make_chunks(f"doc_{i:016d}", [f"chunk for {model}"])
        uri = store.put_chunks(f"cs_{model}", records)
        store.put_embeddings(model, _emb_batch(model, 2, seed=10 + i))
        manifest = make_manifest(
            chunk_params={"max_tokens": 60 + i, "overlap": 0},
            artifacts={"chunks": uri, "embeddings": f"artifacts/emb/{model}"},
        )
        snaps.append(store.commit_snapshot(manifest))

    result = store.gc(keep_last=1, pinned={snaps[0].snapshot_id})

    assert result == {"removed_chunksets": 1, "removed_emb_files": 1}
    assert store.has_chunkset("cs_m1") and store.has_chunkset("cs_m3")
    assert not store.has_chunkset("cs_m2")
    assert not (store.base / snaps[1].artifacts["chunks"]).exists()
    assert (store.base / snaps[0].artifacts["chunks"]).exists()
    assert (store.base / snaps[2].artifacts["chunks"]).exists()

    m1_keys = list(_emb_batch("m1", 2, seed=10))
    m2_keys = list(_emb_batch("m2", 2, seed=11))
    m3_keys = list(_emb_batch("m3", 2, seed=12))
    assert store.missing_embedding_keys(m1_keys) == []
    assert store.missing_embedding_keys(m3_keys) == []
    assert store.missing_embedding_keys(m2_keys) == m2_keys

    with pytest.raises(KeyError):
        store.get_snapshot(snaps[1].snapshot_id)
    assert [s.snapshot_id for s in store.list_snapshots()] == [
        snaps[0].snapshot_id, snaps[2].snapshot_id]


def test_gc_shared_artifact_kept(store: ProjectStore):
    records = make_chunks("doc_bbbbbbbbbbbbbbbb", ["shared"])
    uri = store.put_chunks("cs_shared", records)
    m1 = store.commit_snapshot(make_manifest(
        chunk_params={"max_tokens": 10, "overlap": 0}, artifacts={"chunks": uri}))
    m2 = store.commit_snapshot(make_manifest(
        chunk_params={"max_tokens": 20, "overlap": 0}, artifacts={"chunks": uri}))
    result = store.gc(keep_last=1)
    assert result["removed_chunksets"] == 0
    assert store.has_chunkset("cs_shared")
    with pytest.raises(KeyError):
        store.get_snapshot(m1.snapshot_id)
    assert store.get_snapshot(m2.snapshot_id).to_json() == m2.to_json()
