from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from recallops import hashing
from recallops.ingest import build_pipeline, chunkset_key, ingest
from recallops.models import QueryRun
from recallops.pipeline import chunkers, parsers
from recallops.pipeline.providers import LocalHashProvider
from recallops.recorder import Recorder
from recallops.store import ProjectStore

CONFIG = {
    "chunker": {"tool": chunkers.MARKDOWN_HEADING, "params": {"max_tokens": 40, "overlap": 8}},
    "embedding": {"provider": "local", "model": "hash-v1", "dims": 64,
                  "params": {"seed": 0, "ngram": [1, 2]}},
}


def _sorted_files(corpus: Path) -> list[tuple[str, bytes]]:
    files = sorted(
        (p for p in corpus.rglob("*") if p.is_file() and p.suffix in (".md", ".txt")),
        key=lambda p: p.relative_to(corpus).as_posix(),
    )
    return [(p.relative_to(corpus).as_posix(), p.read_bytes()) for p in files]


def _record_corpus(rec: Recorder, corpus: Path) -> list[str]:
    pipeline = build_pipeline(CONFIG)
    chunk_stage = pipeline.stage("chunk")
    docs: list[tuple[str, str]] = []
    with rec.stage("parse", tool="text-v1", version="1"):
        for rel, raw in _sorted_files(corpus):
            parsed = parsers.parse(rel, raw, tool="text-v1")
            doc_id = rec.log_document(rel, raw, parsed.text)
            docs.append((doc_id, parsed.text))
    chunk_ids: list[str] = []
    chunker = chunkers.get_chunker(chunk_stage.tool, chunk_stage.params)
    with rec.stage("chunk", tool=chunk_stage.tool, version="1", params=chunk_stage.params):
        for doc_id, text in docs:
            spans = chunker.chunk(text)
            chunk_ids.extend(rec.log_chunks(
                doc_id,
                [{"text": text[s.start:s.end], "span": (s.start, s.end)} for s in spans],
            ))
    return chunk_ids


def test_recorder_matches_managed_chunk_ids(tmp_path: Path, small_corpus: Path):
    managed_store = ProjectStore(tmp_path / "managed")
    pipeline = build_pipeline(CONFIG)
    report = ingest(managed_store, small_corpus, pipeline, adapter=None)
    key = chunkset_key(report.manifest.corpus.merkle_root,
                       pipeline.stage("parse"), pipeline.stage("chunk"))
    managed_ids = [r.chunk_id for r in managed_store.get_chunks(key)]

    rec = Recorder(project="sdk", store=tmp_path / "sdk")
    recorded_ids = _record_corpus(rec, small_corpus)
    snapshot_id = rec.commit()

    assert sorted(recorded_ids) == sorted(managed_ids)
    manifest = rec.store.get_snapshot(snapshot_id)
    assert manifest.corpus.merkle_root == report.manifest.corpus.merkle_root
    assert manifest.artifacts["chunks_uri"] == report.manifest.artifacts["chunks_uri"]
    sdk_ids = [r.chunk_id for r in rec.store.get_chunks(key)]
    assert sorted(sdk_ids) == sorted(managed_ids)


def test_recorder_diffable_against_managed(tmp_path: Path, small_corpus: Path):
    managed_store = ProjectStore(tmp_path / "managed")
    pipeline = build_pipeline(CONFIG)
    report = ingest(managed_store, small_corpus, pipeline, adapter=None)

    rec = Recorder(project="sdk", store=tmp_path / "sdk")
    _record_corpus(rec, small_corpus)
    manifest = rec.store.get_snapshot(rec.commit())

    factors = report.manifest.pipeline.diff_factors(manifest.pipeline)
    assert "parse" not in factors
    assert "chunk" not in factors
    assert set(factors) <= {"embed", "index", "retrieve"}


def test_recorder_embeddings_cached_fp16(tmp_path: Path, small_corpus: Path):
    rec = Recorder(project="sdk", store=tmp_path / "sdk")
    chunk_ids = _record_corpus(rec, small_corpus)
    provider = LocalHashProvider(dims=64, seed=0)
    texts = [r.text for r in rec.logged_chunks()]
    vectors = provider.embed(texts)
    rec.log_embeddings(provider="local", model="hash-v1", chunk_ids=chunk_ids,
                       vectors=vectors, dims=64, params=provider.params)
    snapshot_id = rec.commit()

    ph = hashing.params_hash(provider.params)
    keys = [hashing.embedding_key(r.text_hash, "local", "hash-v1", 64, ph)
            for r in rec.logged_chunks()]
    assert rec.store.missing_embedding_keys(keys) == []
    stored = rec.store.get_embeddings(keys[:1])[keys[0]]
    np.testing.assert_allclose(stored, vectors[0], atol=1e-3)
    manifest = rec.store.get_snapshot(snapshot_id)
    assert manifest.artifacts["embeddings_uri"] == f"artifacts/emb/{provider.model_key}"
    assert manifest.pipeline.stage("embed") is not None


def test_sdk_snapshot_with_natural_params_is_evaluable(tmp_path: Path, small_corpus: Path):
    # Finding #4: a BYO caller that omits the canonical param set must still
    # produce a snapshot whose embeddings replay/eval can find. Before the fix the
    # recorder keyed under params_hash({}) while RetrievalEngine reconstructs
    # params_hash({dims,seed,ngram}), so evaluate() raised "embeddings missing".
    from recallops.evalrunner import evaluate
    from recallops.models import GoldenCase, GoldenDataset

    rec = Recorder(project="sdk", store=tmp_path / "sdk")
    chunk_ids = _record_corpus(rec, small_corpus)
    provider = LocalHashProvider(dims=64, seed=0)
    vectors = provider.embed([r.text for r in rec.logged_chunks()])
    # Natural call: dims given, params NOT passed (the bug trigger).
    rec.log_embeddings(provider="local", model="hash-v1",
                       chunk_ids=chunk_ids, vectors=vectors, dims=64)
    manifest = rec.store.get_snapshot(rec.commit())

    dataset = GoldenDataset("sdk-v1", [
        GoldenCase("c1", "How does the bootstrap command validate the manifest checksum?",
                   ["alpha.md"]),
        GoldenCase("c2", "How does the beta gateway route traffic by header affinity?",
                   ["beta.md"]),
    ])
    result = evaluate(rec.store, manifest, dataset)  # replay, must not KeyError
    assert set(result.per_query) == {"c1", "c2"}
    assert result.aggregate["hit_rate@5"] >= 0.5


def test_recorder_commit_deterministic(tmp_path: Path, small_corpus: Path):
    ids = []
    for name in ("one", "two"):
        rec = Recorder(project="sdk", store=tmp_path / name)
        _record_corpus(rec, small_corpus)
        ids.append(rec.commit())
    assert ids[0] == ids[1]


def test_log_retrieval_flushed_on_commit(tmp_path: Path, small_corpus: Path):
    rec = Recorder(project="sdk", store=tmp_path / "sdk")
    _record_corpus(rec, small_corpus)
    rec.log_retrieval("q_014", "dense", [("ch_a01", 0.812), ("ch_b02", 0.5)])
    rec.log_retrieval("q_014", "sparse", [("ch_b02", 1.2)])
    rec.log_retrieval("q_014", "fused", [("ch_a01", 0.9), ("ch_b02", 0.7)])
    assert rec.store.list_json("retrieval_log") == []
    snapshot_id = rec.commit()

    blob = rec.store.get_json("retrieval_log", f"{snapshot_id}:q_014")
    assert blob is not None
    run = QueryRun.from_dict(blob)
    assert run.query_id == "q_014"
    assert run.stages.dense == [("ch_a01", 0.812), ("ch_b02", 0.5)]
    assert run.stages.sparse == [("ch_b02", 1.2)]
    assert run.stages.reranked is None
    assert run.final == [("ch_a01", 0.9), ("ch_b02", 0.7)]


def test_log_retrieval_rerank_becomes_final(tmp_path: Path, small_corpus: Path):
    rec = Recorder(project="sdk", store=tmp_path / "sdk")
    _record_corpus(rec, small_corpus)
    rec.log_retrieval("q1", "dense", [("ch_a", 0.9)])
    rec.log_retrieval("q1", "rerank", [("ch_b", 0.99), ("ch_a", 0.4)])
    snapshot_id = rec.commit()
    run = QueryRun.from_dict(rec.store.get_json("retrieval_log", f"{snapshot_id}:q1"))
    assert run.stages.reranked == [("ch_b", 0.99), ("ch_a", 0.4)]
    assert run.final == run.stages.reranked


def test_stage_reentry_conflict_raises(tmp_path: Path):
    rec = Recorder(project="sdk", store=tmp_path / "sdk")
    with rec.stage("chunk", tool="acme_chunker", version="3.2", params={"size": 10}):
        pass
    with rec.stage("chunk", tool="acme_chunker", version="3.2", params={"size": 10}):
        pass
    with pytest.raises(ValueError):
        with rec.stage("chunk", tool="acme_chunker", version="3.3", params={"size": 10}):
            pass


def test_log_embeddings_unknown_chunk_raises(tmp_path: Path):
    rec = Recorder(project="sdk", store=tmp_path / "sdk")
    with pytest.raises(KeyError):
        rec.log_embeddings(provider="local", model="hash-v1",
                           chunk_ids=["ch_missing"], vectors=np.zeros((1, 8), dtype=np.float32))


def test_log_embeddings_conflicting_model_raises(tmp_path: Path, small_corpus: Path):
    # A second log_embeddings under a different model must raise: silently
    # keying the new vectors under the FIRST stage drops them (the old keys
    # already exist in the store, so put_embeddings writes nothing).
    rec = Recorder(project="sdk", store=tmp_path / "sdk")
    chunk_ids = _record_corpus(rec, small_corpus)
    n = len(chunk_ids)
    rec.log_embeddings(provider="local", model="hash-v1", chunk_ids=chunk_ids,
                       vectors=np.full((n, 16), 0.5, dtype=np.float32), dims=16)
    with pytest.raises(ValueError, match="embed"):
        rec.log_embeddings(provider="local", model="hash-v2", chunk_ids=chunk_ids,
                           vectors=np.full((n, 16), 0.25, dtype=np.float32), dims=16)


def test_log_embeddings_same_spec_in_batches_ok(tmp_path: Path, small_corpus: Path):
    rec = Recorder(project="sdk", store=tmp_path / "sdk")
    chunk_ids = _record_corpus(rec, small_corpus)
    n = len(chunk_ids)
    vectors = np.full((n, 16), 0.5, dtype=np.float32)
    split = n // 2
    rec.log_embeddings(provider="local", model="hash-v1", chunk_ids=chunk_ids[:split],
                       vectors=vectors[:split], dims=16)
    rec.log_embeddings(provider="local", model="hash-v1", chunk_ids=chunk_ids[split:],
                       vectors=vectors[split:], dims=16)
    snapshot_id = rec.commit()
    assert rec.store.get_snapshot(snapshot_id).artifacts["embeddings_uri"]


def test_declared_embed_stage_without_model_gives_clear_error(tmp_path: Path,
                                                              small_corpus: Path):
    rec = Recorder(project="sdk", store=tmp_path / "sdk")
    chunk_ids = _record_corpus(rec, small_corpus)
    with rec.stage("embed", tool="local", params={"dims": 16}):
        pass
    with pytest.raises(ValueError, match="model"):
        rec.log_embeddings(provider="local", model="hash-v1", chunk_ids=chunk_ids,
                           vectors=np.ones((len(chunk_ids), 16), dtype=np.float32), dims=16)


def _sdk_rerank_snapshot(tmp_path: Path, small_corpus: Path, log_rerank: bool):
    rec = Recorder(project="sdk", store=tmp_path / "sdk")
    chunk_ids = _record_corpus(rec, small_corpus)
    provider = LocalHashProvider(dims=64, seed=0)
    texts = [r.text for r in rec.logged_chunks()]
    rec.log_embeddings(provider="local", model="hash-v1", chunk_ids=chunk_ids,
                       vectors=provider.embed(texts), dims=64, params=provider.params)
    with rec.stage("rerank", tool="acme.cross-encoder", version="3", params={"top_n": 2}):
        pass
    target = next(r.chunk_id for r in rec.logged_chunks() if "thirty seconds" in r.text)
    other = next(r.chunk_id for r in rec.logged_chunks() if r.chunk_id != target)
    if log_rerank:
        rec.log_retrieval("q1", "rerank", [(target, 0.99), (other, 0.5)])
    snapshot_id = rec.commit()
    return rec.store, rec.store.get_snapshot(snapshot_id), target


def test_sdk_bespoke_rerank_replayed_from_log(tmp_path: Path, small_corpus: Path):
    # A bespoke rerank stage has no local implementation; replay must serve the
    # candidates the Recorder logged (FR-2.2) instead of crashing every eval.
    from recallops.evalrunner import evaluate
    from recallops.models import GoldenCase, GoldenDataset

    store, manifest, target = _sdk_rerank_snapshot(tmp_path, small_corpus, log_rerank=True)
    dataset = GoldenDataset("gold-v1", [
        GoldenCase("q1", "what is the default gateway timeout", ["beta.md"])])
    result = evaluate(store, manifest, dataset)
    qe = result.per_query["q1"]
    assert qe.ranked_chunks[0][0] == target
    assert qe.run.stages.reranked[0][0] == target
    assert qe.metrics["recall@5"] == 1.0


def test_sdk_bespoke_rerank_without_log_raises_clearly(tmp_path: Path, small_corpus: Path):
    from recallops.evalrunner import evaluate
    from recallops.models import GoldenCase, GoldenDataset

    store, manifest, _ = _sdk_rerank_snapshot(tmp_path, small_corpus, log_rerank=False)
    dataset = GoldenDataset("gold-v1", [
        GoldenCase("q1", "what is the default gateway timeout", ["beta.md"])])
    with pytest.raises(ValueError, match="log_retrieval"):
        evaluate(store, manifest, dataset)


def test_recorder_dedupes_duplicate_content_docs(tmp_path: Path):
    # Byte-identical files share one doc_id; log_chunks per file must not
    # append duplicate chunk records (managed ingest chunks each doc once, and
    # both write under the same content-addressed chunkset key).
    from recallops.ingest import chunkset_key
    rec = Recorder(project="sdk", store=tmp_path / "sdk")
    text = "refund policy applies to all orders"
    with rec.stage("parse", tool="text-v1", version="1"):
        d1 = rec.log_document("a.md", text.encode(), text)
        d2 = rec.log_document("b.md", text.encode(), text)
    assert d1 == d2
    with rec.stage("chunk", tool="recall.chunkers.fixed_token", version="1",
                   params={"max_tokens": 800, "overlap": 0}) as chunk_stage:
        rec.log_chunks(d1, [{"text": text, "span": (0, len(text))}])
        rec.log_chunks(d2, [{"text": text, "span": (0, len(text))}])
    ids = [c.chunk_id for c in rec.logged_chunks()]
    assert len(ids) == len(set(ids)) == 1
    snapshot_id = rec.commit()
    manifest = rec.store.get_snapshot(snapshot_id)
    assert manifest.corpus.chunk_count == 1
    key = chunkset_key(manifest.corpus.merkle_root,
                       rec.store.get_snapshot(snapshot_id).pipeline.stage("parse"), chunk_stage)
    assert len(rec.store.get_chunks(key)) == 1


def test_log_embeddings_declared_stage_nondefault_params_ok(tmp_path: Path, small_corpus: Path):
    # A user who declares the embed stage with non-default params and then calls
    # log_embeddings without repeating them must NOT be rejected: vectors are
    # keyed under the declared stage that lands in the manifest.
    rec = Recorder(project="sdk", store=tmp_path / "sdk")
    chunk_ids = _record_corpus(rec, small_corpus)
    with rec.stage("embed", tool="local", version="1",
                   params={"provider": "local", "model": "hash-v1", "dims": 8,
                           "seed": 7, "ngram": [1, 2]}):
        pass
    rec.log_embeddings(provider="local", model="hash-v1", chunk_ids=chunk_ids,
                       vectors=np.ones((len(chunk_ids), 8), dtype=np.float32), dims=8)
    snapshot_id = rec.commit()
    assert rec.store.get_snapshot(snapshot_id).artifacts["embeddings_uri"]


def test_log_embeddings_declared_stage_genuine_conflict_raises(tmp_path: Path, small_corpus: Path):
    rec = Recorder(project="sdk", store=tmp_path / "sdk")
    chunk_ids = _record_corpus(rec, small_corpus)
    with rec.stage("embed", tool="local", version="1",
                   params={"provider": "local", "model": "hash-v1", "dims": 8,
                           "seed": 7, "ngram": [1, 2]}):
        pass
    with pytest.raises(ValueError, match="conflict"):
        rec.log_embeddings(provider="local", model="hash-v2", chunk_ids=chunk_ids,
                           vectors=np.ones((len(chunk_ids), 8), dtype=np.float32), dims=8)


def test_log_embeddings_param_change_same_model_raises(tmp_path: Path, small_corpus: Path):
    # Same model but a changed param (seed) between two calls must raise, not
    # silently drop the second call's vectors under the first stage's keys.
    rec = Recorder(project="sdk", store=tmp_path / "sdk")
    chunk_ids = _record_corpus(rec, small_corpus)
    n = len(chunk_ids)
    rec.log_embeddings(provider="local", model="hash-v1", chunk_ids=chunk_ids,
                       vectors=np.full((n, 8), 0.5, dtype=np.float32), dims=8)
    with pytest.raises(ValueError, match="conflict"):
        rec.log_embeddings(provider="local", model="hash-v1", chunk_ids=chunk_ids,
                           vectors=np.full((n, 8), 0.25, dtype=np.float32), dims=8,
                           params={"seed": 5})
