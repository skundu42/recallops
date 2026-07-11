from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from recallops import ingest as ingest_mod
from recallops import retrieval
from recallops.adapters.local import LocalIndexAdapter
from recallops.ingest import build_pipeline, ingest
from recallops.pipeline import chunkers
from recallops.pipeline.providers import LocalHashProvider
from recallops.rerankers import get_reranker
from recallops.retrieval import MIN_CANDIDATE_DEPTH, RetrievalEngine, collection_name
from recallops.store import ProjectStore

CRAFTED_DOCS = {
    "exact.md": (
        "Zephyrite deposits occur inside a crystal lattice formation near the old survey site. "
        "The field team catalogued basalt cores, measured seismic drift, and logged aquifer "
        "pressure at dawn. Their notebooks describe gravel terraces, moraine ridges, and the "
        "slow creep of talus fans down the valley. A final appendix lists drill depths, "
        "borehole temperatures, and the calibration record for every instrument used."
    ),
    "fuzzy.md": (
        "Crystal lattice geometry. The crystal lattice bends light. Crystal lattice energy "
        "defines every crystal lattice boundary."
    ),
    "harbor.md": (
        "The harbor master schedules tugboats around the tide tables and berth assignments. "
        "Cargo manifests list crystal glassware stacked in a lattice of padded crates. Crane "
        "operators certify rigging before each lift and log wind speed hourly."
    ),
    "orchard.md": (
        "Orchard crews prune the apple rows before the first frost settles. A trellis holds "
        "each espalier in a lattice pattern, and dew beads like crystal on the netting. "
        "Harvest bins are weighed and tagged at the packing shed."
    ),
    "telescope.md": (
        "The telescope array tracks pulsars across the southern sky each winter. Engineers "
        "polish every crystal oscillator and align the antenna lattice before observation "
        "runs. Data tapes ship weekly to the correlation facility."
    ),
    "pottery.md": (
        "The pottery studio fires stoneware glazes in a downdraft kiln overnight. Silica "
        "melts into a crystal sheen while shelf posts form a sturdy lattice inside the "
        "chamber. Apprentices wedge clay and label test tiles."
    ),
    "glacier.md": (
        "Glacier surveyors stake ablation lines along the icefall every spring. Meltwater "
        "refreezes into crystal veins that thread a lattice through the firn. Camp rations "
        "and fuel are cached beneath marked cairns."
    ),
    "railway.md": (
        "Railway signallers rehearse failover drills in the interlocking tower. Frost leaves "
        "crystal patterns on the gantry lattice above the junction each January. Timetables "
        "shift when maintenance possessions close the loop line."
    ),
}

CRAFTED_QUERY = "zephyrite crystal lattice"

SMALL_QUESTIONS = (
    "bootstrap command validates the manifest checksum",
    "gateway routes traffic by header affinity",
    "replication lag alerts and storage quotas",
)


@pytest.fixture
def crafted_corpus(tmp_path: Path) -> Path:
    root = tmp_path / "crafted"
    root.mkdir()
    for name, text in CRAFTED_DOCS.items():
        (root / name).write_text(text + "\n")
    return root


def crafted_config(bm25_weight: float, top_k: int = 5, rerank: bool = False) -> dict:
    config: dict = {
        "chunker": {"tool": chunkers.FIXED_TOKEN, "params": {"max_tokens": 200, "overlap": 0}},
        "retrieve": {
            "top_k": top_k,
            "hybrid": {"sparse": "bm25", "fusion": "weighted", "bm25_weight": bm25_weight},
        },
    }
    if rerank:
        config["rerank"] = {"tool": "recall.rerankers.overlap", "params": {}, "top_n": 3}
    return config


def build_snapshot(root: Path, source_dir: Path, config: dict | None = None, adapter=None):
    store = ProjectStore(root)
    report = ingest(store, source_dir, build_pipeline(config or {}), adapter)
    return store, report.manifest


def ids(pairs: list[tuple[str, float]]) -> list[str]:
    return [cid for cid, _ in pairs]


def scores(pairs: list[tuple[str, float]]) -> list[float]:
    return [s for _, s in pairs]


def test_query_vector_is_cached_across_engines(tmp_project: Path, small_corpus: Path) -> None:
    # A network embedding provider must not re-embed the same query on every eval
    # pass. The per-store cache means the second lookup (even from a fresh engine)
    # does not call the provider again.
    store, manifest = build_snapshot(tmp_project / "proj", small_corpus)
    engine = RetrievalEngine(store, manifest)
    provider = engine.provider
    calls = {"n": 0}
    real_embed = provider.embed

    def counting_embed(texts):
        calls["n"] += 1
        return real_embed(texts)

    provider.embed = counting_embed
    v1 = engine.query_vector("how does the bootstrap command validate the manifest?")
    v2 = engine.query_vector("how does the bootstrap command validate the manifest?")
    assert calls["n"] == 1  # second call served from cache
    np.testing.assert_array_equal(v1, v2)

    # a fresh engine on the SAME store reuses the cached vector (0 more calls)
    engine2 = RetrievalEngine(store, manifest)
    engine2._provider = provider  # same counting provider
    engine2.query_vector("how does the bootstrap command validate the manifest?")
    assert calls["n"] == 1


def test_replay_matches_live_exact_adapter(tmp_project: Path, small_corpus: Path) -> None:
    adapter = LocalIndexAdapter(tmp_project / "index")
    store, manifest = build_snapshot(tmp_project / "proj", small_corpus, adapter=adapter)
    live = RetrievalEngine(store, manifest, adapter=adapter)
    replay = RetrievalEngine(store, manifest)
    for i, question in enumerate(SMALL_QUESTIONS):
        run_live = live.run_query(f"q{i}", question)
        run_replay = replay.run_query(f"q{i}", question)
        assert ids(run_live.stages.dense) == ids(run_replay.stages.dense)
        assert ids(run_live.stages.sparse) == ids(run_replay.stages.sparse)
        assert ids(run_live.stages.fused) == ids(run_replay.stages.fused)
        assert ids(run_live.final) == ids(run_replay.final)
        assert np.allclose(scores(run_live.stages.dense), scores(run_replay.stages.dense))
        assert np.allclose(scores(run_live.final), scores(run_replay.final))


def test_replay_matches_live_on_example_corpus(tmp_project: Path, corpus_dir: Path) -> None:
    adapter = LocalIndexAdapter(tmp_project / "index")
    store, manifest = build_snapshot(tmp_project / "proj", corpus_dir, adapter=adapter)
    live = RetrievalEngine(store, manifest, adapter=adapter)
    replay = RetrievalEngine(store, manifest)
    for i, question in enumerate((
        "how do refunds work for annual invoices",
        "configure single sign-on with the identity provider",
    )):
        run_live = live.run_query(f"q{i}", question)
        run_replay = replay.run_query(f"q{i}", question)
        assert ids(run_live.stages.dense) == ids(run_replay.stages.dense)
        assert ids(run_live.final) == ids(run_replay.final)


def test_bm25_weight_high_favors_exact_term_doc(tmp_project: Path, crafted_corpus: Path) -> None:
    store = ProjectStore(tmp_project / "proj")
    dense_manifest = ingest(store, crafted_corpus, build_pipeline(crafted_config(0.0)), None).manifest
    sparse_manifest = ingest(store, crafted_corpus, build_pipeline(crafted_config(1.0)), None).manifest

    dense_engine = RetrievalEngine(store, dense_manifest)
    sparse_engine = RetrievalEngine(store, sparse_manifest)
    dense_top = dense_engine.run_query("q", CRAFTED_QUERY).final[0][0]
    sparse_top = sparse_engine.run_query("q", CRAFTED_QUERY).final[0][0]

    assert dense_engine.chunk_doc_map()[dense_top] == "fuzzy.md"
    assert sparse_engine.chunk_doc_map()[sparse_top] == "exact.md"


def test_rerank_stage_applied(tmp_project: Path, crafted_corpus: Path) -> None:
    config = crafted_config(0.0, top_k=8, rerank=True)
    store, manifest = build_snapshot(tmp_project / "proj", crafted_corpus, config)
    engine = RetrievalEngine(store, manifest)
    run = engine.run_query("q", CRAFTED_QUERY)

    assert run.stages.reranked is not None
    assert run.final == run.stages.reranked[:3]
    assert len(run.final) == 3

    texts = engine.chunk_texts()
    reranker = get_reranker("recall.rerankers.overlap", {})
    expected = reranker(CRAFTED_QUERY, [(cid, texts[cid]) for cid, _ in run.stages.fused[:8]])
    assert run.stages.reranked == expected

    doc_map = engine.chunk_doc_map()
    assert doc_map[run.stages.fused[0][0]] == "fuzzy.md"
    assert doc_map[run.final[0][0]] == "exact.md"


def test_dense_only_populates_all_stages(tmp_project: Path, small_corpus: Path) -> None:
    config = {"retrieve": {"top_k": 5, "hybrid": None}}
    store, manifest = build_snapshot(tmp_project / "proj", small_corpus, config)
    engine = RetrievalEngine(store, manifest)
    run = engine.run_query("q", "gateway routes traffic by header affinity")

    assert run.stages.fused == run.stages.dense
    assert run.stages.sparse
    assert run.stages.reranked is None
    assert run.final == run.stages.dense[:5]
    assert all(isinstance(s, float) for s in scores(run.stages.sparse))


def test_candidate_depth(tmp_project: Path, corpus_dir: Path) -> None:
    config = {"retrieve": {"top_k": 2,
                           "hybrid": {"sparse": "bm25", "fusion": "weighted", "bm25_weight": 0.45}}}
    store, manifest = build_snapshot(tmp_project / "proj", corpus_dir, config)
    engine = RetrievalEngine(store, manifest)
    run = engine.run_query("q", "how are api rate limits enforced per token")

    depth = max(2 * 4, MIN_CANDIDATE_DEPTH)
    assert depth == 20
    assert len(run.stages.dense) == min(depth, manifest.corpus.chunk_count)
    assert len(run.stages.sparse) <= depth
    assert len(run.final) == 2


def test_determinism_across_engine_and_store_instances(tmp_project: Path, small_corpus: Path) -> None:
    store, manifest = build_snapshot(tmp_project / "proj", small_corpus)
    engine = RetrievalEngine(store, manifest)
    first = engine.run_query("q0", SMALL_QUESTIONS[0]).to_dict()
    second = engine.run_query("q0", SMALL_QUESTIONS[0]).to_dict()
    assert first == second

    fresh_store = ProjectStore(tmp_project / "proj")
    fresh_manifest = fresh_store.get_snapshot(manifest.snapshot_id)
    fresh = RetrievalEngine(fresh_store, fresh_manifest).run_query("q0", SMALL_QUESTIONS[0]).to_dict()
    assert fresh == first


def test_exact_dense_ranks_full_corpus(tmp_project: Path, small_corpus: Path) -> None:
    adapter = LocalIndexAdapter(tmp_project / "index")
    store, manifest = build_snapshot(tmp_project / "proj", small_corpus, adapter=adapter)
    replay = RetrievalEngine(store, manifest)
    live = RetrievalEngine(store, manifest, adapter=adapter)
    question = SMALL_QUESTIONS[0]

    exact = replay.exact_dense_ranks(question)
    assert len(exact) == manifest.corpus.chunk_count
    assert scores(exact) == sorted(scores(exact), reverse=True)
    assert set(ids(exact)) == set(replay.chunk_texts())

    run = replay.run_query("q0", question)
    assert ids(run.stages.dense) == ids(exact)[: len(run.stages.dense)]

    live_exact = live.exact_dense_ranks(question)
    assert ids(live_exact) == ids(exact)
    assert np.allclose(scores(live_exact), scores(exact))


def test_chunk_maps_and_query_vector(tmp_project: Path, small_corpus: Path) -> None:
    store, manifest = build_snapshot(tmp_project / "proj", small_corpus)
    engine = RetrievalEngine(store, manifest)

    key = retrieval.chunkset_key_from_uri(manifest.artifacts["chunks_uri"])
    records = store.get_chunks(key)
    assert engine.chunk_texts() == {r.chunk_id: r.text for r in records}

    doc_map = engine.chunk_doc_map()
    assert set(doc_map) == {r.chunk_id for r in records}
    assert set(doc_map.values()) == {"alpha.md", "beta.md", "gamma.md"}

    question = SMALL_QUESTIONS[1]
    vector = engine.query_vector(question)
    provider = LocalHashProvider(dims=256, seed=0)
    assert np.allclose(vector, provider.embed([question])[0])
    assert vector.dtype == np.float32
    assert np.isclose(float(np.linalg.norm(vector)), 1.0, atol=1e-5)


def test_bm25_built_once_per_engine(tmp_project: Path, small_corpus: Path,
                                    monkeypatch: pytest.MonkeyPatch) -> None:
    store, manifest = build_snapshot(tmp_project / "proj", small_corpus)
    engine = RetrievalEngine(store, manifest)
    calls: list[int] = []
    real = retrieval.BM25Index

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(retrieval, "BM25Index", counting)
    engine.run_query("q0", SMALL_QUESTIONS[0])
    engine.run_query("q1", SMALL_QUESTIONS[1])
    assert len(calls) == 1


def test_collection_name_reexported_from_ingest() -> None:
    assert collection_name is ingest_mod.collection_name


def test_managed_unknown_reranker_surfaces_real_error(tmp_path, small_corpus):
    # A managed snapshot (no retrieval log) with a typo'd rerank tool must get
    # the real "unknown reranker" error, not a misleading SDK log-replay hint.
    store = ProjectStore(tmp_path / "proj")
    pipeline = build_pipeline({"rerank": {"tool": "recall.rerankers.overlp", "top_n": 5}})
    manifest = ingest(store, small_corpus, pipeline, adapter=None).manifest
    engine = RetrievalEngine(store, manifest)
    with pytest.raises(ValueError, match="unknown reranker"):
        engine.run_query("q1", "how do I bootstrap the alpha widget")


def test_query_vector_uses_embed_queries(corpus_dir):
    from click.testing import CliRunner

    from recallops.cli import main
    from recallops.retrieval import RetrievalEngine
    from recallops.store import ProjectStore

    runner = CliRunner()
    with runner.isolated_filesystem():
        for args in (["init", "--source", str(corpus_dir)], ["ingest"]):
            result = runner.invoke(main, args)
            assert result.exit_code == 0, result.output
        store = ProjectStore(".")
        manifest = store.list_snapshots()[-1]
        engine = RetrievalEngine(store, manifest)
        provider = engine.provider
        calls: list[list[str]] = []
        original = provider.embed_queries

        def spy(texts):
            calls.append(list(texts))
            return original(texts)

        provider.embed_queries = spy
        vec = engine.query_vector("what is the refund window?")
        assert calls == [["what is the refund window?"]]
        assert vec.shape == (provider.dims,)
