from __future__ import annotations

import pytest

from recallops import models
from recallops.models import (
    Arm,
    ArmResult,
    AttributionReport,
    CalibrationRecord,
    ChunkFate,
    ChunkRecord,
    CorpusInfo,
    DiffResult,
    EvalResult,
    Factor,
    FunnelReport,
    GateResult,
    GoldenCase,
    GoldenDataset,
    Hypothesis,
    PipelineDAG,
    QueryDiff,
    QueryEval,
    QueryRun,
    SnapshotManifest,
    StageCandidates,
    StageSpec,
    VerifiedCause,
)


def make_dag() -> PipelineDAG:
    return PipelineDAG((
        StageSpec("parse", "text-v1", "1"),
        StageSpec("chunk", "recall.chunkers.markdown_heading", "1",
                  {"max_tokens": 800, "overlap": 120}, ("parse",)),
        StageSpec("embed", "local", "1", {"model": "hash-v1", "dims": 256, "seed": 0}, ("chunk",)),
        StageSpec("retrieve", "recall.retrieve", "1",
                  {"top_k": 5, "hybrid": {"sparse": "bm25", "fusion": "weighted", "bm25_weight": 0.45}},
                  ("embed",)),
    ))


def test_dag_validate_ok_and_cycle_detection():
    make_dag().validate()
    bad = PipelineDAG((StageSpec("a", "t", inputs=("b",)), StageSpec("b", "t", inputs=("a",))))
    with pytest.raises(ValueError):
        bad.validate()
    with pytest.raises(ValueError):
        PipelineDAG((StageSpec("a", "t", inputs=("missing",)),)).validate()


def test_dag_replace_and_diff_factors():
    dag = make_dag()
    new_chunk = StageSpec("chunk", "recall.chunkers.fixed_token", "1",
                          {"max_tokens": 60, "overlap": 0}, ("parse",))
    dag2 = dag.replace("chunk", new_chunk)
    diff = dag.diff_factors(dag2)
    assert set(diff) == {"chunk"}
    assert diff["chunk"]["before"]["tool"] == "recall.chunkers.markdown_heading"
    assert diff["chunk"]["after"]["tool"] == "recall.chunkers.fixed_token"
    assert dag.diff_factors(dag) == {}


def test_manifest_identity_is_content_derived():
    dag = make_dag()
    corpus = CorpusInfo(3, 12, "mr_abc")
    m1 = SnapshotManifest.build(dag, corpus, {"chunks_uri": "x.parquet"}, created_at="2026-07-07T00:00:00Z")
    m2 = SnapshotManifest.build(dag, corpus, {"chunks_uri": "x.parquet"}, created_at="2026-07-08T09:00:00Z")
    assert m1.snapshot_id == m2.snapshot_id
    assert m1.snapshot_id.startswith("snap_")
    m3 = SnapshotManifest.build(dag, CorpusInfo(3, 12, "mr_other"), {"chunks_uri": "x.parquet"})
    assert m3.snapshot_id != m1.snapshot_id


def test_golden_dataset_versioning():
    ds = GoldenDataset("golden-v3", [GoldenCase("q_1", "?", ["a.md"])])
    assert ds.name == "golden" and ds.version == 3
    ds2 = ds.bump([GoldenCase("q_2", "??", ["b.md"])])
    assert ds2.dataset_id == "golden-v4" and len(ds2.cases) == 2
    assert GoldenDataset("plain", []).version == 1


def test_chunk_fate_serializes_with_class_key():
    fate = ChunkFate("split", 0.92, "ch_5f2e", ["ch_a01", "ch_a02"])
    d = fate.to_dict()
    assert d["class"] == "split"
    assert ChunkFate.from_dict(d) == fate


def test_arm_id_deterministic_and_sides():
    a1 = Arm.build({"chunk": "A", "embed": "B"})
    a2 = Arm.build({"embed": "B", "chunk": "A"})
    assert a1.arm_id == a2.arm_id
    assert a1.at_a == frozenset({"chunk"}) and a1.at_b == frozenset({"embed"})


def _query_eval(qid="q_1") -> QueryEval:
    run = QueryRun(qid, "?", StageCandidates([("c1", 1.0)], [("c1", 2.0)], [("c1", 1.5)], None), [("c1", 1.5)])
    return QueryEval(qid, [("c1", 1.5)], ["a.md"], 1, {5: True}, {"recall@5": 1.0}, run)


def test_json_round_trips():
    dag = make_dag()
    manifest = SnapshotManifest.build(dag, CorpusInfo(1, 2, "mr_x"), {"u": "v"})
    ev = EvalResult("ev_1", manifest.snapshot_id, "golden-v1", "replay", "local",
                    "", (1, 5), {"q_1": _query_eval()}, {"recall@5": 1.0})
    funnel = FunnelReport("ch_1", True, {"rank_before": 1, "rank_after": 9, "shadow_exact_rank_after": 9},
                          {"rank_before": 2, "rank_after": 14}, {"rank_before": 1, "rank_after": 11},
                          {"in_candidates_after": False}, False)
    rep = AttributionReport("q_1", "regressed", "stable", funnel,
                            ChunkFate("split", 0.9, "ch_1", ["ch_2"]),
                            [VerifiedCause("chunk", "arm_x", 1)],
                            [Hypothesis("retrieve", evidence="sparse rank drop co-occurred")],
                            "narrative text")
    qd = QueryDiff("q_1", "regressed", "stable", {"recall@5": -1.0}, _query_eval(), _query_eval())
    diff = DiffResult("diff_1", "snap_a", "snap_b", "golden-v1", {"chunk": {"before": None, "after": None}},
                      False, {"recall@5": -0.1}, {"q_1": qd}, {"ch_1": ChunkFate("intact", 1.0, "ch_1", ["ch_1"])},
                      False, True)
    objs = [
        manifest, ev, rep, diff,
        GoldenDataset("g-v1", [GoldenCase("q", "?", ["a.md"], tags=["t"], origin="synthetic")]),
        ChunkRecord("ch_1", "doc_1", 0, 5, 0, "hello", "tx_1", "parse", "chunk"),
        CalibrationRecord("snap_a", 3, {"recall@5": 0.01}, 0.004, {"q_1": 0.0}),
        GateResult(True, "statistical", [], False, []),
        ArmResult("arm_x", ev, 0),
        Factor("chunk", "stage"),
    ]
    for obj in objs:
        rt = type(obj).from_dict(obj.to_dict())
        assert rt.to_dict() == obj.to_dict(), type(obj).__name__


def test_attribution_report_matches_prd_shape():
    d = AttributionReport(
        "q_014", "regressed", "stable",
        FunnelReport("ch_5f2e", True, {}, {}, {}, {}, False),
        ChunkFate("split", 0.92, "ch_5f2e", ["ch_a01", "ch_a02"]),
        [VerifiedCause("chunker", "arm_B_with_chunker_A", 1)],
        [Hypothesis("bm25_weight", evidence="sparse rank drop co-occurred")],
        "...",
    ).to_dict()
    assert set(d) == {"query_id", "classification", "stability", "funnel", "chunk_fate",
                      "verified_causes", "hypotheses", "narrative"}
    assert d["verified_causes"][0]["status"] == "verified"
    assert d["hypotheses"][0]["status"] == "unverified"
    assert d["chunk_fate"]["class"] == "split"


def test_constants():
    assert models.CHUNK_FATES == ("intact", "split", "merged", "boundary-shifted", "dropped")
    assert models.QUERY_CLASSES == ("improved", "regressed", "changed-top-k", "unchanged")
