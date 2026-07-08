from __future__ import annotations

from types import SimpleNamespace

import pytest

from recallops.adapters.local import LocalIndexAdapter
from recallops.diffing import diff
from recallops.evalrunner import evaluate
from recallops.funnel import (
    failing_stage,
    funnel_for_query,
    implicated_factors,
    tracer_chunk,
)
from recallops.ingest import build_pipeline, ingest
from recallops.models import (
    ChunkFate,
    FunnelReport,
    GoldenCase,
    GoldenDataset,
    QueryDiff,
    QueryEval,
    QueryRun,
    StageCandidates,
)
from recallops.pipeline import chunkers
from recallops.retrieval import RetrievalEngine
from recallops.store import ProjectStore

CASES = [
    ("q1", "How are annual plan refunds prorated after the 30-day window closes?",
     ["billing/refunds.md"]),
    ("q2", "Which single sign-on protocols like SAML and OIDC are supported?",
     ["security/sso.md"]),
    ("q3", "What happens when an engineer declares an incident with the incident bot?",
     ["ops/incident-runbook.md"]),
    ("q4", "When is production access granted to new engineers during onboarding?",
     ["hr/onboarding.md"]),
    ("q5", "How long is the key rotation overlap window for API keys by default?",
     ["api/auth.md"]),
    ("q6", "How do refunds relate to invoices and payment methods?",
     ["billing/refunds.md", "billing/invoices.md"]),
    ("q7", "How do pricing plans and the product roadmap cover SSO?",
     ["sales/pricing.md", "product/roadmap.md", "security/sso.md"]),
    ("q8", "What notice periods apply to price changes and subprocessor onboarding?",
     ["sales/pricing.md", "legal/dpa.md"]),
]


def _dataset() -> GoldenDataset:
    return GoldenDataset("hand-v1", [
        GoldenCase(id=cid, question=q, expected_sources=list(sources), tags=[])
        for cid, q, sources in CASES
    ])


def make_funnel(*, target_in_index_after=True, dense=None, sparse=None, fused=None,
                rerank=None, ann_divergence=False, target="ch_x") -> FunnelReport:
    return FunnelReport(
        target_chunk_before=target,
        target_in_index_after=target_in_index_after,
        dense=dense or {"rank_before": 1, "rank_after": 1, "shadow_exact_rank_after": 1},
        sparse=sparse or {"rank_before": 1, "rank_after": 1},
        fused=fused or {"rank_before": 1, "rank_after": 1},
        rerank=rerank or {"in_candidates_after": True},
        ann_divergence=ann_divergence,
    )


class TestTracerChunk:
    def test_picks_top_ranked_expected_doc_chunk(self):
        before = QueryEval(
            query_id="q", ranked_chunks=[("c1", 0.9), ("c2", 0.8), ("c3", 0.7)],
            ranked_docs=[], target_rank=2, hit_at={}, metrics={}, run=None,
        )
        doc_map = {"c1": ("other.md",), "c2": ("billing/refunds.md",), "c3": ("billing/refunds.md",)}
        assert tracer_chunk(before, ["billing/refunds.md"], doc_map) == "c2"

    def test_none_when_no_expected_chunk(self):
        before = QueryEval(
            query_id="q", ranked_chunks=[("c1", 0.9), ("c2", 0.8)],
            ranked_docs=[], target_rank=None, hit_at={}, metrics={}, run=None,
        )
        doc_map = {"c1": ("a.md",), "c2": ("b.md",)}
        assert tracer_chunk(before, ["billing/refunds.md"], doc_map) is None

    def test_matches_any_of_multiple_expected(self):
        before = QueryEval(
            query_id="q", ranked_chunks=[("c1", 0.9), ("c2", 0.8)],
            ranked_docs=[], target_rank=1, hit_at={}, metrics={}, run=None,
        )
        doc_map = {"c1": ("sales/pricing.md",), "c2": ("legal/dpa.md",)}
        assert tracer_chunk(before, ["legal/dpa.md", "sales/pricing.md"], doc_map) == "c1"


class TestImplicatedFactors:
    def test_dense_maps_to_upstream_factors(self):
        assert implicated_factors("dense", {"chunk": {}}) == ["chunk"]
        assert implicated_factors("dense", {"embed": {}, "retrieve": {}}) == ["embed"]

    def test_index_intersection_sorted(self):
        got = implicated_factors("index", {"embed": {}, "index": {}, "chunk": {}, "retrieve": {}})
        assert got == ["chunk", "embed", "index"]

    def test_sparse_excludes_embed_and_retrieve(self):
        assert implicated_factors("sparse", {"chunk": {}, "embed": {}, "retrieve": {}}) == ["chunk"]

    def test_fused_includes_retrieve(self):
        assert implicated_factors("fused", {"retrieve": {}}) == ["retrieve"]
        assert implicated_factors("fused", {"retrieve": {}, "chunk": {}}) == ["chunk", "retrieve"]

    def test_rerank_maps(self):
        assert implicated_factors("rerank", {"rerank": {}, "embed": {}}) == ["rerank"]

    def test_ann_maps_to_index_only(self):
        assert implicated_factors("ann", {"index": {}, "chunk": {}}) == ["index"]
        assert implicated_factors("ann", {}) == []


class TestFailingStage:
    def test_index_when_target_missing(self):
        f = make_funnel(target_in_index_after=False,
                        dense={"rank_before": 1, "rank_after": 1, "shadow_exact_rank_after": 1})
        assert failing_stage(f, 5) == "index"

    def test_dense_crossing(self):
        f = make_funnel(
            dense={"rank_before": 1, "rank_after": 9, "shadow_exact_rank_after": 9},
            sparse={"rank_before": 2, "rank_after": 2},
            fused={"rank_before": 1, "rank_after": 1},
        )
        assert failing_stage(f, 5) == "dense"

    def test_sparse_crossing(self):
        f = make_funnel(
            dense={"rank_before": 1, "rank_after": 1, "shadow_exact_rank_after": 1},
            sparse={"rank_before": 2, "rank_after": 14},
            fused={"rank_before": 1, "rank_after": 1},
        )
        assert failing_stage(f, 5) == "sparse"

    def test_fused_crossing(self):
        f = make_funnel(
            dense={"rank_before": 1, "rank_after": 3, "shadow_exact_rank_after": 3},
            sparse={"rank_before": 2, "rank_after": 4},
            fused={"rank_before": 1, "rank_after": 11},
        )
        assert failing_stage(f, 5) == "fused"

    def test_ann_when_shadow_fine_but_diverged(self):
        f = make_funnel(
            dense={"rank_before": 1, "rank_after": 1, "shadow_exact_rank_after": 2},
            sparse={"rank_before": 1, "rank_after": 1},
            fused={"rank_before": 1, "rank_after": 1},
            ann_divergence=True,
        )
        assert failing_stage(f, 5) == "ann"

    def test_fallback_largest_degradation(self):
        f = make_funnel(
            dense={"rank_before": 1, "rank_after": 4, "shadow_exact_rank_after": 4},
            sparse={"rank_before": 1, "rank_after": 2},
            fused={"rank_before": 1, "rank_after": 1},
        )
        assert failing_stage(f, 10) == "dense"

    def test_missing_after_counts_as_out(self):
        f = make_funnel(
            dense={"rank_before": 1, "rank_after": None, "shadow_exact_rank_after": 30},
            sparse={"rank_before": 1, "rank_after": 1},
            fused={"rank_before": 1, "rank_after": 1},
        )
        assert failing_stage(f, 5) == "dense"


@pytest.fixture(scope="module")
def chunker_scenario(tmp_path_factory, corpus_dir):
    root = tmp_path_factory.mktemp("funnel_chunk")
    store = ProjectStore(root)
    man_a = ingest(store, corpus_dir, build_pipeline({}), None).manifest
    man_b = ingest(store, corpus_dir, build_pipeline({
        "chunker": {"tool": chunkers.FIXED_TOKEN, "params": {"max_tokens": 60, "overlap": 0}},
    }), None).manifest
    dataset = _dataset()
    eval_a = evaluate(store, man_a, dataset)
    eval_b = evaluate(store, man_b, dataset)
    d = diff(store, man_a, man_b, dataset, eval_a, eval_b)
    return SimpleNamespace(
        store=store, dataset=dataset, diff=d,
        engine_a=RetrievalEngine(store, man_a),
        engine_b=RetrievalEngine(store, man_b),
    )


def _regressed_funnels(sc):
    out = []
    for case in sc.dataset.cases:
        qdiff = sc.diff.queries[case.id]
        if qdiff.classification != "regressed":
            continue
        f = funnel_for_query(sc.engine_a, sc.engine_b, qdiff, case, sc.diff.alignment)
        out.append((case, qdiff, f))
    return out


def test_intact_tracer_with_new_id_followed_to_descendant():
    # Finding #6: an intact fate whose descendant has a new content-addressed id
    # (boundary shift within the 0.95 threshold) must be followed, else the
    # target is wrongly reported absent from B and every stage rank collapses.
    def run(cid, score):
        cands = StageCandidates([(cid, score)], [(cid, score)], [(cid, score)], None)
        return QueryRun("q", "question", cands, [(cid, score)])

    before = QueryEval("q", [("ch_old", 0.9)], ["doc.md"], 1, {5: True},
                       {"recall@5": 1.0}, run("ch_old", 0.9))
    after = QueryEval("q", [("ch_new", 0.8)], ["doc.md"], 1, {5: True},
                      {"recall@5": 1.0}, run("ch_new", 0.8))
    qdiff = QueryDiff("q", "regressed", "stable", {"recall@5": 0.0}, before, after)
    alignment = {"ch_old": ChunkFate("intact", 0.98, "ch_old", ["ch_new"])}
    case = GoldenCase("q", "question", ["doc.md"])
    engine_a = SimpleNamespace(chunk_doc_paths=lambda: {"ch_old": ("doc.md",), "ch_new": ("doc.md",)})
    engine_b = SimpleNamespace(
        chunk_texts=lambda: {"ch_new": "content"},          # ch_old is NOT present
        exact_dense_ranks=lambda q: [("ch_new", 0.8)],
        manifest=SimpleNamespace(pipeline=SimpleNamespace(stage=lambda s: None)),
    )
    f = funnel_for_query(engine_a, engine_b, qdiff, case, alignment)
    assert f.target_in_index_after is True
    assert f.dense["rank_after"] == 1


class TestChunkerScenario:
    def test_regressed_queries_exist(self, chunker_scenario):
        assert len(_regressed_funnels(chunker_scenario)) >= 1

    def test_regressed_query_attribution(self, chunker_scenario):
        sc = chunker_scenario
        found = None
        for case, qdiff, f in _regressed_funnels(sc):
            db, da = f.dense["rank_before"], f.dense["rank_after"]
            sb, sa = f.sparse["rank_before"], f.sparse["rank_after"]
            dense_drop = db is not None and (da is None or da > db)
            sparse_drop = sb is not None and (sa is None or sa > sb)
            if dense_drop and sparse_drop:
                found = (case, f)
                break
        assert found is not None, "expected a regressed query with dense+sparse rank drops"
        case, f = found
        assert f.target_in_index_after is True
        stage = failing_stage(f, 5)
        assert stage in {"dense", "sparse", "fused"}
        assert "chunk" in implicated_factors(stage, sc.diff.config_diff)
        fate = sc.diff.alignment.get(f.target_chunk_before)
        assert fate is not None and fate.cls != "intact"

    def test_split_tracer_followed_to_descendants(self, chunker_scenario):
        sc = chunker_scenario
        found = None
        for case in sc.dataset.cases:
            qdiff = sc.diff.queries[case.id]
            f = funnel_for_query(sc.engine_a, sc.engine_b, qdiff, case, sc.diff.alignment)
            fate = sc.diff.alignment.get(f.target_chunk_before)
            if fate is not None and fate.cls == "split":
                found = (case, qdiff, f, fate)
                break
        assert found is not None, "expected a query whose tracer split"
        case, qdiff, f, fate = found
        assert len(fate.new_chunks) >= 2
        assert f.target_in_index_after is True
        after_dense = qdiff.after.run.stages.dense
        pos = {cid: i for i, (cid, _) in enumerate(after_dense, start=1)}
        expected = min((pos[c] for c in fate.new_chunks if c in pos), default=None)
        assert f.dense["rank_after"] == expected

    def test_prd_10_4_funnel_dict_keys(self, chunker_scenario):
        sc = chunker_scenario
        case, qdiff, f = _regressed_funnels(sc)[0]
        fd = f.to_dict()
        assert set(fd) == {
            "target_chunk_before", "target_in_index_after",
            "dense", "sparse", "fused", "rerank", "ann_divergence",
        }
        assert set(fd["dense"]) == {"rank_before", "rank_after", "shadow_exact_rank_after"}
        assert set(fd["sparse"]) == {"rank_before", "rank_after"}
        assert set(fd["fused"]) == {"rank_before", "rank_after"}
        assert set(fd["rerank"]) == {"in_candidates_after"}
        assert isinstance(fd["ann_divergence"], bool)
        assert isinstance(fd["target_in_index_after"], bool)

    def test_shadow_rank_present(self, chunker_scenario):
        sc = chunker_scenario
        for case, qdiff, f in _regressed_funnels(sc):
            assert "shadow_exact_rank_after" in f.dense


def test_exact_adapter_never_diverges_on_split_tracer(tmp_path_factory, corpus_dir):
    # Finding #2: a split whose best fragment differs from new_chunks[0] must not
    # fabricate ANN divergence against a purely exact index. Shadow now uses the
    # same descendant set as the live comparison, so divergence is False and the
    # shadow rank matches the live dense rank.
    root = tmp_path_factory.mktemp("funnel_ann")
    store = ProjectStore(root)
    man_a = ingest(store, corpus_dir, build_pipeline({}), None).manifest
    adapter = LocalIndexAdapter(root / "idx")  # exact mode: zero approximation
    man_b = ingest(store, corpus_dir, build_pipeline({
        "chunker": {"tool": chunkers.FIXED_TOKEN, "params": {"max_tokens": 60, "overlap": 0}},
    }), adapter=adapter).manifest
    dataset = _dataset()
    d = diff(store, man_a, man_b, dataset,
             evaluate(store, man_a, dataset), evaluate(store, man_b, dataset))
    engine_a = RetrievalEngine(store, man_a)
    engine_b_live = RetrievalEngine(store, man_b, adapter=adapter)
    engine_b = RetrievalEngine(store, man_b)
    checked_split = False
    for case in dataset.cases:
        qdiff = d.queries[case.id]
        live_run_b = engine_b_live.run_query(case.id, case.question)
        f = funnel_for_query(engine_a, engine_b, qdiff, case, d.alignment, live_run_b=live_run_b)
        assert f.ann_divergence is False, f"exact index must not diverge for {case.id}"
        fate = d.alignment.get(f.target_chunk_before)
        if fate is not None and fate.cls == "split":
            checked_split = True
            # shadow rank and live dense rank use the same descendant set now.
            assert f.dense["shadow_exact_rank_after"] == f.dense["rank_after"]
    assert checked_split, "expected at least one split-tracer query to exercise the fix"


@pytest.fixture(scope="module")
def retrieve_scenario(tmp_path_factory, corpus_dir):
    root = tmp_path_factory.mktemp("funnel_retrieve")
    store = ProjectStore(root)
    man_a = ingest(store, corpus_dir, build_pipeline({}), None).manifest
    man_b = ingest(store, corpus_dir, build_pipeline({
        "retrieve": {"top_k": 10,
                     "hybrid": {"sparse": "bm25", "fusion": "weighted", "bm25_weight": 0.0}},
    }), None).manifest
    dataset = _dataset()
    eval_a = evaluate(store, man_a, dataset)
    eval_b = evaluate(store, man_b, dataset)
    d = diff(store, man_a, man_b, dataset, eval_a, eval_b)
    return SimpleNamespace(
        store=store, dataset=dataset, diff=d,
        engine_a=RetrievalEngine(store, man_a),
        engine_b=RetrievalEngine(store, man_b),
    )


class TestRetrieveScenario:
    def test_config_diff_is_retrieve(self, retrieve_scenario):
        assert set(retrieve_scenario.diff.config_diff) == {"retrieve"}

    def test_regressed_query_implicates_retrieve_at_fused(self, retrieve_scenario):
        sc = retrieve_scenario
        found = None
        for case, qdiff, f in _regressed_funnels(sc):
            stage = failing_stage(f, 5)
            if stage == "fused":
                found = (case, f, stage)
                break
        assert found is not None, "expected a regressed query failing at the fused stage"
        case, f, stage = found
        assert f.target_in_index_after is True
        assert "retrieve" in implicated_factors(stage, sc.diff.config_diff)

    def test_dense_and_sparse_unchanged(self, retrieve_scenario):
        sc = retrieve_scenario
        for case, qdiff, f in _regressed_funnels(sc):
            assert f.dense["rank_before"] == f.dense["rank_after"]
            assert f.sparse["rank_before"] == f.sparse["rank_after"]


@pytest.fixture(scope="module")
def ann_scenario(tmp_path_factory, corpus_dir):
    root = tmp_path_factory.mktemp("funnel_ann")
    store = ProjectStore(root)
    adapter = LocalIndexAdapter(root / "ann", ann_mode=True, ann_sigma=0.5, seed=3)
    man = ingest(store, corpus_dir, build_pipeline({}), adapter).manifest
    dataset = _dataset()
    ev = evaluate(store, man, dataset)
    d = diff(store, man, man, dataset, ev, ev)
    return SimpleNamespace(
        store=store, dataset=dataset, diff=d,
        engine=RetrievalEngine(store, man),
        engine_live=RetrievalEngine(store, man, adapter=adapter),
    )


class TestAnnScenario:
    def test_ann_divergence_detected(self, ann_scenario):
        sc = ann_scenario
        diverged = []
        for case in sc.dataset.cases:
            qdiff = sc.diff.queries[case.id]
            live_run = sc.engine_live.run_query(case.id, case.question)
            f = funnel_for_query(sc.engine, sc.engine, qdiff, case,
                                 sc.diff.alignment, live_run_b=live_run)
            if f.ann_divergence:
                diverged.append((case, f))
        assert diverged, "ann noise should diverge live vs shadow for some query"

    def test_replay_never_diverges(self, ann_scenario):
        sc = ann_scenario
        for case in sc.dataset.cases:
            qdiff = sc.diff.queries[case.id]
            f = funnel_for_query(sc.engine, sc.engine, qdiff, case,
                                 sc.diff.alignment, live_run_b=None)
            assert f.ann_divergence is False
