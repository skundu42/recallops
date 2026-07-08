from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from recallops import hashing
from recallops.adapters.local import LocalIndexAdapter
from recallops.evalrunner import evaluate, hit_rate_at_k, mrr, ndcg_at_k, recall_at_k
from recallops.ingest import build_pipeline, ingest
from recallops.models import EvalResult, GoldenCase, GoldenDataset, QueryRun
from recallops.pipeline import chunkers
from recallops.retrieval import RetrievalEngine
from recallops.store import ProjectStore

RANKED = ["a", "b", "c", "d", "e"]


class TestRecallAtK:
    def test_single_expected_at_top(self):
        assert recall_at_k(RANKED, ["a"], 1) == 1.0

    def test_multi_expected_partial(self):
        assert recall_at_k(RANKED, ["a", "c"], 2) == 0.5

    def test_multi_expected_full(self):
        assert recall_at_k(RANKED, ["a", "c"], 3) == 1.0

    def test_expected_absent(self):
        assert recall_at_k(RANKED, ["z"], 5) == 0.0

    def test_empty_ranking(self):
        assert recall_at_k([], ["a"], 5) == 0.0

    def test_empty_expected(self):
        assert recall_at_k(RANKED, [], 5) == 0.0


class TestHitRateAtK:
    def test_hit_inside_k(self):
        assert hit_rate_at_k(RANKED, ["b"], 2) == 1.0

    def test_hit_outside_k(self):
        assert hit_rate_at_k(RANKED, ["b"], 1) == 0.0

    def test_any_expected_counts(self):
        assert hit_rate_at_k(RANKED, ["z", "c"], 3) == 1.0

    def test_miss(self):
        assert hit_rate_at_k(RANKED, ["z"], 5) == 0.0


class TestMRR:
    def test_first_rank(self):
        assert mrr(RANKED, ["a"]) == 1.0

    def test_third_rank(self):
        assert mrr(RANKED, ["c"]) == pytest.approx(1.0 / 3.0)

    def test_first_expected_hit_wins(self):
        assert mrr(RANKED, ["c", "a"]) == 1.0

    def test_absent(self):
        assert mrr(RANKED, ["z"]) == 0.0


class TestNDCGAtK:
    def test_perfect_ranking(self):
        assert ndcg_at_k(["a", "b"], ["a", "b"], 2) == pytest.approx(1.0)

    def test_hand_computed_single_hit(self):
        got = ndcg_at_k(["a", "b", "c", "d"], ["b", "d"], 3)
        dcg = 1.0 / math.log2(3)
        idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
        assert got == pytest.approx(dcg / idcg)

    def test_hand_computed_two_hits(self):
        got = ndcg_at_k(["a", "b", "c", "d"], ["b", "d"], 4)
        dcg = 1.0 / math.log2(3) + 1.0 / math.log2(5)
        idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
        assert got == pytest.approx(dcg / idcg)

    def test_idcg_truncated_at_k(self):
        got = ndcg_at_k(["a", "b"], ["a", "x", "y"], 2)
        idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
        assert got == pytest.approx(1.0 / idcg)

    def test_no_relevant(self):
        assert ndcg_at_k(RANKED, ["z"], 5) == 0.0

    def test_empty_expected(self):
        assert ndcg_at_k(RANKED, [], 5) == 0.0


CASES = [
    ("q1", "How are annual plan refunds prorated after the 30-day window closes?",
     "billing/refunds.md"),
    ("q2", "Which single sign-on protocols like SAML and OIDC are supported?",
     "security/sso.md"),
    ("q3", "What is the default quota of requests per minute for read endpoints?",
     "api/rate-limits.md"),
    ("q4", "What happens when an engineer declares an incident with the incident bot?",
     "ops/incident-runbook.md"),
    ("q5", "How much advance notice is given before onboarding new subprocessors?",
     "legal/dpa.md"),
    ("q6", "When is production access granted to new engineers during onboarding?",
     "hr/onboarding.md"),
]

K_VALUES = (1, 5, 10)


@pytest.fixture(scope="module")
def env(tmp_path_factory, corpus_dir):
    root = tmp_path_factory.mktemp("evalproj")
    store = ProjectStore(root)
    adapter = LocalIndexAdapter(root / "index")
    pipeline = build_pipeline({
        "chunker": {"tool": chunkers.MARKDOWN_HEADING,
                    "params": {"max_tokens": 800, "overlap": 120}},
        "retrieve": {"top_k": 10,
                     "hybrid": {"sparse": "bm25", "fusion": "weighted", "bm25_weight": 0.45}},
    })
    report = ingest(store, corpus_dir, pipeline, adapter)
    dataset = GoldenDataset("hand-v1", [
        GoldenCase(id=cid, question=q, expected_sources=[src], tags=[])
        for cid, q, src in CASES
    ])
    return SimpleNamespace(store=store, adapter=adapter,
                           manifest=report.manifest, dataset=dataset)


@pytest.fixture(scope="module")
def replay_eval(env) -> EvalResult:
    return evaluate(env.store, env.manifest, env.dataset, k_values=K_VALUES)


class TestEvaluate:
    def test_acceptance_recall_at_5(self, replay_eval):
        assert replay_eval.aggregate["recall@5"] >= 0.8

    def test_per_query_metric_keys(self, replay_eval):
        assert set(replay_eval.per_query) == {cid for cid, _, _ in CASES}
        expected_keys = {"mrr"}
        for k in K_VALUES:
            expected_keys |= {f"recall@{k}", f"hit_rate@{k}", f"ndcg@{k}"}
        for qe in replay_eval.per_query.values():
            assert set(qe.metrics) == expected_keys
            assert set(qe.hit_at) == set(K_VALUES)

    def test_query_eval_carries_run(self, replay_eval, env):
        case_by_id = {c.id: c for c in env.dataset.cases}
        for qid, qe in replay_eval.per_query.items():
            assert isinstance(qe.run, QueryRun)
            assert qe.run.query_id == qid
            assert qe.run.question == case_by_id[qid].question
            assert qe.ranked_chunks == qe.run.final

    def test_ranked_docs_is_ordered_dedupe(self, replay_eval, env):
        doc_map = RetrievalEngine(env.store, env.manifest).chunk_doc_map()
        for qe in replay_eval.per_query.values():
            docs = []
            for cid, _ in qe.ranked_chunks:
                doc = doc_map[cid]
                if doc not in docs:
                    docs.append(doc)
            assert qe.ranked_docs == docs

    def test_target_rank_is_first_expected_chunk(self, replay_eval, env):
        doc_map = RetrievalEngine(env.store, env.manifest).chunk_doc_map()
        case_by_id = {c.id: c for c in env.dataset.cases}
        for qid, qe in replay_eval.per_query.items():
            want = set(case_by_id[qid].expected_sources)
            ranks = [rank for rank, (cid, _) in enumerate(qe.ranked_chunks, start=1)
                     if doc_map[cid] in want]
            assert qe.target_rank == (ranks[0] if ranks else None)

    def test_aggregate_is_arithmetic_mean(self, replay_eval):
        n = len(replay_eval.per_query)
        for key, value in replay_eval.aggregate.items():
            mean = sum(qe.metrics[key] for qe in replay_eval.per_query.values()) / n
            assert value == pytest.approx(mean)

    def test_created_at_empty_for_determinism(self, replay_eval):
        assert replay_eval.created_at == ""
        assert replay_eval.mode == "replay"
        assert replay_eval.adapter == "none"
        assert replay_eval.k_values == K_VALUES


class TestModes:
    def test_live_requires_adapter(self, env):
        with pytest.raises(ValueError):
            evaluate(env.store, env.manifest, env.dataset, adapter=None, mode="live")

    def test_unknown_mode_rejected(self, env):
        with pytest.raises(ValueError):
            evaluate(env.store, env.manifest, env.dataset, mode="offline")

    def test_replay_equals_live(self, replay_eval, env):
        live = evaluate(env.store, env.manifest, env.dataset,
                        adapter=env.adapter, k_values=K_VALUES, mode="live")
        assert live.mode == "live"
        assert live.adapter == "local"
        assert live.run_id != replay_eval.run_id
        assert set(live.per_query) == set(replay_eval.per_query)
        for qid, qe in live.per_query.items():
            assert qe.ranked_chunks == replay_eval.per_query[qid].ranked_chunks
            assert qe.ranked_docs == replay_eval.per_query[qid].ranked_docs
            assert qe.metrics == replay_eval.per_query[qid].metrics
        assert live.aggregate == replay_eval.aggregate


class TestPersistence:
    def test_get_eval_roundtrip(self, replay_eval, env):
        loaded = env.store.get_eval(replay_eval.run_id)
        assert loaded.to_dict() == replay_eval.to_dict()
        assert hashing.canonical_json(loaded.to_dict()) == \
            hashing.canonical_json(replay_eval.to_dict())

    def test_find_eval(self, replay_eval, env):
        found = env.store.find_eval(env.manifest.snapshot_id, env.dataset.dataset_id)
        assert found is not None
        assert found.snapshot_id == env.manifest.snapshot_id
        assert found.dataset_id == env.dataset.dataset_id


class TestRunId:
    def test_run_id_formula(self, replay_eval, env):
        expected = "ev_" + hashing.h(
            env.manifest.snapshot_id, env.dataset.dataset_id,
            "replay", "none", str(K_VALUES),
        )
        assert replay_eval.run_id == expected

    def test_run_id_deterministic(self, replay_eval, env):
        again = evaluate(env.store, env.manifest, env.dataset, k_values=K_VALUES)
        assert again.run_id == replay_eval.run_id
        assert again.to_dict() == replay_eval.to_dict()

    def test_run_id_varies_with_k_values(self, replay_eval, env):
        other = evaluate(env.store, env.manifest, env.dataset, k_values=(1, 3))
        assert other.run_id != replay_eval.run_id


class TestDuplicateContentDocs:
    def test_expected_source_matches_any_alias_path(self, tmp_path):
        # Byte-identical files at different paths share one doc_id; a golden
        # case naming EITHER path must score when the shared content is
        # retrieved (a collapsed doc-map silently zeroes all but one path).
        store = ProjectStore(tmp_path / "proj")
        docs = tmp_path / "docs"
        docs.mkdir()
        text = "# Refunds\n\nRefunds are processed within five business days.\n"
        (docs / "a_dup.md").write_text(text)
        (docs / "z_dup.md").write_text(text)
        (docs / "other.md").write_text("# Shipping\n\nShipping takes two days.\n")
        manifest = ingest(store, docs, build_pipeline({}), adapter=None).manifest
        dataset = GoldenDataset("gold-v1", [
            GoldenCase("q_first", "how long do refunds take", ["a_dup.md"]),
            GoldenCase("q_last", "how long do refunds take", ["z_dup.md"]),
        ])
        result = evaluate(store, manifest, dataset)
        for qid in ("q_first", "q_last"):
            assert result.per_query[qid].metrics["recall@5"] == 1.0, qid
            assert result.per_query[qid].target_rank is not None, qid
