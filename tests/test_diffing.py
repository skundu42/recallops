from __future__ import annotations

from types import SimpleNamespace

import pytest

from recallops import hashing
from recallops.diffing import align_chunks, classify_query, diff, fuzzy_align_chunks
from recallops.evalrunner import evaluate
from recallops.ingest import build_pipeline, ingest
from recallops.models import (
    ChunkRecord,
    DiffResult,
    GoldenCase,
    GoldenDataset,
    QueryEval,
)
from recallops.pipeline import chunkers
from recallops.retrieval import chunkset_key_from_uri
from recallops.store import ProjectStore


def make_chunk(doc: str, start: int, end: int, text: str | None = None) -> ChunkRecord:
    body = text if text is not None else "x" * (end - start)
    return ChunkRecord(
        chunk_id=hashing.chunk_id(doc, start, end, body),
        doc_id=doc,
        span_start=start,
        span_end=end,
        ordinal=0,
        text=body,
        text_hash=hashing.text_hash(body),
        parse_stage_id="parse",
        chunk_stage_id="chunk",
    )


class TestAlignChunks:
    def test_identical_span_is_intact_score_one(self):
        old = make_chunk("doc_a", 0, 100)
        new = make_chunk("doc_a", 0, 100)
        fates = align_chunks([old], [new])
        fate = fates[old.chunk_id]
        assert fate.cls == "intact"
        assert fate.alignment_score == pytest.approx(1.0)
        assert fate.new_chunks == [new.chunk_id]
        assert fate.old_chunk == old.chunk_id

    def test_intact_threshold_edges(self):
        old = make_chunk("doc_a", 0, 100)
        barely = make_chunk("doc_a", 5, 100)
        assert align_chunks([old], [barely])[old.chunk_id].cls == "intact"
        shifted = make_chunk("doc_a", 6, 100)
        assert align_chunks([old], [shifted])[old.chunk_id].cls == "boundary-shifted"

    def test_split_in_two_names_both_children(self):
        old = make_chunk("doc_a", 0, 100)
        left = make_chunk("doc_a", 0, 50)
        right = make_chunk("doc_a", 50, 100)
        fate = align_chunks([old], [right, left])[old.chunk_id]
        assert fate.cls == "split"
        assert fate.alignment_score == pytest.approx(1.0)
        assert set(fate.new_chunks) == {left.chunk_id, right.chunk_id}
        assert fate.new_chunks[0] == left.chunk_id

    def test_split_children_sorted_by_overlap_desc(self):
        old = make_chunk("doc_a", 0, 100)
        big = make_chunk("doc_a", 0, 80)
        small = make_chunk("doc_a", 80, 100)
        fate = align_chunks([old], [small, big])[old.chunk_id]
        assert fate.cls == "split"
        assert fate.new_chunks == [big.chunk_id, small.chunk_id]

    def test_merged_into_larger_chunk(self):
        old = make_chunk("doc_a", 0, 40)
        big = make_chunk("doc_a", 0, 100)
        fate = align_chunks([old], [big])[old.chunk_id]
        assert fate.cls == "merged"
        assert fate.alignment_score == pytest.approx(1.0)
        assert fate.new_chunks == [big.chunk_id]

    def test_boundary_shift_dominant_overlap(self):
        old = make_chunk("doc_a", 0, 100)
        new = make_chunk("doc_a", 30, 130)
        fate = align_chunks([old], [new])[old.chunk_id]
        assert fate.cls == "boundary-shifted"
        assert fate.alignment_score == pytest.approx(0.7)
        assert fate.new_chunks == [new.chunk_id]

    def test_removed_text_is_dropped(self):
        old = make_chunk("doc_a", 0, 100)
        far = make_chunk("doc_a", 200, 300)
        fate = align_chunks([old], [far])[old.chunk_id]
        assert fate.cls == "dropped"
        assert fate.alignment_score == 0.0
        assert fate.new_chunks == []

    def test_low_combined_overlap_is_dropped(self):
        old = make_chunk("doc_a", 0, 100)
        slivers = [make_chunk("doc_a", 0, 20), make_chunk("doc_a", 30, 45)]
        fate = align_chunks([old], slivers)[old.chunk_id]
        assert fate.cls == "dropped"
        assert fate.alignment_score == pytest.approx(0.35)

    def test_alignment_scoped_to_same_doc(self):
        old = make_chunk("doc_a", 0, 100)
        other_doc = make_chunk("doc_b", 0, 100)
        fate = align_chunks([old], [other_doc])[old.chunk_id]
        assert fate.cls == "dropped"
        assert fate.new_chunks == []

    def test_doc_missing_in_b_drops_its_chunks(self):
        kept = make_chunk("doc_a", 0, 50)
        gone = make_chunk("doc_b", 0, 50)
        fates = align_chunks([kept, gone], [make_chunk("doc_a", 0, 50)])
        assert fates[kept.chunk_id].cls == "intact"
        assert fates[gone.chunk_id].cls == "dropped"

    def test_one_fate_per_old_chunk(self):
        olds = [make_chunk("doc_a", 0, 50), make_chunk("doc_a", 50, 100)]
        news = [make_chunk("doc_a", 0, 100)]
        fates = align_chunks(olds, news)
        assert set(fates) == {o.chunk_id for o in olds}


class TestFuzzyAlignChunks:
    def test_identical_text_is_intact(self):
        old = make_chunk("doc_a", 0, 30, text="alpha widget bootstrap manifest checksum")
        new = make_chunk("doc_a", 0, 30, text="alpha widget bootstrap manifest checksum")
        fate = fuzzy_align_chunks([old], [new])[old.chunk_id]
        assert fate.cls == "intact"
        assert fate.alignment_score == pytest.approx(1.0)
        assert fate.new_chunks == [new.chunk_id]

    def test_marker_stripping_still_intact(self):
        old = make_chunk("doc_a", 0, 40, text="## Setup\n\nInstall the *alpha* widget now.")
        new = make_chunk("doc_a", 0, 36, text="Setup\n\nInstall the alpha widget now.")
        fate = fuzzy_align_chunks([old], [new])[old.chunk_id]
        assert fate.cls == "intact"

    def test_partial_overlap_is_boundary_shifted(self):
        old = make_chunk("doc_a", 0, 60,
                         text="alpha beta gamma delta epsilon zeta eta theta iota kappa")
        new = make_chunk("doc_a", 0, 58,
                         text="alpha beta gamma delta epsilon zeta eta theta lambda mu")
        fate = fuzzy_align_chunks([old], [new])[old.chunk_id]
        assert fate.cls == "boundary-shifted"
        assert fate.alignment_score == pytest.approx(8 / 12)
        assert fate.new_chunks == [new.chunk_id]

    def test_unrelated_text_is_dropped(self):
        old = make_chunk("doc_a", 0, 20, text="alpha beta gamma delta")
        new = make_chunk("doc_a", 0, 20, text="omega psi chi phi")
        fate = fuzzy_align_chunks([old], [new])[old.chunk_id]
        assert fate.cls == "dropped"
        assert fate.new_chunks == []

    def test_scoped_to_same_doc(self):
        old = make_chunk("doc_a", 0, 20, text="alpha beta gamma delta")
        new = make_chunk("doc_b", 0, 20, text="alpha beta gamma delta")
        assert fuzzy_align_chunks([old], [new])[old.chunk_id].cls == "dropped"

    def test_classes_limited_to_fuzzy_set(self):
        olds = [
            make_chunk("doc_a", 0, 20, text="alpha beta gamma delta"),
            make_chunk("doc_a", 20, 40, text="epsilon zeta eta theta"),
            make_chunk("doc_b", 0, 20, text="iota kappa lambda mu"),
        ]
        news = [
            make_chunk("doc_a", 0, 20, text="alpha beta gamma delta"),
            make_chunk("doc_a", 20, 44, text="epsilon zeta eta theta iota nu"),
        ]
        fates = fuzzy_align_chunks(olds, news)
        assert {f.cls for f in fates.values()} <= {"intact", "boundary-shifted", "dropped"}


def make_eval(metric: float, chunks: list[tuple[str, float]],
              target_rank: int | None = None,
              metric_name: str = "recall@5") -> QueryEval:
    return QueryEval(
        query_id="q",
        ranked_chunks=list(chunks),
        ranked_docs=[],
        target_rank=target_rank,
        hit_at={},
        metrics={metric_name: metric},
        run=None,
    )


FIVE = [("c1", 1.0), ("c2", 0.9), ("c3", 0.8), ("c4", 0.7), ("c5", 0.6)]


class TestClassifyQuery:
    def test_regressed(self):
        cls, _ = classify_query(make_eval(1.0, FIVE), make_eval(0.0, FIVE))
        assert cls == "regressed"

    def test_improved(self):
        cls, _ = classify_query(make_eval(0.0, FIVE), make_eval(1.0, FIVE))
        assert cls == "improved"

    def test_changed_top_k(self):
        after = [("c1", 1.0), ("c2", 0.9), ("c3", 0.8), ("c4", 0.7), ("c9", 0.6)]
        cls, _ = classify_query(make_eval(1.0, FIVE), make_eval(1.0, after))
        assert cls == "changed-top-k"

    def test_unchanged_ignores_order_within_top_k(self):
        after = [("c2", 1.0), ("c1", 0.9), ("c3", 0.8), ("c4", 0.7), ("c5", 0.6)]
        cls, _ = classify_query(make_eval(1.0, FIVE), make_eval(1.0, after))
        assert cls == "unchanged"

    def test_k_parsed_from_primary_metric(self):
        before = make_eval(1.0, [("a", 1.0), ("b", 0.5)], metric_name="recall@1")
        after = make_eval(1.0, [("a", 1.0), ("c", 0.5)], metric_name="recall@1")
        cls, _ = classify_query(before, after, primary_metric="recall@1")
        assert cls == "unchanged"

    def test_stable_with_zero_epsilon(self):
        _, stability = classify_query(make_eval(1.0, FIVE), make_eval(0.0, FIVE, target_rank=5))
        assert stability == "stable"

    def test_target_present_gap_vs_kth(self):
        after = make_eval(1.0, FIVE, target_rank=1)
        _, unstable = classify_query(make_eval(1.0, FIVE), after, epsilon=0.5)
        assert unstable == "unstable"
        _, stable = classify_query(make_eval(1.0, FIVE), after, epsilon=0.3)
        assert stable == "stable"

    def test_target_far_below_k_is_stable_not_a_near_tie(self):
        # Finding #7: a target that crashed far below k (rank 8, score 0.3, vs the
        # k-th score 0.6) has a real 0.3 gap and must be "stable", not misflagged
        # as a near-tie and excluded from the McNemar flip counts.
        chunks = FIVE + [("c6", 0.5), ("c7", 0.4), ("c8", 0.3)]
        after = make_eval(0.0, chunks, target_rank=8)
        _, stability = classify_query(make_eval(1.0, chunks), after, epsilon=0.1)
        assert stability == "stable"
        # Only a genuinely tiny epsilon-dwarfing gap would flag it unstable.
        _, unstable = classify_query(make_eval(1.0, chunks), after, epsilon=0.5)
        assert unstable == "unstable"

    def test_target_absent_is_always_stable(self):
        # Finding #2: a target that fell below the served list is a definite severe
        # regression, never a near-tie, it must NOT be excluded from flip counts,
        # regardless of the unrelated k/k+1 boundary gap or how large epsilon is.
        chunks = FIVE + [("c6", 0.55)]  # rank5 0.6 vs rank6 0.55: a tiny k/k+1 gap
        after = make_eval(0.0, chunks, target_rank=None)
        _, s1 = classify_query(make_eval(1.0, chunks), after, epsilon=0.1)
        assert s1 == "stable"
        _, s2 = classify_query(make_eval(1.0, chunks), after, epsilon=100.0)
        assert s2 == "stable"

    def test_target_absent_without_next_is_stable(self):
        after = make_eval(0.0, FIVE, target_rank=None)
        _, stability = classify_query(make_eval(1.0, FIVE), after, epsilon=100.0)
        assert stability == "stable"


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


@pytest.fixture(scope="module")
def env(tmp_path_factory, corpus_dir):
    root = tmp_path_factory.mktemp("diffproj")
    store = ProjectStore(root)
    manifest_a = ingest(store, corpus_dir, build_pipeline({}), None).manifest
    manifest_b = ingest(store, corpus_dir, build_pipeline({
        "chunker": {"tool": chunkers.FIXED_TOKEN, "params": {"max_tokens": 60, "overlap": 0}},
    }), None).manifest
    manifest_c = ingest(store, corpus_dir, build_pipeline({
        "parser": {"tool": "markdown-v2"},
    }), None).manifest
    dataset = GoldenDataset("hand-v1", [
        GoldenCase(id=cid, question=q, expected_sources=list(sources), tags=[])
        for cid, q, sources in CASES
    ])
    return SimpleNamespace(
        store=store,
        manifest_a=manifest_a,
        manifest_b=manifest_b,
        manifest_c=manifest_c,
        dataset=dataset,
        eval_a=evaluate(store, manifest_a, dataset),
        eval_b=evaluate(store, manifest_b, dataset),
        eval_c=evaluate(store, manifest_c, dataset),
    )


@pytest.fixture(scope="module")
def diff_ab(env) -> DiffResult:
    return diff(env.store, env.manifest_a, env.manifest_b, env.dataset,
                env.eval_a, env.eval_b)


class TestDiffChunkerChange:
    def test_acceptance_finds_regressed_queries(self, diff_ab):
        assert len(diff_ab.by_class("regressed")) >= 2

    def test_config_diff_keys(self, diff_ab):
        assert set(diff_ab.config_diff) == {"chunk"}
        assert diff_ab.config_diff["chunk"]["before"]["tool"] == chunkers.MARKDOWN_HEADING
        assert diff_ab.config_diff["chunk"]["after"]["tool"] == chunkers.FIXED_TOKEN

    def test_flags(self, diff_ab):
        assert diff_ab.corpus_changed is False
        assert diff_ab.parser_changed is False
        assert diff_ab.alignment_available is True

    def test_metric_deltas_negative_for_recall_at_5(self, diff_ab, env):
        assert diff_ab.metric_deltas["recall@5"] < 0
        assert diff_ab.metric_deltas["recall@5"] == pytest.approx(
            env.eval_b.aggregate["recall@5"] - env.eval_a.aggregate["recall@5"])

    def test_alignment_covers_every_old_chunk(self, diff_ab, env):
        old = env.store.get_chunks(
            chunkset_key_from_uri(env.manifest_a.artifacts["chunks_uri"]))
        assert set(diff_ab.alignment) == {r.chunk_id for r in old}

    def test_alignment_has_split_fates(self, diff_ab):
        classes = [f.cls for f in diff_ab.alignment.values()]
        assert "split" in classes
        for fate in diff_ab.alignment.values():
            if fate.cls == "split":
                assert len(fate.new_chunks) >= 2

    def test_per_query_metric_delta(self, diff_ab, env):
        for qid, qdiff in diff_ab.queries.items():
            expected = (env.eval_b.per_query[qid].metrics["recall@5"]
                        - env.eval_a.per_query[qid].metrics["recall@5"])
            assert qdiff.metric_delta["recall@5"] == pytest.approx(expected)
            assert qdiff.before.query_id == qid
            assert qdiff.after.query_id == qid

    def test_diff_id_formula(self, diff_ab, env):
        assert diff_ab.diff_id == "diff_" + hashing.h(
            env.manifest_a.snapshot_id, env.manifest_b.snapshot_id,
            env.dataset.dataset_id)

    def test_persisted_and_roundtrips(self, diff_ab, env):
        stored = env.store.get_json("diff", diff_ab.diff_id)
        assert stored == diff_ab.to_dict()
        assert DiffResult.from_dict(stored).to_dict() == diff_ab.to_dict()

    def test_epsilon_marks_near_ties_unstable(self, env):
        wide = diff(env.store, env.manifest_a, env.manifest_b, env.dataset,
                    env.eval_a, env.eval_b, epsilon=1000.0)
        assert any(q.stability == "unstable" for q in wide.queries.values())


class TestDiffParserChange:
    def test_fuzzy_path_flags_and_classes(self, env):
        d = diff(env.store, env.manifest_a, env.manifest_c, env.dataset,
                 env.eval_a, env.eval_c)
        assert set(d.config_diff) == {"parse"}
        assert d.parser_changed is True
        assert d.alignment_available is False
        assert d.corpus_changed is False
        old = env.store.get_chunks(
            chunkset_key_from_uri(env.manifest_a.artifacts["chunks_uri"]))
        assert set(d.alignment) == {r.chunk_id for r in old}
        assert {f.cls for f in d.alignment.values()} <= {
            "intact", "boundary-shifted", "dropped"}


class TestDiffCorpusChange:
    def test_corpus_factor_and_dropped_chunks(self, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "alpha.md").write_text(
            "# Alpha\n\nThe alpha widget bootstraps with a manifest checksum.\n")
        (docs / "beta.md").write_text(
            "# Beta\n\nThe beta gateway routes traffic by header affinity.\n")
        (docs / "gamma.md").write_text(
            "# Gamma\n\nGamma storage replicates each object three times.\n")
        store = ProjectStore(tmp_path / "proj")
        pipeline = build_pipeline({})
        manifest_a = ingest(store, docs, pipeline, None).manifest
        (docs / "gamma.md").unlink()
        manifest_b = ingest(store, docs, pipeline, None).manifest

        dataset = GoldenDataset("mini-v1", [
            GoldenCase(id="q1", question="How does the alpha widget bootstrap?",
                       expected_sources=["alpha.md"], tags=[]),
            GoldenCase(id="q2", question="How does the beta gateway route traffic?",
                       expected_sources=["beta.md"], tags=[]),
        ])
        eval_a = evaluate(store, manifest_a, dataset)
        eval_b = evaluate(store, manifest_b, dataset)
        d = diff(store, manifest_a, manifest_b, dataset, eval_a, eval_b)

        assert d.corpus_changed is True
        assert d.parser_changed is False
        assert d.alignment_available is True
        assert set(d.config_diff) == {"corpus"}
        assert d.config_diff["corpus"] == {
            "before": {"merkle_root": manifest_a.corpus.merkle_root},
            "after": {"merkle_root": manifest_b.corpus.merkle_root},
        }

        old = store.get_chunks(chunkset_key_from_uri(manifest_a.artifacts["chunks_uri"]))
        sources = {doc["doc_id"]: doc["source_path"]
                   for doc in store.docs_for_merkle(manifest_a.corpus.merkle_root)}
        for record in old:
            fate = d.alignment[record.chunk_id]
            if sources[record.doc_id] == "gamma.md":
                assert fate.cls == "dropped"
                assert fate.new_chunks == []
            else:
                assert fate.cls == "intact"
