from __future__ import annotations

import pytest

from recallops.models import (
    AttributionReport,
    ChunkFate,
    FunnelReport,
    GoldenCase,
    Hypothesis,
    VerifiedCause,
)
from recallops.narrative import narrative_faithfulness_audit, render_narrative

CASE = GoldenCase(id="q_014", question="What is the refund policy?",
                  expected_sources=["billing/refunds.md"], tags=["billing", "exact-term"])

# Failure-mechanism verbs that FR-8.4 forbids for an embedding-swap narrative.
MECHANISM_VERBS = ("split", "merged", "demoted", "outrank", "boundary-shifted",
                   "diluting", "dropped")


def _funnel(ann_divergence: bool = False, **over) -> FunnelReport:
    base = dict(
        target_chunk_before="ch_5f2e",
        target_in_index_after=True,
        dense={"rank_before": 1, "rank_after": 9, "shadow_exact_rank_after": 9},
        sparse={"rank_before": 2, "rank_after": 14},
        fused={"rank_before": 1, "rank_after": 11},
        rerank={"in_candidates_after": False},
        ann_divergence=ann_divergence,
    )
    base.update(over)
    return FunnelReport(**base)


def _report(**over) -> AttributionReport:
    base = dict(
        query_id="q_014",
        classification="regressed",
        stability="stable",
        funnel=_funnel(),
        chunk_fate=None,
        verified_causes=[],
        hypotheses=[],
        narrative="",
    )
    base.update(over)
    return AttributionReport(**base)


def _split_chunker_report() -> AttributionReport:
    return _report(
        chunk_fate=ChunkFate("split", 0.92, "ch_5f2e", ["ch_a01", "ch_a02"]),
        verified_causes=[VerifiedCause("chunker", "arm_B_with_chunker_A", 1, "verified")],
        hypotheses=[Hypothesis("bm25_weight", "unverified", "sparse rank drop co-occurred")],
    )


def test_split_chunker_narrative_has_fate_descendants_and_arm():
    rep = _split_chunker_report()
    text = render_narrative(rep, CASE)
    assert "split" in text
    assert "2 chunks" in text  # descendant count
    assert "(verified: arm_B_with_chunker_A)" in text


def test_split_narrative_is_faithful():
    rep = _split_chunker_report()
    text = render_narrative(rep, CASE)
    assert narrative_faithfulness_audit(rep, text) == []


@pytest.mark.parametrize("cls,token", [
    ("merged", "merged"),
    ("boundary-shifted", "boundary-shifted"),
    ("dropped", "dropped"),
])
def test_other_fate_mechanisms_render_and_are_faithful(cls, token):
    rep = _report(
        chunk_fate=ChunkFate(cls, 0.7, "ch_5f2e", ["ch_a01"]),
        verified_causes=[VerifiedCause("chunk", "arm_chunkfate", 3, "verified")],
    )
    text = render_narrative(rep, CASE)
    assert token in text
    assert "(verified: arm_chunkfate)" in text
    assert narrative_faithfulness_audit(rep, text) == []


def test_retrieve_cause_is_fusion_reweighting():
    rep = _report(
        verified_causes=[VerifiedCause("retrieve", "arm_retrieve_A", 2, "verified")],
    )
    text = render_narrative(rep, CASE)
    assert "Fusion re-weighting" in text
    assert "(verified: arm_retrieve_A)" in text
    assert narrative_faithfulness_audit(rep, text) == []


def test_embed_swap_is_characterization_not_mechanism():
    rep = _report(
        chunk_fate=None,
        verified_causes=[VerifiedCause("embed", "arm_embed_A", 1, "verified")],
    )
    text = render_narrative(rep, CASE)
    assert "black box" in text
    assert "per-tag" in text
    assert "(verified: arm_embed_A)" in text
    lowered = text.lower()
    for verb in MECHANISM_VERBS:
        assert verb not in lowered, f"embed narrative asserted mechanism verb {verb!r}"
    assert narrative_faithfulness_audit(rep, text) == []


def test_corpus_drift_narrative():
    rep = _report(
        verified_causes=[VerifiedCause("corpus", "arm_corpus_A", 1, "verified")],
    )
    text = render_narrative(rep, CASE)
    assert "corpus drift" in text
    assert "outranks" in text
    assert "no config change is responsible" in text
    assert "(verified: arm_corpus_A)" in text
    assert narrative_faithfulness_audit(rep, text) == []


def test_ann_divergence_appends_ef_search_note():
    rep = _report(
        funnel=_funnel(ann_divergence=True),
        verified_causes=[],
    )
    text = render_narrative(rep, CASE)
    assert "ef_search" in text
    assert "quantization" in text
    assert "rebuild variance" in text
    assert narrative_faithfulness_audit(rep, text) == []


def test_fallback_cites_only_funnel_numbers_and_labels_unverified():
    rep = _report(verified_causes=[], chunk_fate=None)
    text = render_narrative(rep, CASE)
    assert "Unverified:" in text
    # funnel rank numbers are cited
    assert "dense 1→9" in text
    assert "sparse 2→14" in text
    assert "fused 1→11" in text
    # no verified arm citation exists in the fallback
    assert "(verified:" not in text
    assert narrative_faithfulness_audit(rep, text) == []


def test_hypotheses_are_prefixed_unverified_with_evidence():
    rep = _report(
        verified_causes=[VerifiedCause("chunk", "arm_x", 1, "verified")],
        chunk_fate=ChunkFate("split", 0.9, "ch_5f2e", ["ch_a01", "ch_a02"]),
        hypotheses=[Hypothesis("bm25_weight", "unverified", "sparse rank drop co-occurred")],
    )
    text = render_narrative(rep, CASE)
    assert "Unverified: bm25_weight" in text
    assert "sparse rank drop co-occurred" in text
    assert narrative_faithfulness_audit(rep, text) == []


def test_audit_catches_injected_fake_arm_id():
    rep = _split_chunker_report()
    text = render_narrative(rep, CASE)
    tampered = text + " A second arm (verified: arm_DEADBEEF9999) also helped."
    violations = narrative_faithfulness_audit(rep, tampered)
    assert any("arm_DEADBEEF9999" in v for v in violations)
    # the genuine arm id must not be reported as a violation
    assert not any("arm_B_with_chunker_A" in v for v in violations)


def test_audit_catches_fabricated_fate_class():
    rep = _report(chunk_fate=None, verified_causes=[VerifiedCause("embed", "arm_e", 1, "verified")])
    text = render_narrative(rep, CASE)
    tampered = text + " The section was also merged into a neighbour."
    violations = narrative_faithfulness_audit(rep, tampered)
    assert any("merged" in v for v in violations)


def test_audit_catches_fabricated_ann_claim():
    rep = _report(funnel=_funnel(ann_divergence=False),
                  verified_causes=[VerifiedCause("embed", "arm_e", 1, "verified")])
    text = render_narrative(rep, CASE)
    tampered = text + " This looks like an ann index issue."
    violations = narrative_faithfulness_audit(rep, tampered)
    assert any("ann" in v for v in violations)


def test_render_is_deterministic():
    rep = _split_chunker_report()
    assert render_narrative(rep, CASE) == render_narrative(rep, CASE)
