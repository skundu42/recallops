"""Evidence-constrained narrative rendering (PRD FR-8.3-FR-8.5, FR-6.3, §13).

``render_narrative`` is a deterministic template renderer: every sentence's
facts come from the structured ``AttributionReport`` (chunk-alignment class,
funnel ranks, verified-cause arm ids, hypotheses). No LLM, no fabrication,this is the ``--no-llm`` default path (FR-8.3). Branches are keyed on the
structured signals: chunk-fate class (split/merged/boundary-shifted/dropped),
verified-cause factor (chunk / retrieve / embed / corpus), ANN divergence, and
a funnel-only fallback when nothing is verified. Verified sentences cite their
arm id as ``(verified: <arm_id>)`` (FR-8.5); every hypothesis is prefixed
``Unverified:`` and carries its evidence (FR-8.2).

``narrative_faithfulness_audit`` is the §13 narrative-faithfulness guard: it
scans a rendered narrative for controlled tokens (arm ids, chunk-fate classes,
stage names, factor names) that are not backed by the report's structured
evidence and returns them as violations, an audited-faithful narrative returns
``[]``.
"""
from __future__ import annotations

import re

from .models import (
    CHUNK_FATES,
    AttributionReport,
    GoldenCase,
    Hypothesis,
    VerifiedCause,
)

__all__ = ["render_narrative", "narrative_faithfulness_audit"]

ANN_NOTE = ("Index/approximation effect: check ef_search / quantization / rebuild "
            "variance.")

_FACTOR_CANON: dict[str, str] = {
    "chunk": "chunk", "chunker": "chunk", "chunking": "chunk",
    "embed": "embed", "embedding": "embed", "embeddings": "embed",
    "retrieve": "retrieve", "retrieval": "retrieve", "fusion": "retrieve",
    "bm25_weight": "retrieve", "hybrid": "retrieve",
    "corpus": "corpus",
    "parse": "parse", "parser": "parse", "parsing": "parse",
    "rerank": "rerank", "reranker": "rerank",
    "index": "index",
}

_FACTOR_WORDS: dict[str, str] = {
    "chunker": "chunk", "chunking": "chunk",
    "embedding": "embed", "embeddings": "embed",
    "corpus": "corpus",
    "fusion": "retrieve", "bm25_weight": "retrieve",
    "parser": "parse", "parsing": "parse",
}

_STAGE_UNIVERSE = ("dense", "sparse", "fused", "rerank", "index", "ann")
_ALWAYS_STAGES = frozenset({"dense", "sparse", "fused", "rerank", "index"})

_ARM_RE = re.compile(r"arm_[0-9A-Za-z_]+")


def _canon(factor: str) -> str:
    return _FACTOR_CANON.get(factor, factor)


def _fmt_score(value: float) -> str:
    return f"{value:.2f}"


def _rank(value: int | None) -> str:
    return "absent" if value is None else str(value)


def _rank_phrase(value: int | None) -> str:
    return f"rank {value}" if value is not None else "the target"


def _fate_sentence(fate) -> str:
    n = len(fate.new_chunks)
    score = _fmt_score(fate.alignment_score)
    if fate.cls == "split":
        return (f"The target section was split across {n} chunks by the chunker; each "
                f"fragment scores lower on both dense and BM25 (heading text no longer "
                f"in-chunk).")
    if fate.cls == "merged":
        return (f"The target section was merged into a larger chunk by the chunker, "
                f"diluting its match signal (alignment {score}).")
    if fate.cls == "boundary-shifted":
        return (f"The target chunk was boundary-shifted; the tracer text is only "
                f"partially contained in the best-matching new chunk (alignment "
                f"{score}).")
    if fate.cls == "dropped":
        return (f"The target chunk was dropped: no new chunk substantially covers it "
                f"(alignment {score}).")
    return ""


def _verified_sentence(vc: VerifiedCause) -> str:
    tag = f"(verified: {vc.arm_id})"
    where = _rank_phrase(vc.recovered_rank)
    factor = _canon(vc.factor)
    if factor == "chunk":
        return f"Reverting the chunker alone restores {where} {tag}."
    if factor == "retrieve":
        return (f"Fusion re-weighting demoted the target; reverting the retrieval "
                f"weights alone restores {where} {tag}.")
    if factor == "embed":
        return (f"Embedding model change is a black box; see per-tag metric deltas, "
                f"no mechanistic story is asserted. The embedding-revert arm restores "
                f"{where} {tag}.")
    if factor == "corpus":
        return (f"Newly ingested content now outranks the target (corpus drift); no "
                f"config change is responsible. Reverting the corpus alone restores "
                f"{where} {tag}.")
    return f"Reverting {vc.factor} alone restores {where} {tag}."


def _fallback_sentence(rep: AttributionReport) -> str:
    f = rep.funnel
    parts = [
        f"dense {_rank(f.dense.get('rank_before'))}→{_rank(f.dense.get('rank_after'))}",
        f"sparse {_rank(f.sparse.get('rank_before'))}→{_rank(f.sparse.get('rank_after'))}",
        f"fused {_rank(f.fused.get('rank_before'))}→{_rank(f.fused.get('rank_after'))}",
    ]
    return (f"Unverified: no counterfactual arm confirmed a cause. Funnel ranks "
            f"{', '.join(parts)}; every signal here is unverified.")


def _hypothesis_sentence(h: Hypothesis) -> str:
    evidence = h.evidence.strip()
    if evidence:
        return f"Unverified: {h.factor} implicated, {evidence}."
    return f"Unverified: {h.factor} implicated but not confirmed."


def _sorted_causes(causes: list[VerifiedCause]) -> list[VerifiedCause]:
    return sorted(causes, key=lambda vc: (vc.factor, vc.arm_id))


def render_narrative(rep: AttributionReport, case: GoldenCase,
                     chunk_texts: dict[str, str] | None = None) -> str:
    """Deterministic, evidence-only narrative for one attribution report (FR-8.3).

    Composed as: an observational chunk-fate mechanism sentence (when the tracer
    was not intact), then either one verified sentence per verified cause (each
    citing its arm id, FR-8.5) or a funnel-only fallback labelling everything
    unverified; an ANN-divergence note is appended when the funnel flagged one
    (FR-6.3), and each hypothesis is rendered as an ``Unverified:`` line with its
    evidence (FR-8.2). Embedding-swap causes are characterised, never given a
    mechanistic story (FR-8.4). ``chunk_texts`` is accepted for side-by-side
    callers but the narrative never depends on it, keeping output deterministic.
    """
    sentences: list[str] = []

    fate = rep.chunk_fate
    if fate is not None and fate.cls != "intact":
        sentences.append(_fate_sentence(fate))

    if rep.verified_causes:
        sentences.extend(_verified_sentence(vc) for vc in _sorted_causes(rep.verified_causes))
    else:
        sentences.append(_fallback_sentence(rep))

    if rep.funnel.ann_divergence:
        sentences.append(ANN_NOTE)

    sentences.extend(_hypothesis_sentence(h) for h in rep.hypotheses)

    return " ".join(s for s in sentences if s)


def _word_present(text: str, word: str) -> bool:
    pattern = r"(?<![\w-])" + re.escape(word) + r"(?![\w-])"
    return re.search(pattern, text, re.IGNORECASE) is not None


def narrative_faithfulness_audit(rep: AttributionReport, text: str) -> list[str]:
    """Return faithfulness violations: controlled tokens in ``text`` not backed by
    ``rep``'s structured evidence (§13). Returns ``[]`` for a faithful narrative.

    Four controlled vocabularies are checked: arm ids (any ``arm_*`` token must
    appear in a verified cause), chunk-fate classes (must equal the report's fate
    or appear in hypothesis evidence), stage names (only funnel stages are
    licensed; ``ann`` requires ``ann_divergence``), and factor names (must map to
    a verified/hypothesis factor, or ``chunk`` when a chunk-fate is present).
    """
    violations: list[str] = []

    allowed_arms = {vc.arm_id for vc in rep.verified_causes}
    for token in _ARM_RE.findall(text):
        if token not in allowed_arms:
            violations.append(f"unverified arm id in narrative: {token}")

    evidence_text = " ".join(h.evidence for h in rep.hypotheses)
    allowed_fates = {rep.chunk_fate.cls} if rep.chunk_fate is not None else set()
    allowed_fates |= {w for w in CHUNK_FATES if _word_present(evidence_text, w)}
    for w in CHUNK_FATES:
        if w not in allowed_fates and _word_present(text, w):
            violations.append(f"chunk-fate class not in evidence: {w}")

    allowed_stages = set(_ALWAYS_STAGES)
    if rep.funnel.ann_divergence:
        allowed_stages.add("ann")
    for w in _STAGE_UNIVERSE:
        if w not in allowed_stages and _word_present(text, w):
            violations.append(f"stage not in evidence: {w}")

    allowed_factors = {_canon(vc.factor) for vc in rep.verified_causes}
    allowed_factors |= {_canon(h.factor) for h in rep.hypotheses}
    if rep.chunk_fate is not None:
        allowed_factors.add("chunk")
    for word, canonical in _FACTOR_WORDS.items():
        if canonical not in allowed_factors and _word_present(text, word):
            violations.append(f"factor not in evidence: {word}")

    return violations
