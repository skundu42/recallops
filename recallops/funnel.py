"""Layer-1 funnel attribution (PRD FR-6).

For each regressed query, localize where the previously-winning chunk (the
*tracer*, FR-6 preamble) was lost as it flows through the new pipeline's
stages. Everything here is observational: no re-embedding and no counterfactual
runs (FR-6.4). Ranks come from the per-stage candidate lists already captured
on each ``QueryRun`` (``qdiff.before.run`` for A, ``qdiff.after.run`` for B),
1-based; a chunk absent from a stage list has rank ``None``.

The tracer's identity in snapshot B is resolved through the chunk alignment
(FR-5.3): a split tracer is followed to its descendant chunks, an intact tracer
(or one with no alignment entry, e.g. a retrieve-only change) stays itself.
``shadow_exact_rank_after`` re-scores the resolved target with exact cosine over
B's stored embeddings, independent of the serving DB (FR-6.2); comparing it to
a live run surfaces ANN/index divergence as its own failure class (FR-6.3).
"""
from __future__ import annotations

from .models import ChunkFate, FunnelReport, GoldenCase, QueryDiff, QueryEval, QueryRun
from .retrieval import RetrievalEngine

__all__ = [
    "tracer_chunk",
    "funnel_for_query",
    "failing_stage",
    "implicated_factors",
    "ANN_DIVERGENCE_POSITIONS",
    "STAGE_FACTORS",
]

ANN_DIVERGENCE_POSITIONS = 2

STAGE_FACTORS: dict[str, frozenset[str]] = {
    "index": frozenset({"embed", "index", "chunk", "parse", "corpus"}),
    "dense": frozenset({"embed", "index", "chunk", "parse", "corpus"}),
    "sparse": frozenset({"chunk", "parse", "corpus"}),
    "fused": frozenset({"retrieve", "chunk", "parse", "corpus", "embed"}),
    "rerank": frozenset({"rerank", "chunk", "corpus"}),
    "ann": frozenset({"index"}),
}

_LARGE = 10 ** 9


def tracer_chunk(before: QueryEval, expected_sources: list[str],
                 chunk_doc_paths_a: dict[str, tuple[str, ...]]) -> str | None:
    """A's previously-winning chunk: the top-ranked chunk in A's final list
    whose source document is one of ``expected_sources`` (FR-6 preamble).
    A chunk's document may exist at several byte-identical paths; any of them
    matching counts. Returns ``None`` when A ranked no expected-doc chunk."""
    want = set(expected_sources)
    for cid, _ in before.ranked_chunks:
        if any(p in want for p in chunk_doc_paths_a.get(cid, ())):
            return cid
    return None


def _stage(run: QueryRun | None, name: str) -> list[tuple[str, float]]:
    if run is None:
        return []
    return getattr(run.stages, name)


def _best_rank(pairs: list[tuple[str, float]], target_ids: list[str]) -> int | None:
    wanted = set(target_ids)
    for i, (cid, _) in enumerate(pairs, start=1):
        if cid in wanted:
            return i
    return None


def _retrieve_top_k(engine: RetrievalEngine) -> int:
    retrieve = engine.manifest.pipeline.stage("retrieve")
    if retrieve is None:
        return 10
    return int(retrieve.params.get("top_k", 10))


def _ann_divergence(live_run_b: QueryRun | None, after_ids: list[str],
                    shadow_rank: int | None) -> bool:
    if live_run_b is None or shadow_rank is None:
        return False
    live_dense = live_run_b.stages.dense
    live_rank = _best_rank(live_dense, after_ids)
    if live_rank is None:
        return shadow_rank <= len(live_dense)
    return abs(live_rank - shadow_rank) > ANN_DIVERGENCE_POSITIONS


def funnel_for_query(engine_a: RetrievalEngine, engine_b: RetrievalEngine,
                     qdiff: QueryDiff, case: GoldenCase,
                     alignment: dict[str, ChunkFate],
                     live_run_b: QueryRun | None = None) -> FunnelReport:
    """Stage-wise before/after funnel for one query (FR-6.1).

    Resolves the tracer's identity in B via ``alignment``: a split tracer is
    followed to all its descendant chunks (rank_after is the best rank among
    them in each stage); any other aligned tracer is followed to its descendant
    ``new_chunks[0]`` (an intact chunk may have a new content-addressed id after
    a boundary shift); a tracer with no alignment entry or no descendants stays
    itself. Dense/sparse/fused ranks read from the captured stage candidate
    lists; ``shadow_exact_rank_after`` from B's exact-cosine shadow scorer;
    ``ann_divergence`` compares an optional live run's dense rank against that
    shadow (FR-6.2, FR-6.3).
    """
    tracer = tracer_chunk(qdiff.before, case.expected_sources, engine_a.chunk_doc_paths())
    if tracer is None:
        return FunnelReport(
            target_chunk_before="",
            target_in_index_after=False,
            dense={"rank_before": None, "rank_after": None, "shadow_exact_rank_after": None},
            sparse={"rank_before": None, "rank_after": None},
            fused={"rank_before": None, "rank_after": None},
            rerank={"in_candidates_after": False},
            ann_divergence=False,
        )

    # Follow the tracer to its descendant even when the fate is "intact": an
    # intact chunk whose boundary shifted (within the 0.95 threshold) has a NEW
    # content-addressed id, so keeping the old id would falsely report the target
    # as absent from B (finding #6). Only a dropped fate (no descendants) or a
    # retrieve-only change (no alignment entry) falls back to the tracer id.
    fate = alignment.get(tracer)
    if fate is None or not fate.new_chunks:
        resolved = tracer
    else:
        resolved = fate.new_chunks[0]
    after_ids = list(fate.new_chunks) if (fate is not None and fate.cls == "split") else [resolved]

    target_in_index_after = resolved in engine_b.chunk_texts()

    before_run = qdiff.before.run
    after_run = qdiff.after.run

    # Shadow rank must use the SAME descendant set as dense.rank_after and the ANN
    # comparison, else a split whose best fragment differs from new_chunks[0]
    # fabricates a divergence against a purely exact index (finding #2).
    shadow_ranks = engine_b.exact_dense_ranks(case.question)
    shadow_after = _best_rank(shadow_ranks, after_ids)

    dense = {
        "rank_before": _best_rank(_stage(before_run, "dense"), [tracer]),
        "rank_after": _best_rank(_stage(after_run, "dense"), after_ids),
        "shadow_exact_rank_after": shadow_after,
    }
    sparse = {
        "rank_before": _best_rank(_stage(before_run, "sparse"), [tracer]),
        "rank_after": _best_rank(_stage(after_run, "sparse"), after_ids),
    }
    fused = {
        "rank_before": _best_rank(_stage(before_run, "fused"), [tracer]),
        "rank_after": _best_rank(_stage(after_run, "fused"), after_ids),
    }

    top_k = _retrieve_top_k(engine_b)
    candidate_window = _stage(after_run, "fused")[:top_k]
    rerank: dict = {"in_candidates_after": _best_rank(candidate_window, after_ids) is not None}
    reranked = after_run.stages.reranked if after_run is not None else None
    if reranked is not None:
        rerank["rank_after"] = _best_rank(reranked, after_ids)

    return FunnelReport(
        target_chunk_before=tracer,
        target_in_index_after=target_in_index_after,
        dense=dense,
        sparse=sparse,
        fused=fused,
        rerank=rerank,
        ann_divergence=_ann_divergence(live_run_b, after_ids, shadow_after),
    )


def _within(rank: int | None, top_k: int) -> bool:
    return rank is not None and rank <= top_k


def _rerank_crossed(f: FunnelReport, top_k: int) -> bool:
    if not _within(f.fused.get("rank_before"), top_k):
        return False
    if not f.rerank.get("in_candidates_after", False):
        return True
    rank_after = f.rerank.get("rank_after")
    return rank_after is not None and not _within(rank_after, top_k)


def _penalty(rank: int | None) -> int:
    return _LARGE if rank is None else rank


def _largest_degradation(f: FunnelReport) -> str:
    best_name = "dense"
    best_deg: int | None = None
    for name, d in (("dense", f.dense), ("sparse", f.sparse), ("fused", f.fused)):
        deg = _penalty(d.get("rank_after")) - _penalty(d.get("rank_before"))
        if best_deg is None or deg > best_deg:
            best_deg, best_name = deg, name
    return best_name


def failing_stage(f: FunnelReport, top_k: int) -> str:
    """First stage where the target left ``top_k`` (before within, after out),
    scanning index -> dense -> sparse -> fused -> rerank; then "ann" when the
    exact shadow rank is still within ``top_k`` but a live run diverged
    (FR-6.3); otherwise the stage with the largest rank degradation.

    "within top_k" means a non-null rank <= top_k; a missing (None) after-rank
    counts as out. Index crosses when the target is absent from B's chunk set.
    Ties in the degradation fallback break by stage order (dense < sparse <
    fused); rerank is excluded from the fallback since it has no rank_before.
    """
    if not f.target_in_index_after:
        return "index"
    for name, d in (("dense", f.dense), ("sparse", f.sparse), ("fused", f.fused)):
        if _within(d.get("rank_before"), top_k) and not _within(d.get("rank_after"), top_k):
            return name
    if _rerank_crossed(f, top_k):
        return "rerank"
    if f.ann_divergence and _within(f.dense.get("shadow_exact_rank_after"), top_k):
        return "ann"
    return _largest_degradation(f)


def implicated_factors(stage: str, config_diff: dict[str, dict]) -> list[str]:
    """Candidate factor names for a failing stage, intersected with the factors
    that actually changed in ``config_diff`` (stage ids plus synthetic
    "corpus"). Returned sorted for determinism."""
    return sorted(STAGE_FACTORS.get(stage, frozenset()) & set(config_diff))
