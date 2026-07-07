"""Snapshot diffing and chunk-fate alignment (PRD FR-5).

Chunk alignment maps every old chunk to its descendants in the new chunkset via
char-span overlap on the parsed text (FR-5.3), possible only while the parser
is unchanged, since spans index into the same parsed text. When the parser
changed, ``fuzzy_align_chunks`` degrades to token-set Jaccard matching and the
diff labels mechanism attribution as unavailable (FR-5.4). Alignment only ever
compares chunks of the same ``doc_id`` (content hash of raw bytes, stable
across parser changes); old chunks of documents missing from snapshot B are
"dropped".
"""
from __future__ import annotations

import math

from . import hashing
from .bm25 import tokenize
from .models import (
    ChunkFate,
    ChunkRecord,
    DiffResult,
    EvalResult,
    GoldenDataset,
    QueryDiff,
    QueryEval,
    SnapshotManifest,
)
from .retrieval import chunkset_key_from_uri
from .store import ProjectStore

__all__ = ["align_chunks", "fuzzy_align_chunks", "classify_query", "diff",
           "CHUNK_RELEVANT_FACTORS"]

CHUNK_RELEVANT_FACTORS = frozenset({"parse", "chunk", "corpus"})

INTACT_THRESHOLD = 0.95
SPLIT_MIN_OVERLAP = 0.15
SPLIT_COMBINED = 0.9
MERGED_MAX_COVERAGE = 0.7
DROPPED_COMBINED = 0.5
FUZZY_MATCH_THRESHOLD = 0.6

PRIMARY_METRIC = "recall@5"
DEFAULT_K = 10


def _by_doc(records: list[ChunkRecord]) -> dict[str, list[ChunkRecord]]:
    grouped: dict[str, list[ChunkRecord]] = {}
    for record in records:
        grouped.setdefault(record.doc_id, []).append(record)
    return grouped


def _union_length(intervals: list[tuple[int, int]]) -> int:
    total = 0
    cursor_start: int | None = None
    cursor_end = 0
    for start, end in sorted(intervals):
        if cursor_start is None or start > cursor_end:
            if cursor_start is not None:
                total += cursor_end - cursor_start
            cursor_start, cursor_end = start, end
        else:
            cursor_end = max(cursor_end, end)
    if cursor_start is not None:
        total += cursor_end - cursor_start
    return total


def _span_fate(old: ChunkRecord, candidates: list[ChunkRecord]) -> ChunkFate:
    length = old.span_end - old.span_start
    if length <= 0:
        return ChunkFate("dropped", 0.0, old.chunk_id, [])

    matches: list[tuple[float, float, ChunkRecord]] = []
    for new in candidates:
        inter = min(old.span_end, new.span_end) - max(old.span_start, new.span_start)
        if inter <= 0:
            continue
        new_length = new.span_end - new.span_start
        coverage = inter / new_length if new_length > 0 else 0.0
        matches.append((inter / length, coverage, new))
    matches.sort(key=lambda m: (-m[0], m[2].span_start, m[2].chunk_id))
    if not matches:
        return ChunkFate("dropped", 0.0, old.chunk_id, [])

    def union_fraction(subset: list[tuple[float, float, ChunkRecord]]) -> float:
        return _union_length([
            (max(old.span_start, m[2].span_start), min(old.span_end, m[2].span_end))
            for m in subset
        ]) / length

    combined = union_fraction(matches)
    qualifying = [m for m in matches if m[0] >= SPLIT_MIN_OVERLAP]
    full = [m for m in matches if m[0] >= INTACT_THRESHOLD]
    dominant_overlap, dominant_coverage, _ = matches[0]

    if len(full) == 1 and full[0][1] >= INTACT_THRESHOLD:
        cls = "intact"
    elif len(qualifying) >= 2 and union_fraction(qualifying) >= SPLIT_COMBINED:
        cls = "split"
    elif dominant_overlap >= INTACT_THRESHOLD and dominant_coverage < MERGED_MAX_COVERAGE:
        cls = "merged"
    elif combined < DROPPED_COMBINED:
        return ChunkFate("dropped", combined, old.chunk_id, [])
    else:
        cls = "boundary-shifted"
    return ChunkFate(cls, combined, old.chunk_id, [m[2].chunk_id for m in matches])


def align_chunks(old: list[ChunkRecord], new: list[ChunkRecord]) -> dict[str, ChunkFate]:
    """Span-overlap alignment: one ChunkFate per old chunk, keyed by chunk id.

    Overlap = span-intersection length / old span length; coverage = the same
    intersection over the new chunk's length. Classes, in priority order:

    - intact: exactly one new chunk with overlap >= 0.95 whose coverage by the
      old chunk is also >= 0.95
    - split: >= 2 new chunks each overlapping >= 0.15 whose combined (union)
      overlap is >= 0.9
    - merged: dominant new chunk covers the old >= 0.95 but the old covers
      < 0.7 of it
    - dropped: combined overlap < 0.5 (including no overlap at all)
    - boundary-shifted: otherwise (dominant overlap typically in [0.5, 0.95))

    ``alignment_score`` is the combined overlap fraction (union of span
    intersections over the old span length); ``new_chunks`` lists overlapping
    new chunks sorted by overlap descending (empty when dropped).
    """
    grouped = _by_doc(new)
    return {r.chunk_id: _span_fate(r, grouped.get(r.doc_id, [])) for r in old}


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _fuzzy_fate(old: ChunkRecord, candidates: list[ChunkRecord]) -> ChunkFate:
    old_tokens = frozenset(tokenize(old.text))
    best: ChunkRecord | None = None
    best_score = 0.0
    for new in sorted(candidates, key=lambda r: (r.span_start, r.chunk_id)):
        score = _jaccard(old_tokens, frozenset(tokenize(new.text)))
        if score > best_score:
            best, best_score = new, score
    if best is None or best_score < FUZZY_MATCH_THRESHOLD:
        return ChunkFate("dropped", best_score, old.chunk_id, [])
    cls = "intact" if best_score >= INTACT_THRESHOLD else "boundary-shifted"
    return ChunkFate(cls, best_score, old.chunk_id, [best.chunk_id])


def fuzzy_align_chunks(old: list[ChunkRecord], new: list[ChunkRecord]) -> dict[str, ChunkFate]:
    """Parser-changed alignment (FR-5.4): token-set Jaccard, spans ignored.

    The best same-doc match with Jaccard >= 0.6 becomes the single descendant;
    classes are limited to intact (>= 0.95), boundary-shifted ([0.6, 0.95)) and
    dropped (< 0.6). ``alignment_score`` is the best Jaccard found.
    """
    grouped = _by_doc(new)
    return {r.chunk_id: _fuzzy_fate(r, grouped.get(r.doc_id, [])) for r in old}


def _metric_k(primary_metric: str) -> int:
    _, _, suffix = primary_metric.partition("@")
    return int(suffix) if suffix.isdigit() else DEFAULT_K


def _score_gap(after: QueryEval, k: int) -> float:
    scores = [score for _, score in after.ranked_chunks]
    if not scores:
        return math.inf

    def at(position: int) -> float:
        return scores[min(max(position, 1), len(scores)) - 1]

    if after.target_rank is not None:
        return abs(at(after.target_rank) - at(k))
    # Target absent from the served list => it fell below EVERY served candidate,
    # a definite severe drop, never a near-tie. The gap between two unrelated
    # items at the k/k+1 cutoff has nothing to do with the target's fall, and
    # using it would exclude the worst regressions from the flip counts (#2).
    return math.inf


def classify_query(before: QueryEval, after: QueryEval, primary_metric: str = PRIMARY_METRIC,
                   epsilon: float = 0.0) -> tuple[str, str]:
    """Classify one query across snapshots -> (classification, stability).

    Classification compares ``primary_metric``: decreased -> "regressed",
    increased -> "improved"; when equal, "changed-top-k" if the top-k chunk-id
    sets differ (k parsed from the metric name, e.g. "recall@5" -> 5, default
    10) else "unchanged".

    Stability (FR-9.3) is pinned to the AFTER run's score gap around the k-th
    final candidate, positions clamped to the list length:

    - target present: gap = ``|score_at(target_rank) - score_at(k)|`` (the
      target's actual score, so a target that crashed far below k shows its true
      large gap and stays "stable" rather than being mislabeled a near-tie)
    - target absent from the served list: gap = +inf (a definite severe drop,
      never a near-tie; also +inf for an empty list)

    "unstable" iff gap < epsilon; with the default epsilon of 0.0 every query
    is "stable".
    """
    delta = after.metrics[primary_metric] - before.metrics[primary_metric]
    k = _metric_k(primary_metric)
    if delta < 0:
        classification = "regressed"
    elif delta > 0:
        classification = "improved"
    else:
        before_top = {cid for cid, _ in before.ranked_chunks[:k]}
        after_top = {cid for cid, _ in after.ranked_chunks[:k]}
        classification = "changed-top-k" if before_top != after_top else "unchanged"
    stability = "unstable" if _score_gap(after, k) < epsilon else "stable"
    return classification, stability


def _manifest_chunks(store: ProjectStore, manifest: SnapshotManifest) -> list[ChunkRecord]:
    return store.get_chunks(chunkset_key_from_uri(manifest.artifacts["chunks_uri"]))


def diff(store: ProjectStore, manifest_a: SnapshotManifest, manifest_b: SnapshotManifest,
         dataset: GoldenDataset, eval_a: EvalResult, eval_b: EvalResult,
         epsilon: float = 0.0, primary_metric: str = PRIMARY_METRIC) -> DiffResult:
    """Full snapshot diff (FR-5.1): config diff, metric deltas, per-query
    classification, and chunk alignment when a chunk-relevant factor changed.

    ``config_diff`` is the pipeline stage diff plus a synthetic "corpus" entry
    carrying both merkle roots when the corpus changed. Alignment is computed
    only when a factor in ``CHUNK_RELEVANT_FACTORS`` changed; with an unchanged
    parser it uses exact span overlap, otherwise the fuzzy path with
    ``alignment_available=False`` (FR-5.4). The result is persisted under
    ``("diff", diff_id)``.
    """
    config_diff = manifest_a.pipeline.diff_factors(manifest_b.pipeline)
    corpus_changed = manifest_a.corpus.merkle_root != manifest_b.corpus.merkle_root
    if corpus_changed:
        config_diff["corpus"] = {
            "before": {"merkle_root": manifest_a.corpus.merkle_root},
            "after": {"merkle_root": manifest_b.corpus.merkle_root},
        }
    parser_changed = "parse" in config_diff

    alignment: dict[str, ChunkFate] = {}
    alignment_available = True
    if CHUNK_RELEVANT_FACTORS & set(config_diff):
        old_chunks = _manifest_chunks(store, manifest_a)
        new_chunks = _manifest_chunks(store, manifest_b)
        if parser_changed:
            alignment = fuzzy_align_chunks(old_chunks, new_chunks)
            alignment_available = False
        else:
            alignment = align_chunks(old_chunks, new_chunks)

    metric_deltas = {
        key: eval_b.aggregate[key] - eval_a.aggregate[key]
        for key in eval_a.aggregate if key in eval_b.aggregate
    }

    queries: dict[str, QueryDiff] = {}
    for case in dataset.cases:
        before = eval_a.per_query.get(case.id)
        after = eval_b.per_query.get(case.id)
        if before is None or after is None:
            continue
        classification, stability = classify_query(
            before, after, primary_metric=primary_metric, epsilon=epsilon)
        queries[case.id] = QueryDiff(
            query_id=case.id,
            classification=classification,
            stability=stability,
            metric_delta={key: after.metrics[key] - before.metrics[key]
                          for key in before.metrics if key in after.metrics},
            before=before,
            after=after,
        )

    result = DiffResult(
        diff_id="diff_" + hashing.h(manifest_a.snapshot_id, manifest_b.snapshot_id,
                                    dataset.dataset_id),
        snapshot_a=manifest_a.snapshot_id,
        snapshot_b=manifest_b.snapshot_id,
        dataset_id=dataset.dataset_id,
        config_diff=config_diff,
        corpus_changed=corpus_changed,
        metric_deltas=metric_deltas,
        queries=queries,
        alignment=alignment,
        parser_changed=parser_changed,
        alignment_available=alignment_available,
    )
    store.save_json("diff", result.diff_id, result.to_dict())
    return result
