"""Eval runner and native retrieval metrics (PRD FR-4).

Metrics are deterministic functions of a ranked doc list and the expected
sources (FR-4.1). ``evaluate`` runs every golden case through the
``RetrievalEngine``, replayed from stored artifacts or live against an
adapter (FR-4.2), and persists the result with full context (FR-4.4).
``created_at`` is left empty so identical inputs yield byte-identical eval
artifacts; the ``run_id`` is content-addressed from the eval's inputs.

nDCG uses binary relevance: DCG sums ``1/log2(rank+1)`` over relevant docs in
the top-k, IDCG over ``min(len(expected), k)`` ideal positions.
"""
from __future__ import annotations

import math

from . import hashing
from .adapters.base import VectorAdapter
from .models import (
    EvalResult,
    GoldenCase,
    GoldenDataset,
    QueryEval,
    QueryRun,
    SnapshotManifest,
)
from .retrieval import RetrievalEngine
from .store import ProjectStore

__all__ = ["recall_at_k", "hit_rate_at_k", "mrr", "ndcg_at_k", "evaluate", "MODES"]

MODES = ("replay", "live")


def recall_at_k(ranked_docs: list[str], expected: list[str], k: int) -> float:
    want = set(expected)
    if not want:
        return 0.0
    return len(want & set(ranked_docs[:k])) / len(want)


def hit_rate_at_k(ranked_docs: list[str], expected: list[str], k: int) -> float:
    return 1.0 if set(expected) & set(ranked_docs[:k]) else 0.0


def mrr(ranked_docs: list[str], expected: list[str]) -> float:
    want = set(expected)
    for rank, doc in enumerate(ranked_docs, start=1):
        if doc in want:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_docs: list[str], expected: list[str], k: int) -> float:
    want = set(expected)
    if not want or k <= 0:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc in enumerate(ranked_docs[:k], start=1)
        if doc in want
    )
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(want), k) + 1))
    return dcg / idcg


def _metric_keys(k_values: tuple[int, ...]) -> list[str]:
    keys: list[str] = []
    for k in k_values:
        keys.extend((f"recall@{k}", f"hit_rate@{k}", f"ndcg@{k}"))
    keys.append("mrr")
    return keys


def _case_eval(case: GoldenCase, run: QueryRun, doc_map: dict[str, str],
               k_values: tuple[int, ...]) -> QueryEval:
    ranked_docs: list[str] = []
    seen: set[str] = set()
    for cid, _ in run.final:
        doc = doc_map.get(cid)
        if doc is not None and doc not in seen:
            seen.add(doc)
            ranked_docs.append(doc)

    want = set(case.expected_sources)
    target_rank = next(
        (rank for rank, (cid, _) in enumerate(run.final, start=1)
         if doc_map.get(cid) in want),
        None,
    )

    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(ranked_docs, case.expected_sources, k)
        metrics[f"hit_rate@{k}"] = hit_rate_at_k(ranked_docs, case.expected_sources, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(ranked_docs, case.expected_sources, k)
    metrics["mrr"] = mrr(ranked_docs, case.expected_sources)

    return QueryEval(
        query_id=case.id,
        ranked_chunks=list(run.final),
        ranked_docs=ranked_docs,
        target_rank=target_rank,
        hit_at={k: metrics[f"hit_rate@{k}"] == 1.0 for k in k_values},
        metrics=metrics,
        run=run,
    )


def evaluate(store: ProjectStore, manifest: SnapshotManifest, dataset: GoldenDataset,
             adapter: VectorAdapter | None = None, k_values: tuple[int, ...] = (1, 5, 10),
             mode: str = "replay") -> EvalResult:
    if mode not in MODES:
        raise ValueError(f"unknown eval mode {mode!r}; expected one of {MODES}")
    if mode == "live" and adapter is None:
        raise ValueError("mode 'live' requires a vector adapter")
    k_values = tuple(int(k) for k in k_values)

    engine = RetrievalEngine(store, manifest, adapter=adapter if mode == "live" else None)
    doc_map = engine.chunk_doc_map()

    per_query: dict[str, QueryEval] = {}
    for case in dataset.cases:
        run = engine.run_query(case.id, case.question)
        per_query[case.id] = _case_eval(case, run, doc_map, k_values)

    n = len(per_query)
    aggregate = {
        key: (sum(qe.metrics[key] for qe in per_query.values()) / n) if n else 0.0
        for key in _metric_keys(k_values)
    }

    adapter_name = adapter.name if mode == "live" and adapter is not None else "none"
    run_id = "ev_" + hashing.h(
        manifest.snapshot_id, dataset.dataset_id, mode, adapter_name, str(k_values)
    )
    result = EvalResult(
        run_id=run_id,
        snapshot_id=manifest.snapshot_id,
        dataset_id=dataset.dataset_id,
        mode=mode,
        adapter=adapter_name,
        created_at="",
        k_values=k_values,
        per_query=per_query,
        aggregate=aggregate,
    )
    store.save_eval(result)
    return result
