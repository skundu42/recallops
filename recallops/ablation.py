"""Counterfactual ablation engine (PRD FR-7).

Given a diff A->B, the changed factors (pipeline stages plus a synthetic
"corpus" factor) define a lattice of *arms*: manifests where each factor is
independently held at its A or B state. Reverting a single factor toward A,
holding the rest at B, is the counterfactual that the confirmation rule
(FR-8.1) turns into a *verified* cause.

Arms are materialized entirely from the provenance store: the chosen corpus
merkle root resolves to its documents (raw bytes included), which are re-parsed,
re-chunked and embedded under the arm's composed pipeline. Because every
identifier is content-addressed, an arm that recomposes an already-materialized
state resolves to the existing snapshot with zero new embedding calls (FR-7.1),
and arms sharing a chunkset/embedding share the cache. Arm scoring always runs
in replay mode (exact KNN over stored embeddings), so ANN noise never enters
causal analysis (FR-7.2).
"""
from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from math import factorial
from pathlib import Path

from .evalrunner import evaluate
from .ingest import chunkset_key, embedding_keys, embeddings_uri, parse_ph
from .models import (
    Arm,
    ArmResult,
    CorpusInfo,
    DiffResult,
    Factor,
    GoldenDataset,
    PipelineDAG,
    SnapshotManifest,
    StageSpec,
)
from .pipeline import chunkers, parsers
from .pipeline.providers import EmbeddingProvider, estimate_embed_cost, get_provider
from .store import ProjectStore

__all__ = [
    "enumerate_factors",
    "build_arms",
    "ArmPlan",
    "plan_arms",
    "materialize_arm",
    "run_arms",
    "shapley",
    "STAGE_ORDER",
    "DEFAULT_REGISTERED_PAIRS",
]

STAGE_ORDER = ("parse", "chunk", "embed", "index", "retrieve", "rerank")
DEFAULT_REGISTERED_PAIRS = (("embed", "chunk"), ("retrieve", "chunk"))
CORPUS = "corpus"


def enumerate_factors(diffres: DiffResult) -> list[Factor]:
    """Changed pipeline stages as ``kind="stage"`` factors plus a ``kind="corpus"``
    factor when the corpus content changed (FR-7.6). The synthetic "corpus" key
    that ``diff`` writes into ``config_diff`` is not itself a stage factor."""
    factors = [
        Factor(name=sid, kind="stage")
        for sid in sorted(k for k in diffres.config_diff if k != CORPUS)
    ]
    if diffres.corpus_changed:
        factors.append(Factor(name=CORPUS, kind="corpus"))
    return factors


def _factor_names(factors: list[Factor]) -> list[str]:
    return [f.name for f in factors]


def build_arms(factors: list[Factor], mode: str = "auto",
               pruned_to: list[str] | None = None,
               registered_pairs: tuple[tuple[str, str], ...] = DEFAULT_REGISTERED_PAIRS) -> list[Arm]:
    """Construct the arm set for a factor list (FR-7.3).

    ``k = len(factors)``. With ``k <= 3`` or ``mode == "full"`` the full lattice
    over the flippable factors is enumerated (every flippable factor in {A, B}).
    With ``k > 3`` and ``mode == "auto"`` the set is one-factor-at-a-time (each
    flippable factor at A, the rest at B) plus both-at-A arms for the registered
    interaction pairs present, plus the all-B and all-A endpoints.

    ``pruned_to`` (typically the funnel's implicated factors, FR-7.4) restricts
    which factors may be flipped to A; the rest are pinned at B. Every arm's
    assignment covers *all* factors so materialization is fully specified.
    """
    names = _factor_names(factors)
    if pruned_to is None:
        flippable = list(names)
    else:
        allowed = set(pruned_to)
        flippable = [n for n in names if n in allowed]
    k = len(names)

    arms: dict[str, Arm] = {}

    def add(a_factors: set[str]) -> None:
        assignment = {n: ("A" if n in a_factors else "B") for n in names}
        arm = Arm.build(assignment)
        arms.setdefault(arm.arm_id, arm)

    if mode == "full" or k <= 3:
        for r in range(len(flippable) + 1):
            for combo in itertools.combinations(flippable, r):
                add(set(combo))
    else:
        add(set())
        for n in flippable:
            add({n})
        for x, y in registered_pairs:
            if x in flippable and y in flippable:
                add({x, y})
        add(set(flippable))

    return list(arms.values())


def _stage_maps(manifest_a: SnapshotManifest, manifest_b: SnapshotManifest):
    return (
        {s.id: s for s in manifest_a.pipeline.stages},
        {s.id: s for s in manifest_b.pipeline.stages},
    )


def _compose_pipeline(arm: Arm, manifest_a: SnapshotManifest,
                      manifest_b: SnapshotManifest) -> PipelineDAG:
    """Compose the arm's pipeline: every factor stage taken from A or B per the
    assignment, non-factor stages taken from B (the current state). A stage
    absent on the chosen side is dropped, so added/removed stages revert
    correctly. Stages are ordered by the canonical RecallOps stage order."""
    a_stages, b_stages = _stage_maps(manifest_a, manifest_b)
    all_ids = set(a_stages) | set(b_stages)
    ordered = [sid for sid in STAGE_ORDER if sid in all_ids]
    ordered += sorted(all_ids - set(STAGE_ORDER))

    stages = []
    for sid in ordered:
        spec = a_stages.get(sid) if arm.assignment.get(sid) == "A" else b_stages.get(sid)
        if spec is not None:
            stages.append(spec)
    dag = PipelineDAG(tuple(stages))
    dag.validate()
    return dag


def _chosen_merkle(arm: Arm, manifest_a: SnapshotManifest, manifest_b: SnapshotManifest) -> str:
    if arm.assignment.get(CORPUS) == "A":
        return manifest_a.corpus.merkle_root
    return manifest_b.corpus.merkle_root


def _arm_records(store: ProjectStore, pipeline: PipelineDAG, merkle: str):
    parse_stage = pipeline.stage("parse")
    chunk_stage = pipeline.stage("chunk")
    key = chunkset_key(merkle, parse_stage, chunk_stage)
    if store.has_chunkset(key):
        return store.get_chunks(key)
    records = []
    chunked: set[str] = set()
    for doc in store.docs_for_merkle(merkle):
        parsed = parsers.parse(doc["source_path"], doc["raw"], tool=parse_stage.tool)
        store.put_doc(doc["source_path"], doc["raw"], parsed.text, parse_ph(parse_stage))
        # Byte-identical files share one doc_id; chunk each document once so
        # arm chunksets match managed ingest's content under the same key.
        if doc["doc_id"] in chunked:
            continue
        chunked.add(doc["doc_id"])
        records.extend(chunkers.chunk_doc(
            doc["doc_id"], parsed.text, chunk_stage.tool, chunk_stage.params,
            parse_stage.id, chunk_stage.id,
        ))
    return records


def _materialize(store: ProjectStore, pipeline: PipelineDAG, merkle: str,
                 provider: EmbeddingProvider | None = None,
                 parent: str | None = None) -> tuple[SnapshotManifest, int]:
    embed_stage = pipeline.stage("embed")
    docs = store.docs_for_merkle(merkle)
    records = _arm_records(store, pipeline, merkle)
    chunks_uri = store.put_chunks(chunkset_key(merkle, pipeline.stage("parse"),
                                               pipeline.stage("chunk")), records)

    if provider is None:
        provider = get_provider(dict(embed_stage.params))
    keys = embedding_keys(records, provider)
    missing = store.missing_embedding_keys(keys)
    embed_calls = len(missing)
    if missing:
        text_by_key: dict[str, str] = {}
        for record, k in zip(records, keys):
            text_by_key.setdefault(k, record.text)
        vectors = provider.embed([text_by_key[k] for k in missing])
        store.put_embeddings(provider.model_key, {k: vectors[i] for i, k in enumerate(missing)})

    artifacts = {
        "chunks_uri": chunks_uri,
        "embeddings_uri": embeddings_uri(provider.model_key),
    }
    corpus = CorpusInfo(doc_count=len(docs), chunk_count=len(records), merkle_root=merkle)
    manifest = SnapshotManifest.build(pipeline, corpus, artifacts, parent=parent)
    committed = store.commit_snapshot(manifest)
    return committed, embed_calls


def _materialize_arm(store: ProjectStore, arm: Arm, manifest_a: SnapshotManifest,
                     manifest_b: SnapshotManifest, source_dir: Path) -> tuple[SnapshotManifest, int]:
    pipeline = _compose_pipeline(arm, manifest_a, manifest_b)
    merkle = _chosen_merkle(arm, manifest_a, manifest_b)
    return _materialize(store, pipeline, merkle)


def materialize_arm(store: ProjectStore, arm: Arm, manifest_a: SnapshotManifest,
                    manifest_b: SnapshotManifest, source_dir: Path) -> SnapshotManifest:
    """Materialize one arm's snapshot from the store (FR-7.1, FR-7.2).

    The composed pipeline takes each factor stage from A or B per the arm and
    the corpus state (A or B, default B) from the corpus factor. Documents are
    re-parsed from stored raw bytes, re-chunked and embedded; content addressing
    makes an already-materialized arm resolve to its existing snapshot with no
    new work. ``source_dir`` is accepted for signature parity with ingest but is
    unused, the corpus is reconstructed from the store."""
    manifest, _ = _materialize_arm(store, arm, manifest_a, manifest_b, source_dir)
    return manifest


@dataclass
class ArmPlan:
    arms: list[Arm]
    new_embed_texts: int
    est_usd: float
    est_wall_s: float


def plan_arms(store: ProjectStore, manifest_a: SnapshotManifest, manifest_b: SnapshotManifest,
              source_dir: Path, arms: list[Arm],
              provider_factory: Callable[[StageSpec], EmbeddingProvider]) -> ArmPlan:
    """Dry-run cost gate for an arm set (FR-7.5).

    Counts the embedding keys that would still be missing across all arms,
    deduplicated by key so arms sharing the cache are only charged once, and
    prices them with ``estimate_embed_cost``. ``provider_factory`` maps an arm's
    composed embed stage to its provider; a local provider prices at $0.00.
    No embeddings are computed and no snapshots are committed.
    """
    seen: set[str] = set()
    per_provider: dict[str, tuple[EmbeddingProvider, list[str]]] = {}
    for arm in arms:
        pipeline = _compose_pipeline(arm, manifest_a, manifest_b)
        merkle = _chosen_merkle(arm, manifest_a, manifest_b)
        provider = provider_factory(pipeline.stage("embed"))
        records = _arm_records(store, pipeline, merkle)
        keys = embedding_keys(records, provider)
        text_by_key: dict[str, str] = {}
        for record, k in zip(records, keys):
            text_by_key.setdefault(k, record.text)
        for k in store.missing_embedding_keys(keys):
            if k in seen:
                continue
            seen.add(k)
            per_provider.setdefault(provider.model_key, (provider, []))[1].append(text_by_key[k])

    est_usd = 0.0
    est_wall_s = 0.0
    for provider, texts in per_provider.values():
        est = estimate_embed_cost(provider, texts)
        est_usd += est["usd"]
        est_wall_s += est["wall_s"]
    return ArmPlan(arms=list(arms), new_embed_texts=len(seen), est_usd=est_usd, est_wall_s=est_wall_s)


def run_arms(store: ProjectStore, arms: list[Arm], manifest_a: SnapshotManifest,
             manifest_b: SnapshotManifest, source_dir: Path, dataset: GoldenDataset,
             checkpoint_key: str, k_values: tuple[int, ...] = (1, 5, 10)) -> dict[str, ArmResult]:
    """Materialize and replay-evaluate every arm, resumably (FR-7.5).

    After each arm the checkpoint ``("job", checkpoint_key)`` records
    ``{arm_id: run_id}``; on re-entry a checkpointed arm is loaded from its saved
    eval with no materialization and no embedding calls, so an interrupted job
    resumes without repeating completed work.
    """
    checkpoint: dict[str, str] = dict(store.get_json("job", checkpoint_key) or {})
    results: dict[str, ArmResult] = {}
    for arm in arms:
        done = checkpoint.get(arm.arm_id)
        if done is not None:
            results[arm.arm_id] = ArmResult(arm.arm_id, store.get_eval(done), 0)
            continue
        manifest, embed_calls = _materialize_arm(store, arm, manifest_a, manifest_b, source_dir)
        ev = evaluate(store, manifest, dataset, adapter=None, k_values=k_values, mode="replay")
        results[arm.arm_id] = ArmResult(arm.arm_id, ev, embed_calls)
        checkpoint[arm.arm_id] = ev.run_id
        store.save_json("job", checkpoint_key, checkpoint)
    return results


def shapley(metric_by_assignment: dict[frozenset[str], float],
            factors: list[str]) -> dict[str, float]:
    """Exact Shapley attribution over the boolean factor lattice.

    Each key is the set of factors held at B; ``value(S)`` is that arm's metric.
    ``factor -> phi`` is the standard weighted marginal of moving that factor to
    B, so the values sum to ``value(all-B) - value(all-A)``.
    """
    names = list(factors)
    n = len(names)
    if n == 0:
        return {}
    result: dict[str, float] = {}
    for i in names:
        others = [f for f in names if f != i]
        phi = 0.0
        for r in range(len(others) + 1):
            weight = factorial(r) * factorial(n - r - 1) / factorial(n)
            for combo in itertools.combinations(others, r):
                s = frozenset(combo)
                phi += weight * (metric_by_assignment[s | {i}] - metric_by_assignment[s])
        result[i] = phi
    return result
