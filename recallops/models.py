"""Core data model. PRD §10 schemas are normative, field names match exactly.

Every dataclass round-trips through ``to_dict``/``from_dict`` (plain-JSON types
only) so artifacts can be persisted and re-rendered without re-running anything
(FR-13).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import hashing


def _pairs(items) -> list[tuple[str, float]]:
    return [(str(c), float(s)) for c, s in items]


@dataclass(frozen=True)
class StageSpec:
    id: str
    tool: str
    version: str = ""
    params: dict = field(default_factory=dict)
    inputs: tuple[str, ...] = ()

    @property
    def params_hash(self) -> str:
        return hashing.params_hash(self.params)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "tool": self.tool,
            "version": self.version,
            "params": self.params,
            "params_hash": self.params_hash,
            "inputs": list(self.inputs),
        }

    @classmethod
    def from_dict(cls, d: dict) -> StageSpec:
        return cls(
            id=d["id"],
            tool=d["tool"],
            version=d.get("version", ""),
            params=d.get("params", {}),
            inputs=tuple(d.get("inputs", ())),
        )


@dataclass(frozen=True)
class PipelineDAG:
    stages: tuple[StageSpec, ...]

    def stage(self, stage_id: str) -> StageSpec | None:
        for s in self.stages:
            if s.id == stage_id:
                return s
        return None

    def validate(self) -> None:
        ids = [s.id for s in self.stages]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate stage ids")
        known = set(ids)
        for s in self.stages:
            for inp in s.inputs:
                if inp not in known:
                    raise ValueError(f"stage {s.id!r} references unknown input {inp!r}")
        seen: set[str] = set()
        state: dict[str, int] = {}
        by_id = {s.id: s for s in self.stages}

        def visit(sid: str) -> None:
            if state.get(sid) == 1:
                raise ValueError(f"cycle involving stage {sid!r}")
            if sid in seen:
                return
            state[sid] = 1
            for inp in by_id[sid].inputs:
                visit(inp)
            state[sid] = 2
            seen.add(sid)

        for sid in ids:
            visit(sid)

    def replace(self, stage_id: str, spec: StageSpec) -> PipelineDAG:
        if self.stage(stage_id) is None:
            raise KeyError(stage_id)
        return PipelineDAG(tuple(spec if s.id == stage_id else s for s in self.stages))

    def diff_factors(self, other: PipelineDAG) -> dict[str, dict]:
        """Changed/added/removed stages: {stage_id: {"before": ..., "after": ...}}."""
        out: dict[str, dict] = {}
        mine = {s.id: s for s in self.stages}
        theirs = {s.id: s for s in other.stages}
        for sid in sorted(set(mine) | set(theirs)):
            a, b = mine.get(sid), theirs.get(sid)
            if a is not None and b is not None and a.to_dict() == b.to_dict():
                continue
            out[sid] = {
                "before": a.to_dict() if a else None,
                "after": b.to_dict() if b else None,
            }
        return out

    def to_dict(self) -> dict:
        return {"dag": [s.to_dict() for s in self.stages]}

    @classmethod
    def from_dict(cls, d: dict) -> PipelineDAG:
        return cls(tuple(StageSpec.from_dict(s) for s in d["dag"]))


@dataclass(frozen=True)
class CorpusInfo:
    doc_count: int
    chunk_count: int
    merkle_root: str

    def to_dict(self) -> dict:
        return {"doc_count": self.doc_count, "chunk_count": self.chunk_count, "merkle_root": self.merkle_root}

    @classmethod
    def from_dict(cls, d: dict) -> CorpusInfo:
        return cls(d["doc_count"], d["chunk_count"], d["merkle_root"])


@dataclass
class SnapshotManifest:
    snapshot_id: str
    parent_snapshot: str | None
    created_at: str
    pipeline: PipelineDAG
    corpus: CorpusInfo
    artifacts: dict[str, str]

    def core(self) -> dict:
        return {
            "pipeline": self.pipeline.to_dict(),
            "corpus": self.corpus.to_dict(),
            "artifacts": dict(sorted(self.artifacts.items())),
        }

    @classmethod
    def build(cls, pipeline: PipelineDAG, corpus: CorpusInfo, artifacts: dict[str, str],
              parent: str | None = None, created_at: str = "") -> SnapshotManifest:
        core = {
            "pipeline": pipeline.to_dict(),
            "corpus": corpus.to_dict(),
            "artifacts": dict(sorted(artifacts.items())),
        }
        return cls(
            snapshot_id=hashing.snapshot_hash(core),
            parent_snapshot=parent,
            created_at=created_at,
            pipeline=pipeline,
            corpus=corpus,
            artifacts=dict(sorted(artifacts.items())),
        )

    def to_dict(self) -> dict:
        return {
            "snapshot_id": self.snapshot_id,
            "parent_snapshot": self.parent_snapshot,
            "created_at": self.created_at,
            "pipeline": self.pipeline.to_dict(),
            "corpus": self.corpus.to_dict(),
            "artifacts": self.artifacts,
        }

    def to_json(self) -> str:
        return hashing.canonical_json(self.to_dict()).decode("utf-8")

    @classmethod
    def from_dict(cls, d: dict) -> SnapshotManifest:
        return cls(
            snapshot_id=d["snapshot_id"],
            parent_snapshot=d.get("parent_snapshot"),
            created_at=d.get("created_at", ""),
            pipeline=PipelineDAG.from_dict(d["pipeline"]),
            corpus=CorpusInfo.from_dict(d["corpus"]),
            artifacts=d.get("artifacts", {}),
        )


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    span_start: int
    span_end: int
    ordinal: int
    text: str
    text_hash: str
    parse_stage_id: str
    chunk_stage_id: str

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id, "doc_id": self.doc_id,
            "span_start": self.span_start, "span_end": self.span_end,
            "ordinal": self.ordinal, "text": self.text, "text_hash": self.text_hash,
            "parse_stage_id": self.parse_stage_id, "chunk_stage_id": self.chunk_stage_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ChunkRecord:
        return cls(**d)


@dataclass
class GoldenCase:
    id: str
    question: str
    expected_sources: list[str]
    expected_spans: None = None
    tags: list[str] = field(default_factory=list)
    origin: str = "manual"
    source_trace: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id, "question": self.question,
            "expected_sources": self.expected_sources, "expected_spans": self.expected_spans,
            "tags": self.tags, "origin": self.origin, "source_trace": self.source_trace,
        }

    @classmethod
    def from_dict(cls, d: dict) -> GoldenCase:
        return cls(
            id=d["id"], question=d["question"], expected_sources=d["expected_sources"],
            expected_spans=d.get("expected_spans"), tags=d.get("tags", []),
            origin=d.get("origin", "manual"), source_trace=d.get("source_trace"),
        )


@dataclass
class GoldenDataset:
    dataset_id: str
    cases: list[GoldenCase]

    @property
    def name(self) -> str:
        base, _, v = self.dataset_id.rpartition("-v")
        return base if base and v.isdigit() else self.dataset_id

    @property
    def version(self) -> int:
        _, _, v = self.dataset_id.rpartition("-v")
        return int(v) if v.isdigit() else 1

    def bump(self, new_cases: list[GoldenCase]) -> GoldenDataset:
        return GoldenDataset(f"{self.name}-v{self.version + 1}", list(self.cases) + list(new_cases))

    def case(self, case_id: str) -> GoldenCase | None:
        for c in self.cases:
            if c.id == case_id:
                return c
        return None

    def to_dict(self) -> dict:
        return {"dataset_id": self.dataset_id, "cases": [c.to_dict() for c in self.cases]}

    @classmethod
    def from_dict(cls, d: dict) -> GoldenDataset:
        return cls(d["dataset_id"], [GoldenCase.from_dict(c) for c in d["cases"]])


@dataclass
class StageCandidates:
    dense: list[tuple[str, float]]
    sparse: list[tuple[str, float]]
    fused: list[tuple[str, float]]
    reranked: list[tuple[str, float]] | None = None

    def to_dict(self) -> dict:
        return {
            "dense": [[c, s] for c, s in self.dense],
            "sparse": [[c, s] for c, s in self.sparse],
            "fused": [[c, s] for c, s in self.fused],
            "reranked": None if self.reranked is None else [[c, s] for c, s in self.reranked],
        }

    @classmethod
    def from_dict(cls, d: dict) -> StageCandidates:
        return cls(
            dense=_pairs(d["dense"]), sparse=_pairs(d["sparse"]), fused=_pairs(d["fused"]),
            reranked=None if d.get("reranked") is None else _pairs(d["reranked"]),
        )


@dataclass
class QueryRun:
    query_id: str
    question: str
    stages: StageCandidates
    final: list[tuple[str, float]]

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id, "question": self.question,
            "stages": self.stages.to_dict(), "final": [[c, s] for c, s in self.final],
        }

    @classmethod
    def from_dict(cls, d: dict) -> QueryRun:
        return cls(d["query_id"], d["question"], StageCandidates.from_dict(d["stages"]), _pairs(d["final"]))


@dataclass
class QueryEval:
    query_id: str
    ranked_chunks: list[tuple[str, float]]
    ranked_docs: list[str]
    target_rank: int | None
    hit_at: dict[int, bool]
    metrics: dict[str, float]
    run: QueryRun | None = None

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "ranked_chunks": [[c, s] for c, s in self.ranked_chunks],
            "ranked_docs": self.ranked_docs,
            "target_rank": self.target_rank,
            "hit_at": {str(k): v for k, v in self.hit_at.items()},
            "metrics": self.metrics,
            "run": self.run.to_dict() if self.run else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> QueryEval:
        return cls(
            query_id=d["query_id"], ranked_chunks=_pairs(d["ranked_chunks"]),
            ranked_docs=d["ranked_docs"], target_rank=d.get("target_rank"),
            hit_at={int(k): v for k, v in d.get("hit_at", {}).items()},
            metrics=d.get("metrics", {}),
            run=QueryRun.from_dict(d["run"]) if d.get("run") else None,
        )


@dataclass
class EvalResult:
    run_id: str
    snapshot_id: str
    dataset_id: str
    mode: str
    adapter: str
    created_at: str
    k_values: tuple[int, ...]
    per_query: dict[str, QueryEval]
    aggregate: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id, "snapshot_id": self.snapshot_id, "dataset_id": self.dataset_id,
            "mode": self.mode, "adapter": self.adapter, "created_at": self.created_at,
            "k_values": list(self.k_values),
            "per_query": {q: e.to_dict() for q, e in self.per_query.items()},
            "aggregate": self.aggregate,
        }

    @classmethod
    def from_dict(cls, d: dict) -> EvalResult:
        return cls(
            run_id=d["run_id"], snapshot_id=d["snapshot_id"], dataset_id=d["dataset_id"],
            mode=d["mode"], adapter=d.get("adapter", ""), created_at=d.get("created_at", ""),
            k_values=tuple(d["k_values"]),
            per_query={q: QueryEval.from_dict(e) for q, e in d["per_query"].items()},
            aggregate=d["aggregate"],
        )


CHUNK_FATES = ("intact", "split", "merged", "boundary-shifted", "dropped")


@dataclass
class ChunkFate:
    cls: str
    alignment_score: float
    old_chunk: str
    new_chunks: list[str]

    def to_dict(self) -> dict:
        return {
            "class": self.cls, "alignment_score": self.alignment_score,
            "old_chunk": self.old_chunk, "new_chunks": self.new_chunks,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ChunkFate:
        return cls(d["class"], d["alignment_score"], d["old_chunk"], d["new_chunks"])


QUERY_CLASSES = ("improved", "regressed", "changed-top-k", "unchanged")


@dataclass
class QueryDiff:
    query_id: str
    classification: str
    stability: str
    metric_delta: dict[str, float]
    before: QueryEval
    after: QueryEval

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id, "classification": self.classification,
            "stability": self.stability, "metric_delta": self.metric_delta,
            "before": self.before.to_dict(), "after": self.after.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> QueryDiff:
        return cls(
            d["query_id"], d["classification"], d["stability"], d["metric_delta"],
            QueryEval.from_dict(d["before"]), QueryEval.from_dict(d["after"]),
        )


@dataclass
class DiffResult:
    diff_id: str
    snapshot_a: str
    snapshot_b: str
    dataset_id: str
    config_diff: dict[str, dict]
    corpus_changed: bool
    metric_deltas: dict[str, float]
    queries: dict[str, QueryDiff]
    alignment: dict[str, ChunkFate]
    parser_changed: bool
    alignment_available: bool

    def by_class(self, classification: str, stable_only: bool = False) -> list[QueryDiff]:
        return [
            q for q in self.queries.values()
            if q.classification == classification and (not stable_only or q.stability == "stable")
        ]

    def to_dict(self) -> dict:
        return {
            "diff_id": self.diff_id, "snapshot_a": self.snapshot_a, "snapshot_b": self.snapshot_b,
            "dataset_id": self.dataset_id, "config_diff": self.config_diff,
            "corpus_changed": self.corpus_changed, "metric_deltas": self.metric_deltas,
            "queries": {q: d.to_dict() for q, d in self.queries.items()},
            "alignment": {c: f.to_dict() for c, f in self.alignment.items()},
            "parser_changed": self.parser_changed, "alignment_available": self.alignment_available,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DiffResult:
        return cls(
            diff_id=d["diff_id"], snapshot_a=d["snapshot_a"], snapshot_b=d["snapshot_b"],
            dataset_id=d["dataset_id"], config_diff=d["config_diff"],
            corpus_changed=d["corpus_changed"], metric_deltas=d["metric_deltas"],
            queries={q: QueryDiff.from_dict(x) for q, x in d["queries"].items()},
            alignment={c: ChunkFate.from_dict(x) for c, x in d["alignment"].items()},
            parser_changed=d["parser_changed"], alignment_available=d["alignment_available"],
        )


@dataclass
class FunnelReport:
    target_chunk_before: str
    target_in_index_after: bool
    dense: dict
    sparse: dict
    fused: dict
    rerank: dict
    ann_divergence: bool

    def to_dict(self) -> dict:
        return {
            "target_chunk_before": self.target_chunk_before,
            "target_in_index_after": self.target_in_index_after,
            "dense": self.dense, "sparse": self.sparse, "fused": self.fused,
            "rerank": self.rerank, "ann_divergence": self.ann_divergence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> FunnelReport:
        return cls(
            d["target_chunk_before"], d["target_in_index_after"],
            d["dense"], d["sparse"], d["fused"], d["rerank"], d["ann_divergence"],
        )


@dataclass(frozen=True)
class Factor:
    name: str
    kind: str  # "stage" | "corpus"

    def to_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind}

    @classmethod
    def from_dict(cls, d: dict) -> Factor:
        return cls(d["name"], d["kind"])


@dataclass
class Arm:
    arm_id: str
    assignment: dict[str, str]  # factor name -> "A" | "B"

    @classmethod
    def build(cls, assignment: dict[str, str]) -> Arm:
        aid = "arm_" + hashing.h(hashing.canonical_json(dict(sorted(assignment.items()))))
        return cls(aid, dict(sorted(assignment.items())))

    @property
    def at_a(self) -> frozenset[str]:
        return frozenset(f for f, side in self.assignment.items() if side == "A")

    @property
    def at_b(self) -> frozenset[str]:
        return frozenset(f for f, side in self.assignment.items() if side == "B")

    def to_dict(self) -> dict:
        return {"arm_id": self.arm_id, "assignment": self.assignment}

    @classmethod
    def from_dict(cls, d: dict) -> Arm:
        return cls(d["arm_id"], d["assignment"])


@dataclass
class ArmResult:
    arm_id: str
    eval: EvalResult
    embed_calls: int

    def to_dict(self) -> dict:
        return {"arm_id": self.arm_id, "eval": self.eval.to_dict(), "embed_calls": self.embed_calls}

    @classmethod
    def from_dict(cls, d: dict) -> ArmResult:
        return cls(d["arm_id"], EvalResult.from_dict(d["eval"]), d["embed_calls"])


@dataclass
class VerifiedCause:
    factor: str
    arm_id: str
    recovered_rank: int | None
    status: str = "verified"

    def to_dict(self) -> dict:
        return {"factor": self.factor, "arm_id": self.arm_id,
                "recovered_rank": self.recovered_rank, "status": self.status}

    @classmethod
    def from_dict(cls, d: dict) -> VerifiedCause:
        return cls(d["factor"], d["arm_id"], d.get("recovered_rank"), d.get("status", "verified"))


@dataclass
class Hypothesis:
    factor: str
    status: str = "unverified"
    evidence: str = ""

    def to_dict(self) -> dict:
        return {"factor": self.factor, "status": self.status, "evidence": self.evidence}

    @classmethod
    def from_dict(cls, d: dict) -> Hypothesis:
        return cls(d["factor"], d.get("status", "unverified"), d.get("evidence", ""))


@dataclass
class AttributionReport:
    query_id: str
    classification: str
    stability: str
    funnel: FunnelReport
    chunk_fate: ChunkFate | None
    verified_causes: list[VerifiedCause]
    hypotheses: list[Hypothesis]
    narrative: str

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "classification": self.classification,
            "stability": self.stability,
            "funnel": self.funnel.to_dict(),
            "chunk_fate": self.chunk_fate.to_dict() if self.chunk_fate else None,
            "verified_causes": [v.to_dict() for v in self.verified_causes],
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "narrative": self.narrative,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AttributionReport:
        return cls(
            query_id=d["query_id"], classification=d["classification"], stability=d["stability"],
            funnel=FunnelReport.from_dict(d["funnel"]),
            chunk_fate=ChunkFate.from_dict(d["chunk_fate"]) if d.get("chunk_fate") else None,
            verified_causes=[VerifiedCause.from_dict(v) for v in d.get("verified_causes", [])],
            hypotheses=[Hypothesis.from_dict(x) for x in d.get("hypotheses", [])],
            narrative=d.get("narrative", ""),
        )


@dataclass
class CalibrationRecord:
    snapshot_id: str
    n_runs: int
    per_metric_std: dict[str, float]
    epsilon: float
    per_query_flip_rate: dict[str, float]
    created_at: str = ""

    def to_dict(self) -> dict:
        return {
            "snapshot_id": self.snapshot_id, "n_runs": self.n_runs,
            "per_metric_std": self.per_metric_std, "epsilon": self.epsilon,
            "per_query_flip_rate": self.per_query_flip_rate, "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CalibrationRecord:
        return cls(
            d["snapshot_id"], d["n_runs"], d["per_metric_std"], d["epsilon"],
            d.get("per_query_flip_rate", {}), d.get("created_at", ""),
        )


@dataclass
class GateResult:
    passed: bool
    mode: str
    reasons: list[str]
    significant_regression: bool
    unstable_query_ids: list[str]
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed, "mode": self.mode, "reasons": self.reasons,
            "significant_regression": self.significant_regression,
            "unstable_query_ids": self.unstable_query_ids, "details": self.details,
        }

    @classmethod
    def from_dict(cls, d: dict) -> GateResult:
        return cls(
            d["passed"], d["mode"], d["reasons"], d["significant_regression"],
            d.get("unstable_query_ids", []), d.get("details", {}),
        )
