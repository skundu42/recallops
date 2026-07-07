"""Phase-0 real-project validation harness (PRD §13, §15).

Unlike ``scorecard`` (a self-test on the synthetic example corpus with the
deterministic local embedder and an exact adapter), this harness runs the §13
go/no-go against a *real serving stack*, a real vector DB adapter (e.g.
pgvector) and, when a key is present, a real embedding provider (OpenAI), on
whatever corpus and golden set a design partner supplies. It adds the two
measurements the synthetic scorecard cannot make:

1. **ANN effect**, exact shadow ranking (FR-6.2) vs the serving index's live
   ranking. A large gap is the "index/approximation effect" (FR-6.3) that
   attribution must quarantine; it is why shadow scoring exists.
2. **Real noise floor**, calibration ε and the stable-regressed rate over
   no-change re-runs of the *real* index (FR-9.2/9.3). The never-flaky claim
   only means something against a real approximate index.

Attribution quality (coverage / fidelity / stage-accuracy / narrative) is
measured by running known-cause config changes through the full engine on the
supplied corpus. Every number is stamped with provenance (provider, adapter,
probes, chunk count, cost) so the result is auditable and reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .ablation import build_arms, enumerate_factors, run_arms
from .adapters.base import VectorAdapter
from .confirm import confirm_causes, fidelity_check
from .diffing import diff
from .evalrunner import evaluate
from .funnel import failing_stage, funnel_for_query, implicated_factors
from .gating import calibrate
from .ingest import build_pipeline, collection_name, ingest
from .models import GoldenDataset, SnapshotManifest
from .narrative import narrative_faithfulness_audit, render_narrative
from .pipeline.providers import EmbeddingProvider, estimate_embed_cost
from .store import ProjectStore

# §13 gates (identical to scorecard).
GATES = {
    "coverage": 0.70,
    "fidelity": 0.995,
    "noise_floor": 0.02,
    "stage_accuracy": 0.80,
    "narrative_violations": 0,
}
TOP_K = 5


@dataclass
class ConfigChange:
    """A known-cause pipeline change with its single true factor."""
    name: str
    overrides: dict
    true_factor: str


# The default known-cause changes, corpus-agnostic. A real partner replaces these
# with the actual config change that caused a real regression.
def default_changes() -> list[ConfigChange]:
    return [
        ConfigChange("chunker",
                     {"chunker": {"tool": "recall.chunkers.fixed_token",
                                  "params": {"max_tokens": 40, "overlap": 0}}},
                     "chunk"),
        ConfigChange("fusion",
                     {"retrieve": {"top_k": 10,
                                   "hybrid": {"sparse": "bm25", "fusion": "weighted",
                                              "bm25_weight": 0.0}}},
                     "retrieve"),
    ]


@dataclass
class Phase0Report:
    provenance: dict
    ann_effect: dict
    noise_floor: dict
    attribution: dict
    gates: dict
    caveats: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        g = self.gates
        return (g["coverage"] >= GATES["coverage"]
                and g["fidelity"] >= GATES["fidelity"]
                and g["noise_floor"] <= GATES["noise_floor"]
                and g["stage_accuracy"] >= GATES["stage_accuracy"]
                and g["narrative_violations"] == GATES["narrative_violations"])

    def to_dict(self) -> dict:
        return {
            "provenance": self.provenance,
            "ann_effect": self.ann_effect,
            "noise_floor": self.noise_floor,
            "attribution": self.attribution,
            "gates": self.gates,
            "passed": self.passed,
            "caveats": self.caveats,
        }


def _ann_effect(store: ProjectStore, manifest: SnapshotManifest, dataset: GoldenDataset,
                adapter: VectorAdapter) -> dict:
    """Exact shadow ranking vs the serving index's live ranking (FR-6.2/6.3)."""
    replay = evaluate(store, manifest, dataset, adapter=None, k_values=(1, 5, 10), mode="replay")
    live = evaluate(store, manifest, dataset, adapter=adapter, k_values=(1, 5, 10), mode="live")
    return {
        "exact_recall@1": replay.aggregate["recall@1"],
        "exact_recall@5": replay.aggregate["recall@5"],
        "live_recall@1": live.aggregate["recall@1"],
        "live_recall@5": live.aggregate["recall@5"],
        "recall@1_divergence": replay.aggregate["recall@1"] - live.aggregate["recall@1"],
        "recall@5_divergence": replay.aggregate["recall@5"] - live.aggregate["recall@5"],
    }


def _noise_floor(store: ProjectStore, manifest: SnapshotManifest, dataset: GoldenDataset,
                 adapter: VectorAdapter, reruns: int, seed: int) -> dict:
    """Calibrated ε and the stable-regressed rate over no-change re-runs of the
    real serving index (FR-9.2/9.3). Any re-run flagged regressed under a
    no-change diff is a gate flake."""
    calibration = calibrate(store, manifest, dataset, adapter, n_runs=3, seed=seed)
    collection = collection_name(manifest, store.project)
    supports_rebuild = adapter.capabilities().supports_rebuild
    if supports_rebuild:
        adapter.rebuild(collection, seed=seed)
    baseline = evaluate(store, manifest, dataset, adapter=adapter, k_values=(1, 5, 10), mode="live")
    n_cases = len(dataset.cases)
    rates: list[float] = []
    for i in range(1, reruns + 1):
        if supports_rebuild:
            adapter.rebuild(collection, seed=seed + i)
        rerun = evaluate(store, manifest, dataset, adapter=adapter, k_values=(1, 5, 10), mode="live")
        dr = diff(store, manifest, manifest, dataset, baseline, rerun, epsilon=calibration.epsilon)
        rates.append(len(dr.by_class("regressed", stable_only=True)) / n_cases if n_cases else 0.0)
    return {
        "epsilon": calibration.epsilon,
        "rate": max(rates) if rates else 0.0,
        "reruns": reruns,
        "per_rerun_rates": rates,
    }


def _run_change(store: ProjectStore, source_dir: Path, base: SnapshotManifest,
                dataset: GoldenDataset, change: ConfigChange,
                base_config: dict, provider: EmbeddingProvider | None) -> dict:
    """Ingest the changed pipeline, run full attribution in replay (exact KNN,
    serving-index-independent), and measure coverage / stage-accuracy / fidelity
    / narrative for this known single-cause change."""
    variant_cfg = {**base_config, **change.overrides}
    man_b = ingest(store, source_dir, build_pipeline(variant_cfg), adapter=None,
                   provider=provider).manifest
    ev_a = evaluate(store, base, dataset, adapter=None, k_values=(1, 5, 10), mode="replay")
    ev_b = evaluate(store, man_b, dataset, adapter=None, k_values=(1, 5, 10), mode="replay")
    dr = diff(store, base, man_b, dataset, ev_a, ev_b)
    regressed = dr.by_class("regressed", stable_only=True)
    from .retrieval import RetrievalEngine
    engine_a = RetrievalEngine(store, base)
    engine_b = RetrievalEngine(store, man_b)
    funnels = {
        q.query_id: funnel_for_query(engine_a, engine_b, q, dataset.case(q.query_id), dr.alignment)
        for q in regressed
    }
    arms = build_arms(enumerate_factors(dr))
    arm_results = run_arms(store, arms, base, man_b, source_dir, dataset,
                           checkpoint_key=f"phase0:{change.name}")
    reports = confirm_causes(dr, dataset, arm_results, arms, funnels, dr.alignment,
                             recovery_threshold="rank1", top_k=TOP_K)
    chunk_texts = engine_b.chunk_texts()
    covered = stage_ok = violations = 0
    for q in regressed:
        rep = reports.get(q.query_id)
        covered += int(bool(rep and rep.verified_causes))
        stage = failing_stage(funnels[q.query_id], TOP_K)
        stage_ok += int(change.true_factor in implicated_factors(stage, dr.config_diff))
        if rep is not None:
            text = render_narrative(rep, dataset.case(q.query_id), chunk_texts)
            violations += len(narrative_faithfulness_audit(rep, text))
    fidelity = fidelity_check(reports, arm_results, arms, dataset, TOP_K, diffres=dr)
    return {
        "name": change.name,
        "true_factor": change.true_factor,
        "regressed": len(regressed),
        "covered": covered,
        "stage_correct": stage_ok,
        "fidelity": fidelity,
        "narrative_violations": violations,
    }


def run_phase0(store: ProjectStore, source_dir: Path, dataset: GoldenDataset,
               adapter: VectorAdapter, base_config: dict | None = None,
               provider: EmbeddingProvider | None = None,
               changes: list[ConfigChange] | None = None,
               noise_reruns: int = 10, seed: int = 0,
               cost_usd: float = 0.0) -> Phase0Report:
    """Run the full Phase-0 measurement against a real serving stack.

    ``adapter`` is the real serving index (e.g. pgvector); ``provider`` the real
    embedding provider (None => the pipeline's configured provider, typically the
    offline local embedder). Ingests the baseline with write-through to
    ``adapter``, measures the ANN effect and real noise floor against it, then
    runs each known-cause ``change`` for attribution quality.
    """
    base_config = base_config or {}
    changes = changes if changes is not None else default_changes()

    base = ingest(store, source_dir, build_pipeline(base_config), adapter=adapter,
                  provider=provider).manifest
    ann = _ann_effect(store, base, dataset, adapter)
    noise = _noise_floor(store, base, dataset, adapter, reruns=noise_reruns, seed=seed)

    per_change = [_run_change(store, source_dir, base, dataset, c, base_config, provider)
                  for c in changes]
    total_reg = sum(c["regressed"] for c in per_change)
    coverage = (sum(c["covered"] for c in per_change) / total_reg) if total_reg else 0.0
    stage_acc = (sum(c["stage_correct"] for c in per_change) / total_reg) if total_reg else 0.0
    fidelity = min((c["fidelity"] for c in per_change), default=1.0)
    violations = sum(c["narrative_violations"] for c in per_change)

    caveats: list[str] = []
    prov_name = (provider.provider if provider else base.pipeline.stage("embed").tool)
    if prov_name == "local":
        caveats.append(
            "Embeddings are the offline local hash provider, not a production model. "
            "The ANN-effect and noise-floor numbers reflect real serving behaviour, "
            "but coverage/stage-accuracy on real regressions require real embeddings.")
    if noise["epsilon"] > 0.1:
        caveats.append(
            f"Serving-index noise is high (epsilon={noise['epsilon']:.3f}); the index is "
            "under-tuned for gating. Raise pgvector ivfflat probes (or index params), "
            "attribution stays correct via exact shadow scoring regardless.")
    if total_reg == 0:
        caveats.append(
            "No stable regressions were produced by the default config changes on this "
            "corpus; supply a real known-regression change for a meaningful coverage number.")

    return Phase0Report(
        provenance={
            "provider": prov_name,
            "adapter": adapter.name,
            "probes": getattr(adapter, "probes", None),
            "doc_count": base.corpus.doc_count,
            "chunk_count": base.corpus.chunk_count,
            "dataset_cases": len(dataset.cases),
            "snapshot_id": base.snapshot_id,
            "cost_usd": cost_usd,
        },
        ann_effect=ann,
        noise_floor=noise,
        attribution={
            "coverage": coverage,
            "fidelity": fidelity,
            "stage_accuracy": stage_acc,
            "narrative_violations": violations,
            "total_regressed": total_reg,
            "per_change": per_change,
        },
        gates={
            "coverage": coverage,
            "fidelity": fidelity,
            "noise_floor": noise["rate"],
            "stage_accuracy": stage_acc,
            "narrative_violations": violations,
        },
        caveats=caveats,
    )


def estimate_cost(provider: EmbeddingProvider, source_dir: Path) -> float:
    """Rough embedding cost for the baseline + N config-change re-ingests."""
    texts = []
    for path in sorted(Path(source_dir).rglob("*")):
        if path.is_file() and path.suffix in (".md", ".txt"):
            texts.append(path.read_text(encoding="utf-8", errors="ignore"))
    return estimate_embed_cost(provider, texts)["usd"]
