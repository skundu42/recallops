"""Attribution-engine self-test, the Phase-0 go/no-go scorecard (PRD §13).

``run_scorecard`` builds fresh projects on the example corpus and runs four
*known-cause* scenarios, each engineered to produce genuine, stable retrieval
regressions with a single true factor:

- **S1 chunker**, ``markdown_heading(800,120) -> fixed_token(60,0)`` on
  multi-source queries, where a second expected document is evicted from the
  top-5 (true factor ``chunk``).
- **S2 fusion**, ``bm25_weight 0.45 -> 0.0`` on exact-term queries whose target
  needs the sparse signal to reach the top-5 (true factor ``retrieve``).
- **S3 embed**, local ``hash-v1 seed=0 -> hash-v1b seed=7`` under dense-only
  retrieval, a black-box characterization verified only by the revert arm (true
  factor ``embed``).
- **S4 corpus drift**, three near-duplicate distractor docs injected alongside a
  no-op index config touch under dense-only retrieval with mirror docs, so the
  target is pushed past the top-5 (true factor ``corpus``).

For each scenario the engine runs ingest A/B -> eval -> diff -> funnel -> deep
counterfactual attribution -> narrative, and the five §13 metrics are measured:
coverage, fidelity, stage accuracy, and narrative faithfulness per scenario, plus
a single global noise-floor from ANN-rebuild re-runs of an unchanged snapshot
after calibration. Coverage and stage accuracy aggregate across scenarios
weighted by regressed-query count; fidelity is the min across scenarios; the
noise floor is the max observed rate.
"""
from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .ablation import build_arms, enumerate_factors, run_arms
from .adapters.local import LocalIndexAdapter
from .confirm import confirm_causes, fidelity_check
from .dataset import generate
from .diffing import diff
from .evalrunner import evaluate
from .funnel import failing_stage, funnel_for_query, implicated_factors
from .gating import calibrate
from .ingest import build_pipeline, collection_name, ingest
from .models import GoldenCase, GoldenDataset, SnapshotManifest
from .narrative import narrative_faithfulness_audit, render_narrative
from .retrieval import RetrievalEngine
from .store import ProjectStore

__all__ = ["ScorecardResult", "run_scorecard", "CORPUS_DIR", "GATES"]

CORPUS_DIR = Path(__file__).resolve().parent.parent / "examples" / "corpus"

TOP_K = 5
NOISE_SIGMA = 0.02
NOISE_RERUNS = 3

GATES = {
    "coverage": 0.70,
    "fidelity": 0.995,
    "noise_floor": 0.02,
    "stage_accuracy": 0.80,
    "narrative_violations": 0,
}

_DENSE_ONLY: dict = {"retrieve": {"top_k": 10, "hybrid": None}}
_HYBRID = {"sparse": "bm25", "fusion": "weighted", "bm25_weight": 0.45}

# S1: multi-source queries, a chunker change evicts a second expected doc from
# the top-5 (single-source targets are top-5 robust on this corpus).
_S1_CASES = [
    ("q6", "How do refunds relate to invoices and payment methods?",
     ["billing/refunds.md", "billing/invoices.md"]),
    ("q7", "How do pricing plans and the product roadmap cover SSO?",
     ["sales/pricing.md", "product/roadmap.md", "security/sso.md"]),
    ("q8", "What notice periods apply to price changes and subprocessor onboarding?",
     ["sales/pricing.md", "legal/dpa.md"]),
    ("q9", "How do API keys and OAuth service accounts handle rotation and rate limits?",
     ["api/auth.md", "api/rate-limits.md"]),
    ("q10", "How do incident runbooks and onboarding cover production access?",
     ["ops/incident-runbook.md", "hr/onboarding.md"]),
    ("q11", "How do refunds and invoices describe proration and credit notes?",
     ["billing/refunds.md", "billing/invoices.md"]),
]

# S4: a distinctive phrase absent from the real corpus, mirrored by two docs that
# outrank the target and pushed past the top-5 by three near-duplicate distractors.
_S4_PHRASE = "quokka ledger epoch synchronization protocol relay"
_S4_QUESTION = ("How does the quokka ledger epoch synchronization protocol relay "
                "coordinate state?")
_S4_TARGET_PATH = "sync/target.md"
_S4_TARGET = (
    "# Ledger Synchronization Guide\n\n"
    f"The {_S4_PHRASE} coordinates distributed ledger state across regional nodes.\n\n"
    "## Operational Detail\n\n"
    "Operators configure the regional coordinator with a rotation calendar, a quorum "
    "threshold, and a fallback contact roster. The runbook documents escalation windows, "
    "maintenance freezes, and the archival policy for superseded snapshots.\n"
)
_S4_MIRRORS = {
    "sync/mirror-one.md": f"# Mirror One\n\n{_S4_PHRASE}. {_S4_PHRASE} confirmed.\n",
    "sync/mirror-two.md": f"# Mirror Two\n\n{_S4_PHRASE}. {_S4_PHRASE} verified.\n",
}
_S4_DISTRACTORS = {
    "sync/dup-alpha.md": f"# Dup Alpha\n\n{_S4_PHRASE} {_S4_PHRASE}. {_S4_PHRASE}.\n",
    "sync/dup-beta.md": f"# Dup Beta\n\n{_S4_PHRASE} {_S4_PHRASE}. {_S4_PHRASE} now.\n",
    "sync/dup-gamma.md": f"# Dup Gamma\n\n{_S4_PHRASE} {_S4_PHRASE}. {_S4_PHRASE} here.\n",
}


@dataclass
class ScorecardResult:
    coverage: float
    fidelity: float
    noise_floor: float
    stage_accuracy: float
    narrative_violations: int
    details: dict = field(default_factory=dict)

    def passes_gates(self) -> bool:
        return (
            self.coverage >= GATES["coverage"]
            and self.fidelity >= GATES["fidelity"]
            and self.noise_floor <= GATES["noise_floor"]
            and self.stage_accuracy >= GATES["stage_accuracy"]
            and self.narrative_violations == GATES["narrative_violations"]
        )

    def to_dict(self) -> dict:
        return {
            "coverage": self.coverage,
            "fidelity": self.fidelity,
            "noise_floor": self.noise_floor,
            "stage_accuracy": self.stage_accuracy,
            "narrative_violations": self.narrative_violations,
            "details": self.details,
        }


def _write_corpus(root: Path, docs: dict[str, str]) -> None:
    for rel, text in docs.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def _dataset(name: str, cases: list[tuple[str, str, list[str]]]) -> GoldenDataset:
    return GoldenDataset(f"{name}-v1", [
        GoldenCase(id=cid, question=q, expected_sources=list(src), tags=[])
        for cid, q, src in cases
    ])


def _run_scenario(name: str, store: ProjectStore, man_a: SnapshotManifest,
                  man_b: SnapshotManifest, dataset: GoldenDataset, source_dir: Path,
                  true_factor: str) -> dict:
    """Full known-cause pass for one scenario -> per-scenario §13 tallies.

    Runs eval A/B, the snapshot diff, per-query funnel attribution, the deep
    counterfactual arms and confirmation rule, then renders and audits each
    narrative. Returns the regressed/covered/stage-correct/violation counts and
    the scenario fidelity used to aggregate the scorecard.
    """
    eval_a = evaluate(store, man_a, dataset, adapter=None, mode="replay")
    eval_b = evaluate(store, man_b, dataset, adapter=None, mode="replay")
    diffres = diff(store, man_a, man_b, dataset, eval_a, eval_b)
    regressed = diffres.by_class("regressed", stable_only=True)

    engine_a = RetrievalEngine(store, man_a)
    engine_b = RetrievalEngine(store, man_b)
    funnels = {
        qd.query_id: funnel_for_query(engine_a, engine_b, qd, dataset.case(qd.query_id),
                                      diffres.alignment)
        for qd in regressed
    }

    factors = enumerate_factors(diffres)
    arms = build_arms(factors)
    arm_results = run_arms(store, arms, man_a, man_b, source_dir, dataset,
                           checkpoint_key=f"scorecard:{name}")
    reports = confirm_causes(diffres, dataset, arm_results, arms, funnels,
                             diffres.alignment, recovery_threshold="rank1", top_k=TOP_K)

    chunk_texts = engine_b.chunk_texts()
    covered = stage_ok = violations = 0
    per_query: dict[str, dict] = {}
    for qd in regressed:
        qid = qd.query_id
        report = reports.get(qid)
        has_cause = bool(report and report.verified_causes)
        covered += int(has_cause)

        stage = failing_stage(funnels[qid], TOP_K)
        implicated = implicated_factors(stage, diffres.config_diff)
        named = true_factor in implicated
        stage_ok += int(named)

        if report is not None:
            narrative = render_narrative(report, dataset.case(qid), chunk_texts)
            violations += len(narrative_faithfulness_audit(report, narrative))

        per_query[qid] = {
            "before_rank": qd.before.target_rank,
            "after_rank": qd.after.target_rank,
            "failing_stage": stage,
            "implicated": implicated,
            "stage_correct": named,
            "verified": has_cause,
            "verified_factors": sorted(c.factor for c in (report.verified_causes if report else [])),
        }

    fidelity = fidelity_check(reports, arm_results, arms, dataset, TOP_K, diffres=diffres)
    return {
        "true_factor": true_factor,
        "regressed": len(regressed),
        "covered": covered,
        "stage_ok": stage_ok,
        "violations": violations,
        "fidelity": fidelity,
        "config_diff": sorted(diffres.config_diff),
        "per_query": per_query,
    }


def _scenario_s1(root: Path) -> dict:
    store = ProjectStore(root / "s1")
    man_a = ingest(store, CORPUS_DIR, build_pipeline({}), None).manifest
    man_b = ingest(store, CORPUS_DIR, build_pipeline({
        "chunker": {"tool": "recall.chunkers.fixed_token",
                    "params": {"max_tokens": 60, "overlap": 0}},
    }), None).manifest
    dataset = _dataset("scorecard-s1", _S1_CASES)
    return _run_scenario("s1", store, man_a, man_b, dataset, CORPUS_DIR, "chunk")


def _scenario_s2(root: Path, seed: int, n: int) -> dict:
    store = ProjectStore(root / "s2")
    man_a = ingest(store, CORPUS_DIR, build_pipeline(
        {"retrieve": {"top_k": 10, "hybrid": dict(_HYBRID)}}), None).manifest
    man_b = ingest(store, CORPUS_DIR, build_pipeline(
        {"retrieve": {"top_k": 10,
                      "hybrid": {"sparse": "bm25", "fusion": "weighted", "bm25_weight": 0.0}}}),
        None).manifest
    dataset = generate(store, man_a, n=n, seed=seed, name="scorecard-s2")
    return _run_scenario("s2", store, man_a, man_b, dataset, CORPUS_DIR, "retrieve")


def _scenario_s3(root: Path, seed: int, n: int) -> dict:
    store = ProjectStore(root / "s3")
    man_a = ingest(store, CORPUS_DIR, build_pipeline(_DENSE_ONLY), None).manifest
    man_b = ingest(store, CORPUS_DIR, build_pipeline({
        **_DENSE_ONLY,
        "embedding": {"provider": "local", "model": "hash-v1b", "dims": 256,
                      "params": {"seed": 7, "ngram": [1, 2]}},
    }), None).manifest
    dataset = generate(store, man_a, n=n, seed=seed, name="scorecard-s3")
    return _run_scenario("s3", store, man_a, man_b, dataset, CORPUS_DIR, "embed")


def _scenario_s4(root: Path) -> dict:
    dir_a = root / "s4-a"
    dir_b = root / "s4-b"
    shutil.copytree(CORPUS_DIR, dir_a)
    shutil.copytree(CORPUS_DIR, dir_b)
    _write_corpus(dir_a, {**_S4_MIRRORS, _S4_TARGET_PATH: _S4_TARGET})
    _write_corpus(dir_b, {**_S4_MIRRORS, _S4_TARGET_PATH: _S4_TARGET, **_S4_DISTRACTORS})

    store = ProjectStore(root / "s4")
    man_a = ingest(store, dir_a, build_pipeline(_DENSE_ONLY), None).manifest
    # B injects three near-duplicate distractors AND a no-op index config touch, so
    # both the corpus and a config factor change together (FR-7.6 / FR-11.1).
    man_b = ingest(store, dir_b, build_pipeline({
        **_DENSE_ONLY, "index": {"adapter": "local", "collection": "scorecard-s4"},
    }), None).manifest
    dataset = _dataset("scorecard-s4",
                       [("qd", _S4_QUESTION, [_S4_TARGET_PATH])])
    return _run_scenario("s4", store, man_a, man_b, dataset, dir_b, "corpus")


def _noise_floor(root: Path, seed: int, n: int) -> dict:
    """Stable-regressed rate over ANN-rebuild re-runs of an unchanged snapshot.

    The index is served through an ``ann_mode`` adapter, calibrated to derive the
    near-tie threshold ``epsilon``; each re-run rebuilds the index with a fresh
    seed and diffs live against the seed-0 baseline. With calibration, ANN-induced
    flips are near-ties excluded from the counts, so the rate stays at the floor,    the never-flaky invariant (FR-9.3). Reported as the max per-rerun rate.
    """
    store = ProjectStore(root / "noise")
    adapter = LocalIndexAdapter(root / "noise-idx", ann_mode=True, ann_sigma=NOISE_SIGMA, seed=seed)
    manifest = ingest(store, CORPUS_DIR, build_pipeline({}), adapter).manifest
    dataset = generate(store, manifest, n=n, seed=seed, name="scorecard-noise")
    calibration = calibrate(store, manifest, dataset, adapter, n_runs=3, seed=seed)

    collection = collection_name(manifest, store.project)
    adapter.rebuild(collection, seed=seed)
    baseline = evaluate(store, manifest, dataset, adapter=adapter, k_values=(1, 5, 10), mode="live")

    n_cases = len(dataset.cases)
    rates: list[float] = []
    total_regressed = 0
    for i in range(1, NOISE_RERUNS + 1):
        adapter.rebuild(collection, seed=seed + i)
        rerun = evaluate(store, manifest, dataset, adapter=adapter, k_values=(1, 5, 10), mode="live")
        diffres = diff(store, manifest, manifest, dataset, baseline, rerun,
                       epsilon=calibration.epsilon)
        stable_regressed = len(diffres.by_class("regressed", stable_only=True))
        total_regressed += stable_regressed
        rates.append(stable_regressed / n_cases if n_cases else 0.0)

    return {
        "epsilon": calibration.epsilon,
        "rate": max(rates) if rates else 0.0,
        "mean_rate": total_regressed / (n_cases * NOISE_RERUNS) if n_cases else 0.0,
        "reruns": NOISE_RERUNS,
        "n_cases": n_cases,
    }


def run_scorecard(workdir: Path, seed: int = 0) -> ScorecardResult:
    """Run the four known-cause scenarios plus the noise floor and return the
    aggregated §13 scorecard.

    Coverage and stage accuracy are weighted by regressed-query count across
    scenarios; fidelity is the minimum scenario fidelity; narrative violations sum
    across every rendered report; the noise floor is the single ANN-rebuild
    measurement. Scenario projects are built under a fresh temporary directory
    inside ``workdir`` and removed afterward.
    """
    if not CORPUS_DIR.is_dir():
        raise FileNotFoundError(
            f"example corpus not found at {CORPUS_DIR}; the scorecard requires the "
            "source-tree examples/corpus."
        )
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="recall-scorecard-", dir=str(workdir)))

    try:
        scenarios = {
            "s1_chunk": _scenario_s1(root),
            "s2_retrieve": _scenario_s2(root, seed=seed, n=50),
            "s3_embed": _scenario_s3(root, seed=seed, n=50),
            "s4_corpus": _scenario_s4(root),
        }
        noise = _noise_floor(root, seed=seed, n=30)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    total_regressed = sum(s["regressed"] for s in scenarios.values())
    total_covered = sum(s["covered"] for s in scenarios.values())
    total_stage_ok = sum(s["stage_ok"] for s in scenarios.values())
    violations = sum(s["violations"] for s in scenarios.values())

    coverage = total_covered / total_regressed if total_regressed else 1.0
    stage_accuracy = total_stage_ok / total_regressed if total_regressed else 1.0
    fidelity = min((s["fidelity"] for s in scenarios.values()), default=1.0)

    details = {
        "scenarios": scenarios,
        "noise": noise,
        "totals": {
            "regressed": total_regressed,
            "covered": total_covered,
            "stage_ok": total_stage_ok,
            "violations": violations,
        },
        "gates": GATES,
    }
    return ScorecardResult(
        coverage=coverage,
        fidelity=fidelity,
        noise_floor=noise["rate"],
        stage_accuracy=stage_accuracy,
        narrative_violations=violations,
        details=details,
    )
