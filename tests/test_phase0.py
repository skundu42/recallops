from __future__ import annotations

from pathlib import Path

from recallops.adapters.local import LocalIndexAdapter
from recallops.dataset import generate
from recallops.ingest import build_pipeline, ingest
from recallops.phase0 import ConfigChange, Phase0Report, run_phase0
from recallops.store import ProjectStore

CORPUS = Path(__file__).resolve().parent.parent / "examples" / "corpus"


def _dataset(store, adapter):
    manifest = ingest(store, CORPUS, build_pipeline({}), adapter).manifest
    return generate(store, manifest, n=12, seed=0, name="phase0")


def test_run_phase0_produces_gated_report(tmp_path):
    # An exact local adapter: ANN effect is zero and the noise floor is zero, so
    # the harness runs its full path (ANN effect, noise floor, attribution) and
    # emits a §13-gated, provenance-stamped report.
    store = ProjectStore(tmp_path / "s")
    adapter = LocalIndexAdapter(tmp_path / "idx")
    ds = _dataset(store, adapter)
    report = run_phase0(store, CORPUS, ds, adapter, noise_reruns=3, seed=0)

    assert isinstance(report, Phase0Report)
    assert report.provenance["adapter"] == "local"
    assert report.provenance["provider"] == "local"
    assert report.provenance["chunk_count"] > 0
    # exact adapter => no ANN divergence, no serving noise
    assert report.ann_effect["recall@1_divergence"] == 0.0
    assert report.noise_floor["epsilon"] == 0.0
    assert report.noise_floor["rate"] == 0.0
    assert report.gates["fidelity"] == 1.0
    assert report.gates["narrative_violations"] == 0
    # local-embedder caveat is always surfaced
    assert any("local hash provider" in c for c in report.caveats)
    assert report.to_dict()["passed"] == report.passed


def test_phase0_surfaces_serving_noise_with_ann_adapter(tmp_path):
    # A noisy ANN adapter: the harness must surface a real ANN effect (exact vs
    # live divergence) and a positive calibrated epsilon, while attribution stays
    # correct via exact shadow scoring (the FR-6.3 quarantine).
    store = ProjectStore(tmp_path / "s")
    adapter = LocalIndexAdapter(tmp_path / "idx", ann_mode=True, ann_sigma=0.08, seed=1)
    ds = _dataset(store, adapter)
    report = run_phase0(store, CORPUS, ds, adapter, noise_reruns=3, seed=1)

    assert report.noise_floor["epsilon"] > 0.0  # real serving noise measured
    # the never-flaky invariant: even under serving noise the flake rate is bounded
    assert report.noise_floor["rate"] <= 0.02
    # exact ground truth is unaffected by the serving noise
    assert report.ann_effect["exact_recall@5"] >= report.ann_effect["live_recall@5"]
    assert any("under-tuned" in c or "noise is high" in c for c in report.caveats)


def test_phase0_custom_change_measures_true_factor(tmp_path):
    store = ProjectStore(tmp_path / "s")
    adapter = LocalIndexAdapter(tmp_path / "idx")
    ds = _dataset(store, adapter)
    change = ConfigChange(
        "fusion", {"retrieve": {"top_k": 10, "hybrid": {"sparse": "bm25",
                   "fusion": "weighted", "bm25_weight": 0.0}}}, "retrieve")
    report = run_phase0(store, CORPUS, ds, adapter,
                        base_config={"retrieve": {"top_k": 10, "hybrid": {"sparse": "bm25",
                                     "fusion": "weighted", "bm25_weight": 0.6}}},
                        changes=[change], noise_reruns=2, seed=0)
    per = report.attribution["per_change"]
    assert len(per) == 1 and per[0]["name"] == "fusion"
    # fidelity is 1.0 by construction (verified causes actually recover)
    assert report.attribution["fidelity"] == 1.0
