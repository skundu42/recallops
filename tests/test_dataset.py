from __future__ import annotations

import json

import pytest

from recallops import dataset
from recallops.ingest import build_pipeline, ingest
from recallops.models import GoldenCase, GoldenDataset
from recallops.store import ProjectStore

PARAPHRASE_PREFIXES = ("How is ", "What happens when ", "What happens if ")


@pytest.fixture(scope="module")
def project(tmp_path_factory, corpus_dir):
    store = ProjectStore(tmp_path_factory.mktemp("proj"))
    report = ingest(store, corpus_dir, build_pipeline({}), adapter=None)
    yield store, report.manifest, corpus_dir
    store.close()


def _chunk_count(store: ProjectStore, manifest) -> int:
    key = manifest.artifacts["chunks_uri"].rsplit("/", 1)[-1].removesuffix(".parquet")
    return len(store.get_chunks(key))


def _normalize(question: str) -> str:
    return " ".join(question.lower().split())


def test_generate_returns_n_unique_cases_with_valid_sources(project):
    store, manifest, corpus = project
    ds = dataset.generate(store, manifest, n=12, seed=0)
    assert ds.dataset_id == "golden-v1"
    assert len(ds.cases) == 12
    assert len({_normalize(c.question) for c in ds.cases}) == 12
    for i, case in enumerate(ds.cases):
        assert case.id == f"q_{i:03d}"
        assert case.origin == "synthetic"
        assert case.question.endswith("?")
        assert len(case.expected_sources) == 1
        assert (corpus / case.expected_sources[0]).is_file()


def test_generate_tags_topdir_plus_both_kinds(project):
    store, manifest, _ = project
    ds = dataset.generate(store, manifest, n=12, seed=0)
    kinds = set()
    for case in ds.cases:
        top, kind = case.tags
        assert top == case.expected_sources[0].split("/")[0]
        assert kind in ("exact-term", "paraphrase")
        kinds.add(kind)
    assert kinds == {"exact-term", "paraphrase"}


def test_generate_question_templates(project):
    store, manifest, _ = project
    ds = dataset.generate(store, manifest, n=10, seed=0)
    for case in ds.cases:
        if "exact-term" in case.tags:
            assert case.question.startswith("What does the documentation say about ")
        else:
            assert case.question.startswith(PARAPHRASE_PREFIXES)


def test_generate_deterministic_given_seed(project):
    store, manifest, _ = project
    first = dataset.generate(store, manifest, n=10, seed=3)
    second = dataset.generate(store, manifest, n=10, seed=3)
    assert first.to_dict() == second.to_dict()
    other = dataset.generate(store, manifest, n=10, seed=4)
    assert other.to_dict() != first.to_dict()


def test_generate_caps_at_available_unique_cases(project):
    store, manifest, _ = project
    ds = dataset.generate(store, manifest, n=10_000, seed=0)
    assert 12 <= len(ds.cases) <= _chunk_count(store, manifest)
    assert len({_normalize(c.question) for c in ds.cases}) == len(ds.cases)


def test_generate_name_param_and_store_roundtrip(project):
    store, manifest, _ = project
    ds = dataset.generate(store, manifest, n=4, seed=0, name="gold")
    assert ds.dataset_id == "gold-v1"
    assert ds.version == 1
    store.save_dataset(ds)
    assert store.get_dataset("gold").dataset_id == "gold-v1"


def test_generate_with_llm_callable(project):
    store, manifest, _ = project
    prompts: list[str] = []

    def llm(prompt: str) -> str:
        prompts.append(prompt)
        return f"LLM question {len(prompts)}?"

    ds = dataset.generate(store, manifest, n=5, seed=0, llm=llm)
    assert [c.question for c in ds.cases] == [f"LLM question {i}?" for i in range(1, 6)]
    assert all(c.origin == "synthetic" for c in ds.cases)
    assert len(prompts) == 5


def test_generate_dedupes_and_returns_fewer_when_exhausted(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    body = (
        "\n\nThe zephyrix calibration matrix requires quarterly rebalancing "
        "of flux capacitors.\n"
    )
    for name, title in (("a.md", "# Alpha note"), ("b.md", "# Beta note"),
                        ("c.md", "# Gamma note")):
        (root / name).write_text(title + body)
    store = ProjectStore(tmp_path / "proj")
    report = ingest(store, root, build_pipeline({}), adapter=None)
    ds = dataset.generate(store, report.manifest, n=3, seed=0)
    assert len(ds.cases) == 2
    assert len({_normalize(c.question) for c in ds.cases}) == 2
    assert {c.tags[1] for c in ds.cases} == {"exact-term", "paraphrase"}
    store.close()


def test_paraphrase_templates_by_sentence_shape():
    when_q = dataset._paraphrase_question(
        ["the", "alert", "fires", "when", "replication", "lag", "exceeds", "five", "seconds"]
    )
    assert when_q == "What happens when replication lag exceeds five seconds?"
    if_q = dataset._paraphrase_question(
        ["retry", "the", "bootstrap", "if", "the", "checksum", "fails"]
    )
    assert if_q == "What happens if the checksum fails?"
    how_q = dataset._paraphrase_question(
        ["gateway", "timeouts", "are", "configurable", "per", "route"]
    )
    assert how_q == "How is gateway timeouts handled?"


def test_import_recallops_json(tmp_path):
    body = {
        "dataset_id": "golden-v4",
        "cases": [{
            "id": "q_014",
            "question": "What is the refund policy for annual plans?",
            "expected_sources": ["billing/refunds.md"],
            "expected_spans": None,
            "tags": ["billing", "exact-term"],
            "origin": "production",
            "source_trace": "langfuse:tr_88a1",
        }],
    }
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(body))
    ds = dataset.import_file(path, "imported-v1")
    assert ds.dataset_id == "imported-v1"
    assert len(ds.cases) == 1
    case = ds.cases[0]
    assert case.id == "q_014"
    assert case.question == "What is the refund policy for annual plans?"
    assert case.expected_sources == ["billing/refunds.md"]
    assert case.tags == ["billing", "exact-term"]
    assert case.origin == "production"
    assert case.source_trace == "langfuse:tr_88a1"


def test_import_ragas_jsonl(tmp_path):
    lines = [
        {"question": "How do refunds work?", "reference_contexts": ["billing/refunds.md"]},
        {"question": "What is SSO?", "ground_truth": "security/sso.md"},
    ]
    path = tmp_path / "ragas.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in lines))
    ds = dataset.import_file(path, "ragas-v1")
    assert ds.dataset_id == "ragas-v1"
    assert [c.id for c in ds.cases] == ["q_000", "q_001"]
    assert ds.cases[0].question == "How do refunds work?"
    assert ds.cases[0].expected_sources == ["billing/refunds.md"]
    assert ds.cases[1].expected_sources == ["security/sso.md"]


def test_import_deepeval_jsonl(tmp_path):
    lines = [
        {"input": "How are invoices numbered?", "expected_output": "Sequentially per account.",
         "context": ["billing/invoices.md"]},
    ]
    path = tmp_path / "deepeval.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in lines))
    ds = dataset.import_file(path, "deepeval-v1")
    assert ds.cases[0].question == "How are invoices numbered?"
    assert ds.cases[0].expected_sources == ["billing/invoices.md"]


def test_import_unknown_record_raises(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"prompt": "nope"}))
    with pytest.raises(ValueError):
        dataset.import_file(path, "bad-v1")


def test_mine_jsonl_sets_production_origin_and_trace(tmp_path):
    lines = [
        {"query": "Why was my card declined?", "expected_sources": ["billing/refunds.md"],
         "trace_id": "langfuse:tr_88a1"},
        {"query": "How do I rotate api keys?", "expected_sources": ["api/auth.md"]},
    ]
    path = tmp_path / "traces.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in lines))
    ds = dataset.mine_jsonl(path, "mined-v1")
    assert ds.dataset_id == "mined-v1"
    assert [c.id for c in ds.cases] == ["q_000", "q_001"]
    assert all(c.origin == "production" for c in ds.cases)
    assert ds.cases[0].source_trace == "langfuse:tr_88a1"
    assert ds.cases[1].source_trace is None
    assert ds.cases[0].question == "Why was my card declined?"
    assert ds.cases[1].expected_sources == ["api/auth.md"]


def _toy_dataset() -> GoldenDataset:
    return GoldenDataset("toy-v1", [
        GoldenCase(id="q_000", question="A?", expected_sources=["billing/refunds.md"],
                   tags=["billing", "exact-term"]),
        GoldenCase(id="q_001", question="B?", expected_sources=["billing/invoices.md"],
                   tags=["billing", "paraphrase"]),
        GoldenCase(id="q_002", question="C?", expected_sources=["api/auth.md"],
                   tags=["api", "exact-term"]),
    ])


def test_curate_filters_rejected_keeps_rest():
    ds = _toy_dataset()
    out = dataset.curate(ds, {"q_000": "accept", "q_001": "reject"})
    assert out.dataset_id == "toy-v1"
    assert [c.id for c in out.cases] == ["q_000", "q_002"]
    assert [c.id for c in ds.cases] == ["q_000", "q_001", "q_002"]


def test_curate_rejects_unknown_decision():
    with pytest.raises(ValueError):
        dataset.curate(_toy_dataset(), {"q_000": "maybe"})


def test_stratification_report_counts_tags():
    report = dataset.stratification_report(_toy_dataset())
    assert report == {"api": 1, "billing": 2, "exact-term": 2, "paraphrase": 1}


def test_stratification_report_on_generated(project):
    store, manifest, _ = project
    ds = dataset.generate(store, manifest, n=12, seed=0)
    report = dataset.stratification_report(ds)
    assert sum(v for k, v in report.items() if k in ("exact-term", "paraphrase")) == 12
    assert report["exact-term"] == 6
    assert report["paraphrase"] == 6
