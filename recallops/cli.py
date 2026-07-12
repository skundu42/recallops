"""``recall`` command-line interface (PRD §11).

The integration surface: every command wires the real engine modules end to
end over a ``ProjectStore`` rooted at the current directory and a
``recall.yaml`` (`config.ProjectConfig`). Local provider is $0 and auto-approves
every cost gate; any provider-billed operation prints an estimate and requires
``--yes`` or ``--max-cost`` (Principle 5). Deep attribution is the two-phase
tail: funnel attribution is fast and always available; verified causes come
from the counterfactual pass.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from . import ablation, confirm
from .config import (
    ProjectConfig,
    build_adapter,
    parse_embedding_spec,
)
from .dataset import (
    curate as ds_curate,
)
from .dataset import (
    generate as ds_generate,
)
from .dataset import (
    import_file as ds_import,
)
from .dataset import (
    mine_jsonl as ds_mine,
)
from .dataset import (
    stratification_report,
)
from .diffing import _metric_k as _metric_k
from .diffing import align_chunks
from .diffing import diff as run_diff
from .evalrunner import evaluate
from .funnel import failing_stage, funnel_for_query, implicated_factors
from .gating import (
    GateNotCalibrated,
    evaluate_gate,
    parse_fail_if,
)
from .gating import (
    calibrate as run_calibrate,
)
from .ingest import build_pipeline
from .ingest import ingest as run_ingest
from .models import (
    CalibrationRecord,
    DiffResult,
    GateResult,
    GoldenDataset,
    SnapshotManifest,
    StageSpec,
)
from .narrative import render_narrative
from .pipeline import chunkers, parsers
from .pipeline.providers import estimate_embed_cost, get_provider
from .report import (
    attribution_to_dict,
    compare_report_markdown,
    diff_report_html,
    diff_summary_markdown,
    diff_table,
    eval_table,
    render_diff_json,
)
from .retrieval import RetrievalEngine
from .store import ProjectStore

console = Console()

DEFAULT_HYBRID_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


# -- shared helpers -----------------------------------------------------------


def _load_config(path: str) -> ProjectConfig:
    p = Path(path)
    if not p.exists():
        raise click.ClickException(f"{path} not found; run `recall init` first.")
    return ProjectConfig.load(p)


def _store() -> ProjectStore:
    return ProjectStore(Path("."))


def _parse_k(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(",") if x.strip())


def _resolve(store: ProjectStore, ref: str) -> SnapshotManifest:
    try:
        return store.resolve_snapshot(ref)
    except (KeyError, ValueError) as exc:
        raise click.ClickException(f"cannot resolve snapshot {ref!r}: {exc}")


def _get_dataset(store: ProjectStore, dataset_id: str | None) -> GoldenDataset:
    if dataset_id:
        try:
            return store.get_dataset(dataset_id)
        except KeyError:
            raise click.ClickException(f"dataset {dataset_id!r} not found")
    names = store.list_datasets()
    if not names:
        raise click.ClickException("no datasets; run `recall dataset generate` first.")
    return store.get_dataset(names[-1])


def _pipeline_config(cfg: ProjectConfig, chunker: str | None = None,
                     chunk_params: str | None = None, embedding: str | None = None,
                     adapter: str | None = None) -> dict:
    pc = copy.deepcopy(cfg.pipeline)
    if chunker:
        pc.setdefault("chunker", {})["tool"] = chunker
        if chunk_params:
            pc["chunker"]["params"] = json.loads(chunk_params)
    elif chunk_params:
        pc.setdefault("chunker", {})["params"] = json.loads(chunk_params)
    if embedding:
        pc["embedding"] = parse_embedding_spec(embedding)
    if adapter:
        pc.setdefault("index", {})["adapter"] = adapter
    return pc


def _provider_for(pipeline) -> object:
    embed = pipeline.stage("embed")
    if embed is None:
        raise click.ClickException("pipeline has no embed stage")
    return get_provider(dict(embed.params))


def _with_gate_k(ks: tuple[int, ...], cfg: ProjectConfig) -> tuple[int, ...]:
    """Merge the configured gate metric's k into the eval k-values so diff() and
    evaluate_gate() can always find ``metrics[primary_metric]`` / ``hit_at[k]``,
    even for a non-default metric like recall@7 (keeps eval consistent with the
    primary_metric threaded through calibrate/diff)."""
    k = _metric_k(cfg.gate.get("primary_metric", "recall@5"))
    return tuple(sorted(set(ks) | {k}))


def _cost_gate(est_usd: float, max_cost: float | None, yes: bool, label: str) -> None:
    if est_usd <= 0.0:
        return
    console.print(f"[yellow]Estimated {label} cost: ${est_usd:.2f}[/yellow]")
    if max_cost is not None and est_usd <= max_cost:
        console.print(f"Within budget (--max-cost ${max_cost:.2f}); proceeding.")
        return
    if yes:
        return
    raise click.ClickException(
        f"Estimated {label} cost ${est_usd:.2f} requires approval; pass --yes or --max-cost N."
    )


def _warn_if_empty_collection(adapter, manifest: SnapshotManifest, store: ProjectStore) -> None:
    """Live metrics against an empty serving collection are silently all-zero
    (a wrong DSN/path or an ingest that skipped write-through looks like a
    catastrophic regression). A MISSING collection already fails loudly at
    query time, so a count that raises is left to that path."""
    from .ingest import collection_name

    try:
        n = adapter.count(collection_name(manifest, store.project))
    except Exception:
        return
    if n == 0:
        console.print(
            "[yellow]warning:[/yellow] the serving collection for this snapshot is "
            "empty (0 vectors); live dense metrics will be zeros. Did ingest "
            "write through this adapter (check adapter config / DSN / path)?"
        )


def _estimate_ingest_cost(store: ProjectStore, source_dir: Path, pipeline, provider) -> dict:
    from . import hashing
    from .ingest import embedding_keys

    parse_stage = pipeline.stage("parse")
    chunk_stage = pipeline.stage("chunk")
    records = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or path.suffix not in (".md", ".txt"):
            continue
        rel = path.relative_to(source_dir).as_posix()
        raw = path.read_bytes()
        parsed = parsers.parse(rel, raw, tool=parse_stage.tool)
        records.extend(chunkers.chunk_doc(
            hashing.doc_id(raw), parsed.text, chunk_stage.tool, chunk_stage.params,
            parse_stage.id, chunk_stage.id,
        ))
    keys = embedding_keys(records, provider)
    text_by_key: dict[str, str] = {}
    for record, key in zip(records, keys):
        text_by_key.setdefault(key, record.text)
    missing = store.missing_embedding_keys(keys)
    return estimate_embed_cost(provider, [text_by_key[k] for k in missing])


def _funnels(store: ProjectStore, dr: DiffResult, dataset: GoldenDataset,
             manifest_a: SnapshotManifest, manifest_b: SnapshotManifest,
             top_k: int) -> tuple[dict, RetrievalEngine]:
    engine_a = RetrievalEngine(store, manifest_a)
    engine_b = RetrievalEngine(store, manifest_b)
    case_by_id = {c.id: c for c in dataset.cases}
    reports: dict = {}
    for qd in dr.by_class("regressed"):
        case = case_by_id.get(qd.query_id)
        if case is None:
            continue
        reports[qd.query_id] = funnel_for_query(engine_a, engine_b, qd, case, dr.alignment)
    return reports, engine_b


def _deep_attribute(store: ProjectStore, dr: DiffResult, dataset: GoldenDataset,
                    manifest_a: SnapshotManifest, manifest_b: SnapshotManifest,
                    source_dir: Path, arms_mode: str, max_cost: float | None,
                    yes: bool, top_k: int) -> dict:
    funnels, engine_b = _funnels(store, dr, dataset, manifest_a, manifest_b, top_k)
    factors = ablation.enumerate_factors(dr)
    if not factors:
        return {}
    arms = ablation.build_arms(factors, mode=arms_mode)
    plan = ablation.plan_arms(
        store, manifest_a, manifest_b, source_dir, arms,
        lambda spec: get_provider(dict(spec.params)),
    )
    _cost_gate(plan.est_usd, max_cost, yes, "counterfactual arms")
    checkpoint_key = f"attr:{dr.diff_id}"
    results = ablation.run_arms(store, arms, manifest_a, manifest_b, source_dir, dataset,
                               checkpoint_key)
    reports = confirm.confirm_causes(dr, dataset, results, arms, funnels, dr.alignment,
                                     top_k=top_k)
    case_by_id = {c.id: c for c in dataset.cases}
    chunk_texts = engine_b.chunk_texts()
    for qid, rep in reports.items():
        case = case_by_id.get(qid)
        if case is not None:
            rep.narrative = render_narrative(rep, case, chunk_texts)
    store.save_json("attribution", dr.diff_id,
                    {qid: attribution_to_dict(rep) for qid, rep in reports.items()})
    return reports


def _print_funnels(store: ProjectStore, dr: DiffResult, funnels: dict, top_k: int) -> None:
    if not funnels:
        console.print("No regressed queries.")
        return
    table = Table(title="Funnel attribution (regressed queries)")
    table.add_column("Query")
    table.add_column("Fate")
    table.add_column("Failing stage")
    table.add_column("Implicated factors")
    for qid in sorted(funnels):
        f = funnels[qid]
        stage = failing_stage(f, top_k)
        implicated = implicated_factors(stage, dr.config_diff)
        fate = dr.alignment.get(f.target_chunk_before)
        table.add_row(qid, fate.cls if fate else "-", stage, ", ".join(implicated) or "-")
    console.print(table)


def _print_verified(reports: dict) -> None:
    any_cause = False
    for qid in sorted(reports):
        rep = reports[qid]
        for vc in rep.verified_causes:
            any_cause = True
            console.print(
                f"[green]verified[/green] {qid}: {vc.factor} -> rank {vc.recovered_rank} "
                f"([cyan]{vc.arm_id}[/cyan])"
            )
        if rep.narrative:
            console.print(f"  {rep.narrative}")
    if not any_cause:
        console.print("No verified causes.")


# -- top-level group ----------------------------------------------------------


@click.group()
@click.version_option(package_name="recallops")
def main() -> None:
    """RecallOps, retrieval regression testing with verified attribution."""


@main.command()
@click.option("--adapter",
              type=click.Choice(["local", "pgvector", "qdrant", "chroma", "lancedb"]),
              default="local")
@click.option("--source", default="docs", help="Documents directory.")
@click.option("--name", default=None, help="Project name (default: directory name).")
@click.option("--dsn", default=None, help="pgvector DSN (pgvector adapter only).")
@click.option("--force", is_flag=True, help="Overwrite an existing recall.yaml.")
def init(adapter: str, source: str, name: str | None, dsn: str | None, force: bool) -> None:
    """Write recall.yaml and create the .recall store."""
    path = Path("recall.yaml")
    if path.exists() and not force:
        raise click.ClickException("recall.yaml already exists (use --force to overwrite).")
    project = name or Path.cwd().name
    cfg = ProjectConfig.default(project=project, source=source, adapter=adapter, dsn=dsn)
    cfg.save(path)
    # Persist the project namespace so serving collections are qualified per
    # project (later commands read it back from the store's meta).
    ProjectStore(Path("."), project=project)
    console.print(
        f"Initialized RecallOps project [bold]{project}[/bold] "
        f"(adapter={adapter}, source={source}). Wrote recall.yaml and .recall/."
    )


@main.command()
@click.argument("path", required=False)
@click.option("--chunker", default=None, help="Override chunker tool.")
@click.option("--chunk-params", default=None, help="Chunker params as JSON.")
@click.option("--embedding", default=None, help="Override embedding spec provider:model:dims.")
@click.option("--adapter", default=None, help="Override index adapter.")
@click.option("--config", "config_path", default="recall.yaml")
@click.option("--yes", is_flag=True, help="Approve a non-zero embedding cost.")
@click.option("--max-cost", type=float, default=None, help="Embedding cost budget in USD.")
def ingest(path: str | None, chunker: str | None, chunk_params: str | None,
           embedding: str | None, adapter: str | None, config_path: str,
           yes: bool, max_cost: float | None) -> None:
    """Ingest a corpus into a new immutable snapshot."""
    cfg = _load_config(config_path)
    store = _store()
    source_dir = Path(path) if path else Path(cfg.source)
    if not source_dir.exists():
        raise click.ClickException(f"source directory {source_dir} does not exist.")
    pc = _pipeline_config(cfg, chunker, chunk_params, embedding, adapter)
    pipeline = build_pipeline(pc)
    provider = _provider_for(pipeline)

    if provider.price_per_1k_tokens() > 0.0:
        est = _estimate_ingest_cost(store, source_dir, pipeline, provider)
        _cost_gate(est["usd"], max_cost, yes, "embedding")

    parent = None
    if store.list_snapshots():
        parent = store.resolve_snapshot("latest").snapshot_id
    vec_adapter = build_adapter(cfg, store)
    try:
        report = run_ingest(store, source_dir, pipeline, adapter=vec_adapter, parent=parent,
                            provider=provider)
    finally:
        vec_adapter.close()

    m = report.manifest
    console.print(
        f"[bold]{m.snapshot_id}[/bold]  docs={m.corpus.doc_count} chunks={m.corpus.chunk_count}\n"
        f"embed_calls={report.embed_calls} new_chunks={report.new_chunks} "
        f"reused_chunks={report.reused_chunks}"
    )


@main.group()
def snapshot() -> None:
    """Inspect snapshots."""


@snapshot.command("list")
def snapshot_list() -> None:
    store = _store()
    snaps = store.list_snapshots()
    if not snaps:
        console.print("No snapshots.")
        return
    table = Table(title="Snapshots")
    table.add_column("Snapshot")
    table.add_column("Parent")
    table.add_column("Docs", justify="right")
    table.add_column("Chunks", justify="right")
    table.add_column("Pinned")
    pins = store.pinned_snapshots()
    for m in snaps:
        table.add_row(m.snapshot_id, m.parent_snapshot or "-",
                      str(m.corpus.doc_count), str(m.corpus.chunk_count),
                      "*" if m.snapshot_id in pins else "-")
    console.print(table)


@snapshot.command("show")
@click.argument("snap")
def snapshot_show(snap: str) -> None:
    store = _store()
    m = _resolve(store, snap)
    console.print_json(json.dumps(m.to_dict()))


@snapshot.command("pin")
@click.argument("snap")
def snapshot_pin(snap: str) -> None:
    """Pin a snapshot so `recall gc` never removes it."""
    store = _store()
    m = _resolve(store, snap)
    store.pin_snapshot(m.snapshot_id)
    console.print(f"Pinned [bold]{m.snapshot_id}[/bold].")


@snapshot.command("unpin")
@click.argument("snap")
def snapshot_unpin(snap: str) -> None:
    """Remove a pin (the snapshot becomes eligible for gc again)."""
    store = _store()
    m = _resolve(store, snap)
    store.unpin_snapshot(m.snapshot_id)
    console.print(f"Unpinned [bold]{m.snapshot_id}[/bold].")


@main.group()
def dataset() -> None:
    """Create and manage golden datasets."""


@dataset.command("generate")
@click.option("--n", default=100, help="Number of cases.")
@click.option("--seed", default=0)
@click.option("--name", default="golden")
@click.option("--snapshot", "snap", default="latest")
@click.option("--llm", "llm_spec", default=None,
              help="Generate questions with an LLM, e.g. 'openai' or 'openai:gpt-4o-mini' "
                   "(uses OPENAI_API_KEY; cost-gated). Default: offline heuristic, $0.")
@click.option("--yes", is_flag=True, help="Approve the LLM generation cost.")
@click.option("--max-cost", type=float, default=None, help="LLM cost budget in USD.")
def dataset_generate(n: int, seed: int, name: str, snap: str, llm_spec: str | None,
                     yes: bool, max_cost: float | None) -> None:
    store = _store()
    m = _resolve(store, snap)
    llm = None
    if llm_spec:
        from .llm import estimate_generation_cost, get_llm

        try:
            llm = get_llm(llm_spec)
        except ValueError as exc:
            raise click.ClickException(str(exc))
        records = RetrievalEngine(store, m).chunks()
        avg_tokens = (sum(len(r.text) // 4 for r in records) / len(records)) if records else 0.0
        est = estimate_generation_cost(llm.model, n, avg_tokens)
        _cost_gate(est, max_cost, yes, "LLM dataset generation")
    ds = ds_generate(store, m, n=n, seed=seed, name=name, llm=llm)
    store.save_dataset(ds)
    console.print(f"[bold]{ds.dataset_id}[/bold]: {len(ds.cases)} cases")
    strat = stratification_report(ds)
    table = Table(title="Stratification")
    table.add_column("Tag")
    table.add_column("Count", justify="right")
    for tag, count in strat.items():
        table.add_row(tag, str(count))
    console.print(table)


@dataset.command("import")
@click.argument("file")
@click.option("--name", default="imported")
def dataset_import(file: str, name: str) -> None:
    store = _store()
    ds = ds_import(Path(file), f"{name}-v1")
    store.save_dataset(ds)
    console.print(f"[bold]{ds.dataset_id}[/bold]: {len(ds.cases)} cases imported")


@dataset.command("mine")
@click.option("--from", "src", required=True, help="JSONL trace file.")
@click.option("--name", default="mined")
def dataset_mine(src: str, name: str) -> None:
    store = _store()
    ds = ds_mine(Path(src), f"{name}-v1")
    store.save_dataset(ds)
    console.print(f"[bold]{ds.dataset_id}[/bold]: {len(ds.cases)} cases mined")


@dataset.command("list")
def dataset_list() -> None:
    store = _store()
    names = store.list_datasets()
    if not names:
        console.print("No datasets.")
        return
    for name in names:
        console.print(name)


@dataset.command("show")
@click.argument("dataset_id")
def dataset_show(dataset_id: str) -> None:
    store = _store()
    try:
        ds = store.get_dataset(dataset_id)
    except KeyError:
        raise click.ClickException(f"dataset {dataset_id!r} not found")
    table = Table(title=ds.dataset_id)
    table.add_column("id")
    table.add_column("question")
    table.add_column("expected")
    table.add_column("tags")
    for case in ds.cases:
        table.add_row(case.id, case.question, ", ".join(case.expected_sources),
                      ", ".join(case.tags))
    console.print(table)


@dataset.command("curate")
@click.argument("dataset_id")
@click.option("--accept", default="", help="Comma-separated case ids to accept.")
@click.option("--reject", default="", help="Comma-separated case ids to reject.")
@click.option("--edit-file", "edit_file", default=None,
              help="JSONL of edits: one {\"id\": ..., \"question\"|\"expected_sources\"|\"tags\": ...} per line.")
def dataset_curate(dataset_id: str, accept: str, reject: str, edit_file: str | None) -> None:
    store = _store()
    try:
        ds = store.get_dataset(dataset_id)
    except KeyError:
        raise click.ClickException(f"dataset {dataset_id!r} not found")
    decisions: dict[str, str] = {}
    for cid in (c.strip() for c in accept.split(",") if c.strip()):
        decisions[cid] = "accept"
    for cid in (c.strip() for c in reject.split(",") if c.strip()):
        decisions[cid] = "reject"
    edits: dict[str, dict] = {}
    if edit_file:
        for line in Path(edit_file).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            try:
                case_id = rec.pop("id")
            except KeyError:
                raise click.ClickException(f"edit record missing 'id': {line!r}")
            edits[case_id] = rec
    try:
        curated = ds_curate(ds, decisions, edits=edits)
    except ValueError as exc:
        raise click.ClickException(str(exc))
    store.save_dataset(curated)
    console.print(f"[bold]{curated.dataset_id}[/bold]: {len(curated.cases)} cases after curation")


@main.command()
@click.argument("dataset_id", required=False)
@click.option("--snapshot", "snap", default="latest")
@click.option("--replay/--live", default=True)
@click.option("--gate", type=click.Choice(["statistical"]), default=None)
@click.option("--fail-if", "fail_if", default=None, help='Raw threshold, e.g. "recall@5<0.85".')
@click.option("-k", "k_values", default="1,5,10")
@click.option("--config", "config_path", default="recall.yaml")
def eval(dataset_id: str | None, snap: str, replay: bool, gate: str | None,
         fail_if: str | None, k_values: str, config_path: str) -> None:
    """Evaluate a snapshot against a golden dataset."""
    cfg = _load_config(config_path)
    store = _store()
    m = _resolve(store, snap)
    ds = _get_dataset(store, dataset_id)
    ks = _parse_k(k_values)
    if gate == "statistical":
        # The statistical gate scores at the configured metric's k, and diff()
        # reads metrics[primary_metric] unconditionally, so the eval must compute
        # that k even when it is outside the default (1,5,10).
        ks = _with_gate_k(ks, cfg)

    mode = "replay" if replay else "live"
    adapter = None if replay else build_adapter(cfg, store)
    try:
        if adapter is not None:
            _warn_if_empty_collection(adapter, m, store)
        ev = evaluate(store, m, ds, adapter=adapter, k_values=ks, mode=mode)
        console.print(eval_table(ev))

        if fail_if:
            metric, op, threshold = parse_fail_if(fail_if)
            value = ev.aggregate.get(metric)
            if value is None:
                raise click.ClickException(f"metric {metric!r} not in eval aggregate")
            comparators = {
                "<": value < threshold, "<=": value <= threshold,
                ">": value > threshold, ">=": value >= threshold,
            }
            if comparators[op]:
                console.print(f"[red]Gate FAIL[/red]: {metric}={value:.4f} {op} {threshold}")
                sys.exit(1)
            console.print(f"[green]Gate PASS[/green]: {metric}={value:.4f}")
            return

        if gate == "statistical":
            calib_raw = store.get_json("calibration", m.snapshot_id)
            if calib_raw is None:
                raise click.ClickException(
                    "statistical gating requires a calibration record for this snapshot; "
                    "run `recall calibrate` first."
                )
            parent = m.parent_snapshot
            if parent is None:
                console.print("No parent snapshot; nothing to gate against.")
                return
            base = store.get_snapshot(parent)
            ev_base = evaluate(store, base, ds, adapter=None, k_values=ks, mode="replay")
            calib = CalibrationRecord.from_dict(calib_raw)
            dr = run_diff(store, base, m, ds, ev_base, ev, epsilon=calib.epsilon,
                          primary_metric=cfg.gate.get("primary_metric", "recall@5"))
            try:
                gres = evaluate_gate(dr, calib, mode="statistical",
                                     primary_metric=cfg.gate.get("primary_metric", "recall@5"),
                                     q=float(cfg.gate.get("q", 0.05)), dataset=ds)
            except GateNotCalibrated as exc:
                raise click.ClickException(str(exc))
            verdict = "PASS" if gres.passed else "FAIL"
            color = "green" if gres.passed else "red"
            console.print(f"[{color}]Gate {verdict}[/{color}] (statistical)")
            for reason in gres.reasons:
                console.print(f"- {reason}")
            if not gres.passed:
                sys.exit(1)
    finally:
        if adapter is not None:
            adapter.close()


@main.command()
@click.option("--snapshot", "snap", default="latest")
@click.option("--dataset", "dataset_id", default=None)
@click.option("--runs", default=3)
@click.option("--config", "config_path", default="recall.yaml")
def calibrate(snap: str, dataset_id: str | None, runs: int, config_path: str) -> None:
    """Calibrate the noise floor for a snapshot (required before statistical gating)."""
    cfg = _load_config(config_path)
    store = _store()
    m = _resolve(store, snap)
    ds = _get_dataset(store, dataset_id)
    adapter = build_adapter(cfg, store)
    try:
        _warn_if_empty_collection(adapter, m, store)
        record = run_calibrate(store, m, ds, adapter, n_runs=runs,
                               primary_metric=cfg.gate.get("primary_metric", "recall@5"))
    finally:
        adapter.close()
    console.print(f"Calibrated {m.snapshot_id}: epsilon={record.epsilon:.6f}, runs={record.n_runs}")
    table = Table(title="Per-metric std")
    table.add_column("Metric")
    table.add_column("Std", justify="right")
    for metric in sorted(record.per_metric_std):
        table.add_row(metric, f"{record.per_metric_std[metric]:.6f}")
    console.print(table)


@main.command()
@click.argument("snap_a")
@click.argument("snap_b")
@click.option("--dataset", "dataset_id", required=True)
@click.option("--attribute", type=click.Choice(["fast", "deep"]), default="fast")
@click.option("--arms", "arms_mode", type=click.Choice(["auto", "full"]), default="auto")
@click.option("--max-cost", type=float, default=None)
@click.option("--yes", is_flag=True)
@click.option("--format", "fmt", type=click.Choice(["table", "json", "md"]), default="table")
@click.option("--config", "config_path", default="recall.yaml")
def diff(snap_a: str, snap_b: str, dataset_id: str, attribute: str, arms_mode: str,
         max_cost: float | None, yes: bool, fmt: str, config_path: str) -> None:
    """Diff two snapshots with funnel (fast) or verified (deep) attribution."""
    cfg = _load_config(config_path)
    store = _store()
    ma = _resolve(store, snap_a)
    mb = _resolve(store, snap_b)
    ds = _get_dataset(store, dataset_id)

    ev_a = evaluate(store, ma, ds, adapter=None, mode="replay")
    ev_b = evaluate(store, mb, ds, adapter=None, mode="replay")
    dr = run_diff(store, ma, mb, ds, ev_a, ev_b)
    top_k = 5

    reports: dict = {}
    if attribute == "deep":
        reports = _deep_attribute(store, dr, ds, ma, mb, Path(cfg.source), arms_mode,
                                  max_cost, yes, top_k)

    if fmt == "json":
        console.print_json(json.dumps(render_diff_json(dr, reports or None)))
        return
    if fmt == "md":
        console.print(diff_summary_markdown(dr, reports or None, gate=None, calibration_ok=False))
        return

    console.print(diff_table(dr))
    console.print("Metric deltas:")
    for metric in sorted(dr.metric_deltas):
        console.print(f"  {metric}: {dr.metric_deltas[metric]:+.4f}")
    funnels, _ = _funnels(store, dr, ds, ma, mb, top_k)
    _print_funnels(store, dr, funnels, top_k)
    if attribute == "deep":
        _print_verified(reports)
    console.print(f"diff_id: {dr.diff_id}")


@main.command()
@click.argument("diff_id")
@click.option("--arms", "arms_mode", type=click.Choice(["auto", "full"]), default="auto")
@click.option("--max-cost", type=float, default=None)
@click.option("--yes", is_flag=True)
@click.option("--config", "config_path", default="recall.yaml")
def attribute(diff_id: str, arms_mode: str, max_cost: float | None, yes: bool,
              config_path: str) -> None:
    """Run deep counterfactual attribution on a stored diff."""
    cfg = _load_config(config_path)
    store = _store()
    stored = store.get_json("diff", diff_id)
    if stored is None:
        raise click.ClickException(f"diff {diff_id!r} not found")
    dr = DiffResult.from_dict(stored)
    ma = store.get_snapshot(dr.snapshot_a)
    mb = store.get_snapshot(dr.snapshot_b)
    ds = store.get_dataset(dr.dataset_id)
    reports = _deep_attribute(store, dr, ds, ma, mb, Path(cfg.source), arms_mode,
                              max_cost, yes, top_k=5)
    _print_verified(reports)


@main.command("compare-embeddings")
@click.option("--from", "from_spec", required=True)
@click.option("--to", "to_spec", required=True)
@click.option("--dataset", "dataset_id", required=True)
@click.option("--max-cost", type=float, default=None)
@click.option("--yes", is_flag=True)
@click.option("--config", "config_path", default="recall.yaml")
def compare_embeddings(from_spec: str, to_spec: str, dataset_id: str,
                       max_cost: float | None, yes: bool, config_path: str) -> None:
    """Compare two embedding models with per-tag deltas and a recommendation."""
    cfg = _load_config(config_path)
    store = _store()
    ds = _get_dataset(store, dataset_id)
    source_dir = Path(cfg.source)

    # Gate the COMBINED dual-ingest estimate once (Principle 5 / FR-10.1); gating
    # each arm independently against --max-cost would let total spend reach ~2x the
    # approved budget (#5). Then ingest each arm without re-gating.
    est_from = _estimate_embedding_cost(store, cfg, source_dir, from_spec)
    est_to = _estimate_embedding_cost(store, cfg, source_dir, to_spec)
    _cost_gate(est_from + est_to, max_cost, yes, "dual embedding ingest")
    m_from, cost_from = _ingest_embedding(store, cfg, source_dir, from_spec, None, True)
    m_to, cost_to = _ingest_embedding(store, cfg, source_dir, to_spec, None, True)
    ev_from = evaluate(store, m_from, ds, adapter=None, mode="replay")
    ev_to = evaluate(store, m_to, ds, adapter=None, mode="replay")

    tag_deltas, overall = _tag_deltas(ds, ev_from, ev_to)
    best_weight, best_metric = _best_hybrid_weight(store, m_to, ds)
    recommendation = {"hybrid_bm25_weight": best_weight, "primary_metric_at_best": best_metric}
    cost = {"usd": cost_from + cost_to}
    console.print(compare_report_markdown(tag_deltas, overall, recommendation, cost))


@main.command("compare-chunkers")
@click.option("--from", "from_spec", required=True, help="Chunker config JSON.")
@click.option("--to", "to_spec", required=True, help="Chunker config JSON.")
@click.option("--dataset", "dataset_id", required=True)
@click.option("--yes", is_flag=True)
@click.option("--config", "config_path", default="recall.yaml")
def compare_chunkers(from_spec: str, to_spec: str, dataset_id: str, yes: bool,
                     config_path: str) -> None:
    """Compare two chunkers with per-tag deltas and chunk-fate statistics."""
    cfg = _load_config(config_path)
    store = _store()
    ds = _get_dataset(store, dataset_id)
    source_dir = Path(cfg.source)

    m_from = _ingest_chunker(store, cfg, source_dir, json.loads(from_spec))
    m_to = _ingest_chunker(store, cfg, source_dir, json.loads(to_spec))
    ev_from = evaluate(store, m_from, ds, adapter=None, mode="replay")
    ev_to = evaluate(store, m_to, ds, adapter=None, mode="replay")

    tag_deltas, overall = _tag_deltas(ds, ev_from, ev_to)
    cost = {"usd": 0.0}
    console.print(compare_report_markdown(tag_deltas, overall, {}, cost))

    old_chunks = RetrievalEngine(store, m_from).chunks()
    new_chunks = RetrievalEngine(store, m_to).chunks()
    fates = align_chunks(old_chunks, new_chunks)
    counts: dict[str, int] = {}
    for fate in fates.values():
        counts[fate.cls] = counts.get(fate.cls, 0) + 1
    table = Table(title="Chunk-alignment fate")
    table.add_column("Class")
    table.add_column("Count", justify="right")
    for cls in sorted(counts):
        table.add_row(cls, str(counts[cls]))
    console.print(table)


@main.group()
def sweep() -> None:
    """Parameter sweeps over stored artifacts (zero re-embed)."""


@sweep.command("hybrid")
@click.option("--dataset", "dataset_id", required=True)
@click.option("--snapshot", "snap", default="latest")
@click.option("--grid", default=None, help="Comma-separated bm25 weights.")
def sweep_hybrid(dataset_id: str, snap: str, grid: str | None) -> None:
    """Sweep hybrid bm25_weight in replay and print the frontier."""
    store = _store()
    m = _resolve(store, snap)
    ds = _get_dataset(store, dataset_id)
    weights = ([float(x) for x in grid.split(",") if x.strip()] if grid
               else list(DEFAULT_HYBRID_GRID))

    rows: list[tuple[float, dict]] = []
    for w in weights:
        ephemeral = _reweighted_manifest(store, m, w)
        ev = evaluate(store, ephemeral, ds, adapter=None, mode="replay")
        rows.append((w, ev.aggregate))

    metrics = sorted(rows[0][1]) if rows else []
    table = Table(title="Hybrid bm25_weight sweep")
    table.add_column("weight", justify="right")
    for metric in metrics:
        table.add_column(metric, justify="right")
    for w, agg in rows:
        table.add_row(f"{w:.2f}", *[f"{agg[metric]:.4f}" for metric in metrics])
    console.print(table)

    console.print("Frontier (best weight per metric):")
    for metric in metrics:
        best_w, best_v = max(((w, agg[metric]) for w, agg in rows), key=lambda t: t[1])
        console.print(f"  {metric}: weight={best_w:.2f} -> {best_v:.4f}")


@main.command()
@click.option("--against", "against", required=True, help="Baseline (older-corpus) snapshot.")
@click.option("--dataset", "dataset_id", required=True)
@click.option("--snapshot", "snap", default="latest")
@click.option("--config", "config_path", default="recall.yaml")
def drift(against: str, dataset_id: str, snap: str, config_path: str) -> None:
    """Corpus-drift comparison: config held, corpus changed (diff + funnel)."""
    store = _store()
    base = _resolve(store, against)
    current = _resolve(store, snap)
    ds = _get_dataset(store, dataset_id)

    ev_base = evaluate(store, base, ds, adapter=None, mode="replay")
    ev_current = evaluate(store, current, ds, adapter=None, mode="replay")
    dr = run_diff(store, base, current, ds, ev_base, ev_current)
    if not dr.corpus_changed:
        console.print("[yellow]No corpus change between these snapshots.[/yellow]")
    console.print(diff_table(dr))
    funnels, engine_b = _funnels(store, dr, ds, base, current, top_k=5)
    _print_funnels(store, dr, funnels, top_k=5)

    chunk_texts = engine_b.chunk_texts()
    for qid in sorted(funnels):
        f = funnels[qid]
        after = dr.queries[qid].after
        for cid, _ in after.ranked_chunks[:3]:
            if cid in chunk_texts and cid != f.target_chunk_before:
                console.print(f"  [{qid}] distractor {cid}: {chunk_texts[cid][:80]!r}")
                break


@main.command()
@click.option("--config", "config_path", default="recall.yaml")
@click.option("--base", "base", default=None, help="Baseline snapshot (default: parent).")
@click.option("--dataset", "dataset_id", default=None)
@click.option("-o", "--out", "out", default="recall-report.md")
def ci(config_path: str, base: str | None, dataset_id: str | None, out: str) -> None:
    """Phase-1 CI gate: ingest, eval, diff, funnel; write recall-report.md."""
    cfg = _load_config(config_path)
    store = _store()
    source_dir = Path(cfg.source)
    pipeline = build_pipeline(cfg.pipeline)
    provider = _provider_for(pipeline)
    if provider.price_per_1k_tokens() > 0.0:
        est = _estimate_ingest_cost(store, source_dir, pipeline, provider)
        _cost_gate(est["usd"], None, True, "embedding")
    parent = None
    if store.list_snapshots():
        parent = store.resolve_snapshot("latest").snapshot_id
    vec_adapter = build_adapter(cfg, store)
    try:
        report = run_ingest(store, source_dir, pipeline, adapter=vec_adapter,
                            parent=parent, provider=provider)
    finally:
        vec_adapter.close()
    current = report.manifest
    ds = _get_dataset(store, dataset_id)

    base_ref = base or current.parent_snapshot
    if base_ref is None:
        Path(out).write_text(
            "## RecallOps CI\n\nNo base snapshot to diff against (first ingest).\n",
            encoding="utf-8",
        )
        console.print(f"No base snapshot; wrote {out}. Phase 2 deep attribution runs asynchronously.")
        return
    base_m = _resolve(store, base_ref)

    ks = _with_gate_k((1, 5, 10), cfg)  # include the gate metric's k for diff/gate
    ev_base = evaluate(store, base_m, ds, adapter=None, k_values=ks, mode="replay")
    ev_current = evaluate(store, current, ds, adapter=None, k_values=ks, mode="replay")
    dr = run_diff(store, base_m, current, ds, ev_base, ev_current)
    funnels, _ = _funnels(store, dr, ds, base_m, current, top_k=5)

    # Calibration lives on the BASELINE snapshot (the "before" side): the current
    # snapshot is created fresh this run and can never have been calibrated, so a
    # config-change PR would otherwise silently deactivate the statistical gate (#3).
    calib_raw = store.get_json("calibration", base_m.snapshot_id)
    gate = None
    if calib_raw is not None:
        calib = CalibrationRecord.from_dict(calib_raw)
        dr = run_diff(store, base_m, current, ds, ev_base, ev_current, epsilon=calib.epsilon,
                      primary_metric=cfg.gate.get("primary_metric", "recall@5"))
        try:
            gate = evaluate_gate(dr, calib, mode="statistical",
                                 primary_metric=cfg.gate.get("primary_metric", "recall@5"),
                                 q=float(cfg.gate.get("q", 0.05)), dataset=ds)
        except GateNotCalibrated:
            gate = None

    if gate is not None:
        store.save_json("gate", dr.diff_id, gate.to_dict())

    markdown = diff_summary_markdown(dr, None, gate, calibration_ok=calib_raw is not None)
    Path(out).write_text(markdown, encoding="utf-8")
    console.print(f"Wrote {out}.")
    console.print(
        "Phase 1 complete (eval + diff + funnel). Phase 2 deep attribution "
        f"runs asynchronously: `recall attribute {dr.diff_id} --arms deep`."
    )
    if gate is not None and not gate.passed:
        console.print("[red]Gate FAIL[/red]")
        sys.exit(1)


@main.command()
@click.option("--diff", "diff_id", required=True)
@click.option("--format", "fmt", type=click.Choice(["md", "html", "json"]), default="md")
@click.option("-o", "--out", "out", default=None)
def report(diff_id: str, fmt: str, out: str | None) -> None:
    """Re-render a stored diff (no re-run)."""
    store = _store()
    stored = store.get_json("diff", diff_id)
    if stored is None:
        raise click.ClickException(f"diff {diff_id!r} not found")
    dr = DiffResult.from_dict(stored)
    attr_raw = store.get_json("attribution", diff_id) or {}
    from .models import AttributionReport

    attributions = {qid: AttributionReport.from_dict(a) for qid, a in attr_raw.items()}

    gate_raw = store.get_json("gate", diff_id)
    gate = GateResult.from_dict(gate_raw) if gate_raw else None
    calibration_ok = store.get_json("calibration", dr.snapshot_a) is not None

    if fmt == "json":
        rendered = json.dumps(render_diff_json(dr, attributions or None), indent=2)
    elif fmt == "html":
        rendered = diff_report_html(dr, attributions or None, gate=gate)
    else:
        rendered = diff_summary_markdown(dr, attributions or None, gate=gate,
                                         calibration_ok=calibration_ok)

    if out:
        Path(out).write_text(rendered, encoding="utf-8")
        console.print(f"Wrote {out}.")
    else:
        click.echo(rendered)


@main.command()
@click.option("--keep", default=5, help="Snapshots to keep.")
def gc(keep: int) -> None:
    """Garbage-collect old snapshots' artifacts."""
    store = _store()
    pinned = store.pinned_snapshots()
    result = store.gc(keep_last=keep, pinned=pinned)
    console.print(
        f"Removed {result['removed_chunksets']} chunkset(s), "
        f"{result['removed_emb_files']} embedding file(s); "
        f"{len(pinned)} pinned snapshot(s) kept."
    )


@main.command()
@click.option("--dataset", "dataset_id", default=None)
@click.option("--reruns", default=10, help="No-change re-runs for the real noise floor.")
@click.option("--seed", default=0)
@click.option("--yes", is_flag=True, help="Approve embedding cost without prompting.")
@click.option("--max-cost", type=float, default=None)
@click.option("--out", "-o", default="phase0-report.json")
@click.option("--config", "config_path", default="recall.yaml")
def phase0(dataset_id: str | None, reruns: int, seed: int, yes: bool,
           max_cost: float | None, out: str, config_path: str) -> None:
    """Run the §13 Phase-0 go/no-go against the REAL serving stack (adapter + provider).

    Ingests the configured pipeline with write-through to the real vector DB,
    measures the ANN effect (exact vs live) and the real noise floor, runs
    known-cause config changes for attribution quality, and writes a provenance-
    stamped report. Uses the configured embedding provider (OpenAI when
    OPENAI_API_KEY is set; cost-gated), otherwise the offline local embedder.
    """
    import json as _json

    from .phase0 import default_changes, estimate_cost, run_phase0

    cfg = _load_config(config_path)
    store = _store()
    ds = _get_dataset(store, dataset_id)
    source_dir = Path(cfg.source)
    adapter = build_adapter(cfg, store)
    try:
        provider = _provider_for(build_pipeline(_pipeline_config(cfg)))
        real_embeddings = provider.price_per_1k_tokens() > 0.0

        cost = 0.0
        if real_embeddings:
            # baseline ingest + one re-ingest per known-cause change
            cost = estimate_cost(provider, source_dir) * (1 + len(default_changes()))
            _cost_gate(cost, max_cost, yes, "phase-0 real-embedding ingest")

        report = run_phase0(store, source_dir, ds, adapter,
                            base_config=_pipeline_config(cfg),
                            provider=provider if real_embeddings else None,
                            noise_reruns=reruns, seed=seed, cost_usd=cost)
    finally:
        adapter.close()

    g = report.gates
    table = Table(title=f"Phase-0 validation (§13), {report.provenance['provider']} + "
                        f"{report.provenance['adapter']}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_column("Gate", justify="right")
    checks = [("coverage", 0.70, ">="), ("fidelity", 0.995, ">="),
              ("noise_floor", 0.02, "<="), ("stage_accuracy", 0.80, ">=")]
    for key, gate, cmp in checks:
        val = g[key]
        ok = (val >= gate) if cmp == ">=" else (val <= gate)
        table.add_row(key, f"{val:.3f}", f"[{'green' if ok else 'red'}]{cmp}{gate}[/]")
    table.add_row("narrative_violations", str(g["narrative_violations"]),
                  f"[{'green' if g['narrative_violations'] == 0 else 'red'}]==0[/]")
    console.print(table)
    console.print(f"ANN effect: exact recall@1={report.ann_effect['exact_recall@1']:.3f} "
                  f"vs live={report.ann_effect['live_recall@1']:.3f} "
                  f"(divergence {report.ann_effect['recall@1_divergence']:+.3f}); "
                  f"noise-floor epsilon={report.noise_floor['epsilon']:.4f}, "
                  f"flake-rate={report.noise_floor['rate']:.4f}")
    verdict = "GO" if report.passed else "NO-GO"
    console.print(f"[{'green' if report.passed else 'red'}]Phase-0 verdict: {verdict}[/]")
    for c in report.caveats:
        console.print(f"[yellow]caveat:[/yellow] {c}")
    Path(out).write_text(_json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    console.print(f"Wrote {out}.")


@main.command()
@click.option("--seed", default=0)
def scorecard(seed: int) -> None:
    """Run the attribution-engine self-test (§13) on the example corpus."""
    try:
        from .scorecard import run_scorecard
    except ImportError:
        raise click.ClickException(
            "scorecard module not available in this build (recallops.scorecard)."
        )
    result = run_scorecard(Path("."), seed=seed)
    table = Table(title="Engine scorecard (§13)")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("coverage", f"{result.coverage:.3f}")
    table.add_row("fidelity", f"{result.fidelity:.3f}")
    table.add_row("noise_floor", f"{result.noise_floor:.3f}")
    table.add_row("stage_accuracy", f"{result.stage_accuracy:.3f}")
    table.add_row("narrative_violations", str(result.narrative_violations))
    console.print(table)


# -- compare / sweep internals ------------------------------------------------


def _estimate_embedding_cost(store: ProjectStore, cfg: ProjectConfig, source_dir: Path,
                             spec: str) -> float:
    pipeline = build_pipeline(_pipeline_config(cfg, embedding=spec))
    provider = _provider_for(pipeline)
    if provider.price_per_1k_tokens() <= 0.0:
        return 0.0
    return _estimate_ingest_cost(store, source_dir, pipeline, provider)["usd"]


def _ingest_embedding(store: ProjectStore, cfg: ProjectConfig, source_dir: Path,
                      spec: str, max_cost: float | None, yes: bool) -> tuple[SnapshotManifest, float]:
    pc = _pipeline_config(cfg, embedding=spec)
    pipeline = build_pipeline(pc)
    provider = _provider_for(pipeline)
    usd = 0.0
    if provider.price_per_1k_tokens() > 0.0:
        est = _estimate_ingest_cost(store, source_dir, pipeline, provider)
        usd = est["usd"]
        _cost_gate(usd, max_cost, yes, "embedding")
    report = run_ingest(store, source_dir, pipeline, adapter=None, provider=provider)
    return report.manifest, usd


def _ingest_chunker(store: ProjectStore, cfg: ProjectConfig, source_dir: Path,
                    chunker_cfg: dict) -> SnapshotManifest:
    pc = copy.deepcopy(cfg.pipeline)
    pc["chunker"] = chunker_cfg
    pipeline = build_pipeline(pc)
    provider = _provider_for(pipeline)
    report = run_ingest(store, source_dir, pipeline, adapter=None, provider=provider)
    return report.manifest


def _tag_deltas(ds: GoldenDataset, ev_from, ev_to) -> tuple[dict, dict]:
    tags_by_id = {c.id: list(c.tags) for c in ds.cases}
    by_tag: dict[str, dict[str, list[float]]] = {}
    overall_acc: dict[str, list[float]] = {}
    for qid, qe_to in ev_to.per_query.items():
        qe_from = ev_from.per_query.get(qid)
        if qe_from is None:
            continue
        for metric, value in qe_to.metrics.items():
            delta = value - qe_from.metrics.get(metric, 0.0)
            overall_acc.setdefault(metric, []).append(delta)
            for tag in tags_by_id.get(qid, []):
                by_tag.setdefault(tag, {}).setdefault(metric, []).append(delta)
    tag_deltas = {
        tag: {metric: sum(vals) / len(vals) for metric, vals in metrics.items()}
        for tag, metrics in by_tag.items()
    }
    overall = {metric: sum(vals) / len(vals) for metric, vals in overall_acc.items()}
    return tag_deltas, overall


def _reweighted_manifest(store: ProjectStore, manifest: SnapshotManifest,
                         bm25_weight: float) -> SnapshotManifest:
    retrieve = manifest.pipeline.stage("retrieve")
    top_k = int(retrieve.params.get("top_k", 10)) if retrieve is not None else 10
    new_stage = StageSpec(
        id="retrieve", tool="recall.retrieve", version="1",
        params={"top_k": top_k,
                "hybrid": {"sparse": "bm25", "fusion": "weighted", "bm25_weight": bm25_weight}},
        inputs=("index",),
    )
    new_pipeline = manifest.pipeline.replace("retrieve", new_stage)
    return SnapshotManifest.build(new_pipeline, manifest.corpus, manifest.artifacts,
                                  parent=manifest.snapshot_id)


def _best_hybrid_weight(store: ProjectStore, manifest: SnapshotManifest,
                        ds: GoldenDataset, metric: str = "recall@5") -> tuple[float, float]:
    best_w, best_v = 0.0, -1.0
    for w in DEFAULT_HYBRID_GRID:
        ephemeral = _reweighted_manifest(store, manifest, w)
        ev = evaluate(store, ephemeral, ds, adapter=None, mode="replay")
        value = ev.aggregate.get(metric, 0.0)
        if value > best_v:
            best_w, best_v = w, value
    return best_w, best_v


if __name__ == "__main__":
    main()
