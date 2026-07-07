# CLI reference

The `recall` command-line interface wires the RecallOps engine end-to-end over a
`ProjectStore` (rooted at the current directory) and a `recall.yaml`. This page
documents every command, grouped by workflow, with its purpose and key flags.

Two conventions apply throughout:

- **Snapshot references** accept `latest`, a full `snap_…` id, or any unambiguous
  id prefix.
- **Cost gating** (Principle 5): the default local provider is $0 and
  auto-approves. Any provider-billed operation prints a cost estimate and
  requires `--yes` or a `--max-cost N` budget before it will spend.

Global: `recall --version` prints the installed `recallops` version; every
command takes `--help`.

---

## Setup

### `recall init`

Writes `recall.yaml` (the project + pipeline definition) and creates the
`.recall/` provenance store. Also persists the project namespace so serving
collections stay per-project when several projects share one vector DB.

| Flag | Default | Purpose |
|---|---|---|
| `--adapter [local\|pgvector]` | `local` | Vector adapter to configure. |
| `--source` | `docs` | Documents directory. |
| `--name` | current dir name | Project name (namespaces serving collections). |
| `--dsn` | none | pgvector DSN (pgvector adapter only). |
| `--force` | off | Overwrite an existing `recall.yaml`. |

### `recall ingest [PATH]`

Ingests a corpus into a new **immutable, content-addressed snapshot**
(parse → chunk → embed → index). Reuses any chunks/embeddings already in the
store, so re-ingesting an identical pipeline costs zero embedding calls. `PATH`
overrides the configured source directory. Flags override individual pipeline
stages for one run without editing `recall.yaml`, handy for A/B ingests.

| Flag | Default | Purpose |
|---|---|---|
| `--chunker` | config | Override the chunker tool. |
| `--chunk-params` | config | Chunker params as a JSON object. |
| `--embedding` | config | Override the embedding spec `provider:model:dims`. |
| `--adapter` | config | Override the index adapter. |
| `--config` | `recall.yaml` | Project config path. |
| `--yes` / `--max-cost` | none | Approve / budget a non-zero embedding cost. |

### `recall snapshot`

Inspects committed snapshots.

- **`recall snapshot list`**: a table of snapshots (id, parent, doc/chunk counts).
- **`recall snapshot show <snap>`**: the resolved snapshot's full manifest as JSON.

### `recall dataset`

Creates and manages golden datasets (the query set evals run against).

- **`recall dataset generate`**: bootstrap a dataset offline from a snapshot's
  chunks (no LLM/network) and print a stratification report.
  Flags: `--n` (100), `--seed` (0), `--name` (`golden`), `--snapshot` (`latest`),
  `--yes` (approve LLM cost; unused offline).
- **`recall dataset import <file>`**: import a JSON/JSONL golden file.
  Flag: `--name` (`imported`).
- **`recall dataset mine --from <traces.jsonl>`**: mine cases from a production
  trace JSONL. Flag: `--name` (`mined`).
- **`recall dataset list`**: list stored dataset ids.
- **`recall dataset show <dataset_id>`**: a table of cases (id, question,
  expected sources, tags).
- **`recall dataset curate <dataset_id>`**: accept/reject cases into a curated
  copy. Flags: `--accept`, `--reject` (comma-separated case ids).

---

## Evaluate

### `recall eval [DATASET_ID]`

Evaluates a snapshot against a golden dataset and prints the aggregate metrics
(`recall@k`, `hit_rate@k`, `MRR`, `nDCG@k`). Replay mode scores exactly from
stored embeddings; live mode serves through the configured adapter. Optionally
enforces a gate: `--fail-if` is a raw threshold (exits 1 when met), while
`--gate statistical` runs the never-flaky gate against the snapshot's parent and
requires a prior `recall calibrate`. If no `DATASET_ID` is given, the most recent
dataset is used.

| Flag | Default | Purpose |
|---|---|---|
| `--snapshot` | `latest` | Snapshot to evaluate. |
| `--replay / --live` | `--replay` | Score from stored embeddings vs the live adapter. |
| `--gate [statistical]` | off | Run the statistical release gate (needs calibration). |
| `--fail-if` | none | Raw threshold, e.g. `"recall@5<0.85"` (exits 1 if met). |
| `-k` | `1,5,10` | Comma-separated k-values to score. |
| `--config` | `recall.yaml` | Project config path. |

### `recall calibrate`

Measures a snapshot's serving **noise floor**: the prerequisite for statistical
gating. Re-runs an identical snapshot `--runs` times (rebuilding the index
between runs where the adapter supports it), derives the near-tie threshold
`epsilon` and per-metric std, and stores a `CalibrationRecord` for that snapshot.
On the built-in exact adapter every run is identical and `epsilon` collapses to 0.

| Flag | Default | Purpose |
|---|---|---|
| `--snapshot` | `latest` | Snapshot to calibrate. |
| `--dataset` | latest | Golden dataset. |
| `--runs` | `3` | No-change re-runs used to estimate the noise floor. |
| `--config` | `recall.yaml` | Project config path (supplies `gate.primary_metric`). |

---

## Diff and attribute

### `recall diff <snap_a> <snap_b>`

Diffs two snapshots and attributes what changed. `--attribute fast` (default)
runs the observational funnel: stage-wise ranks and the implicated factors per
regressed query. `--attribute deep` additionally runs the **counterfactual
ablation**: it materializes revert arms from the store, applies the confirmation
rule, and prints **verified causes** (each citing the arm that confirmed it) plus
a deterministic narrative. The diff is persisted under its `diff_id` for later
re-rendering.

| Flag | Default | Purpose |
|---|---|---|
| `--dataset` | *required* | Golden dataset to diff over. |
| `--attribute [fast\|deep]` | `fast` | Funnel only vs full counterfactual attribution. |
| `--arms [auto\|full]` | `auto` | Arm-lattice strategy for deep attribution. |
| `--max-cost` / `--yes` | none | Budget / approve arm embedding cost (deep). |
| `--format [table\|json\|md]` | `table` | Output format. |
| `--config` | `recall.yaml` | Project config path. |

### `recall attribute <diff_id>`

Runs (or resumes) **deep counterfactual attribution** on an already-stored diff,
the async Phase-2 step after `recall ci`. Looks up the diff, materializes the
revert arms, applies the confirmation rule, and prints the verified causes. Arm
runs are checkpointed, so an interrupted job resumes without repeating work.

| Flag | Default | Purpose |
|---|---|---|
| `--arms [auto\|full]` | `auto` | Arm-lattice strategy. |
| `--max-cost` / `--yes` | none | Budget / approve arm embedding cost. |
| `--config` | `recall.yaml` | Project config path. |

### `recall drift`

Corpus-drift comparison: the pipeline config is held constant while the corpus
changed. Diffs a baseline (`--against`) snapshot to the current one, runs the
funnel over regressed queries, and surfaces the newly-ingested distractor chunks
that now outrank each target. Warns if the two snapshots share a corpus.

| Flag | Default | Purpose |
|---|---|---|
| `--against` | *required* | Baseline (older-corpus) snapshot. |
| `--dataset` | *required* | Golden dataset. |
| `--snapshot` | `latest` | Current snapshot to compare. |
| `--config` | `recall.yaml` | Project config path. |

### `recall report`

Re-renders a **stored** diff (and any deep attribution attached to it) to
Markdown, HTML, or JSON with **no re-run**: fully reproducible from the persisted
artifacts.

| Flag | Default | Purpose |
|---|---|---|
| `--diff` | *required* | The `diff_id` to render. |
| `--format [md\|html\|json]` | `md` | Output format. |
| `-o, --out` | stdout | Write to a file instead of stdout. |

---

## Migration and tuning

### `recall compare-embeddings`

De-risks an embedding-model migration. Dual-ingests the same corpus under two
embedding specs, evals both, and prints a per-tag metric-delta report plus a
recommended hybrid BM25 weight. Because an embedding swap is a black box, this is
*characterization* (per-tag deltas), not a verified mechanism. The **combined**
dual-ingest cost is gated once, so total spend cannot exceed the approved budget.

| Flag | Default | Purpose |
|---|---|---|
| `--from` / `--to` | *required* | The two embedding specs `provider:model:dims`. |
| `--dataset` | *required* | Golden dataset. |
| `--max-cost` / `--yes` | none | Budget / approve the combined embedding cost. |
| `--config` | `recall.yaml` | Project config path. |

### `recall compare-chunkers`

Compares two chunker configurations on the same corpus: dual-ingest, eval both,
and print a per-tag metric-delta report plus **chunk-fate statistics** (how many
chunks were intact / split / merged / boundary-shifted / dropped between the two
chunkers). Local embeddings make this a $0 comparison.

| Flag | Default | Purpose |
|---|---|---|
| `--from` / `--to` | *required* | The two chunker configs as JSON. |
| `--dataset` | *required* | Golden dataset. |
| `--yes` | none | Approve a non-zero embedding cost (if applicable). |
| `--config` | `recall.yaml` | Project config path. |

### `recall sweep hybrid`

Sweeps the hybrid `bm25_weight` over a grid entirely in **replay** (zero
re-embedding, since only fusion changes) and prints the per-weight metric table
plus the best weight per metric (the frontier).

| Flag | Default | Purpose |
|---|---|---|
| `--dataset` | *required* | Golden dataset. |
| `--snapshot` | `latest` | Snapshot to sweep over. |
| `--grid` | 0.0…1.0 by 0.1 | Comma-separated BM25 weights to try. |

---

## Gating and CI

### `recall ci`

The Phase-1 CI gate (designed to run in under 5 minutes, no counterfactual runs).
Ingests the configured pipeline, evals the current snapshot and its baseline,
diffs them, runs the funnel, and writes a Markdown report for a PR comment. When
the **baseline** snapshot has a calibration record it applies the statistical
gate and exits 1 on a significant regression. It prints the follow-up command for
the async Phase-2 deep attribution (`recall attribute <diff_id>`).

| Flag | Default | Purpose |
|---|---|---|
| `--config` | `recall.yaml` | Project config path. |
| `--base` | parent snapshot | Baseline snapshot to diff against. |
| `--dataset` | latest | Golden dataset. |
| `-o, --out` | `recall-report.md` | Report output path. |

See [`examples/github-action/recall-ci.yml`](../examples/github-action/recall-ci.yml)
for a ready-to-adapt two-phase workflow.

### `recall gc`

Garbage-collects the artifacts (chunk sets and embedding files) of old snapshots,
keeping the most recent `--keep` snapshots (plus any pinned ones). This is the
main lever for bounding store size: see [`sizing.md`](sizing.md).

| Flag | Default | Purpose |
|---|---|---|
| `--keep` | `5` | Number of most-recent snapshots to retain. |

---

## Validation

### `recall scorecard`

Runs the attribution-engine **self-test** on the bundled example corpus: four
known-cause scenarios (chunker, fusion, embedding, corpus drift) plus a noise
floor, measured against the §13 gates (coverage, fidelity, noise floor, stage
accuracy, narrative faithfulness). This validates the engine offline; it does not
use real serving infrastructure.

| Flag | Default | Purpose |
|---|---|---|
| `--seed` | `0` | Seed for the synthetic scenarios. |

### `recall phase0`

Runs the §13 real-project **go/no-go** against the *real serving stack* (the
configured adapter and, when `OPENAI_API_KEY` is set, real embeddings). Ingests
with write-through to the vector DB, measures the ANN effect (exact shadow vs live
ranking) and the real noise floor, runs known-cause changes for attribution
quality, and writes a provenance-stamped report with a GO / NO-GO verdict. With a
billing provider it prints a cost estimate you must approve. See the full runbook
in [`phase0-validation.md`](phase0-validation.md).

| Flag | Default | Purpose |
|---|---|---|
| `--dataset` | latest | Golden dataset. |
| `--reruns` | `10` | No-change re-runs for the real noise floor. |
| `--seed` | `0` | Seed. |
| `--yes` / `--max-cost` | none | Approve / budget the real-embedding ingest cost. |
| `-o, --out` | `phase0-report.json` | Report output path. |
| `--config` | `recall.yaml` | Project config path. |
