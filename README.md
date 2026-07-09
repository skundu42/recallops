# RecallOps

**Retrieval regression testing with _verified_ root-cause attribution for RAG pipelines.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11 to 3.13](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/skundu42/recallops/actions/workflows/ci.yml/badge.svg)](https://github.com/skundu42/recallops/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/recallops.svg)](https://pypi.org/project/recallops/)

Teams change embedding models, chunkers, parsers, hybrid-search weights, rerankers,
and source documents constantly. Any of these can silently degrade retrieval, and
today no tool answers *"did retrieval get better or worse, and **exactly why**?"* with
evidence. RecallOps does, and it never says "why" unless a counterfactual re-run
proves it.

RecallOps lives in the **ingestion path**, which is exactly where root-cause
attribution has to happen. It is **not** a vector database, an observability/tracing
tool, or an eval-metrics library; it integrates with those.

---

## Why RecallOps

The wedge is **verified counterfactual attribution**. When retrieval regresses,
observational funnel analysis can *implicate* a factor, but a correlation is a guess.
RecallOps only reports a cause as **verified** when it has executed a counterfactual
"arm" that reverts *only* that factor (everything else held at the new config) and
observed the failing query recover. Everything it cannot confirm this way is emitted
as an explicitly-labeled **hypothesis**, never merged into the verified set.

That invariant is only tractable because RecallOps sits in the ingestion path and
versions the whole pipeline as content-addressed provenance. Because it stores
embeddings (not just their hashes), it can replay retrieval and run ablation arms with
**zero re-embedding**, and it shadow-scores exactly, so it can tell a real config
regression apart from ANN index noise, which a serving-side tool cannot.

- **Versions the pipeline**: content-addressed manifests of docs, chunks, embeddings, and config.
- **Tests retrieval in CI**: golden-set evals with a statistically gated pass/fail, designed never to flake.
- **Attributes regressions to causes**: observational funnel analysis + memoized counterfactual ablation, under a confirmation rule: *no explanation is emitted unless the arm reverting that factor recovers the query.*
- **De-risks migrations**: first-class `compare-embeddings` / `compare-chunkers` workflows at a printed, pre-approved cost.

## Features

- **Verified root-cause attribution** for chunker, parser, fusion-weight, reranker, and corpus-drift changes.
- **Content-addressed snapshots**: identical inputs produce byte-identical manifests and reproducible ids.
- **Offline by default at $0**: local hash embeddings + a built-in exact-KNN adapter; no keys, no server.
- **pgvector adapter** (opt-in) with exact shadow scoring over the live ANN index.
- **Never-flaky CI gate**: per-snapshot noise calibration, near-tie exclusion, bootstrap CIs, McNemar's test, Benjamini-Hochberg FDR correction.
- **Zero-re-embed replay & sweeps**: retune fusion weights or ablate factors without paying to re-embed.
- **Cost-gated by design**: any provider-billed operation prints an estimate and requires `--yes` or `--max-cost`.
- **Bring-your-own-pipeline SDK** (`Recorder`): record the same provenance from a bespoke ingestion pipeline.
- Ships `py.typed`; typed public API; Python 3.11 to 3.13.

## Install

```bash
pip install recallops
```

> **Name note:** the distribution is `recallops` (the name `recallkit` was already taken
> on PyPI). The **import** name is also `recallops` and the console script is `recall`.
> Pin a version for reproducible installs, e.g. `pip install "recallops==0.1.0"`.

The default stack (local hash embeddings + built-in exact-KNN adapter) runs fully
**offline at $0**, no API keys, no server. For the pgvector adapter:

```bash
pip install 'recallops[pg]'      # adds the psycopg-based pgvector adapter
```

Develop against a checkout:

```bash
pip install -e ".[dev]"          # or: uv pip install -e ".[dev]"
pytest                           # full suite; pgvector live tests auto-skip unless RECALL_PG_DSN is set
ruff check
```

---

## 5-minute quickstart (catch a regression)

The commands below use the bundled example corpus ([`examples/corpus/`](examples/corpus/),
12 markdown docs). Clone the repo or copy it into a `docs/` folder to reproduce the
exact numbers. All ids are **content-addressed**, so you will see the same snapshot ids.
The whole flow runs on the local provider at **$0** and needs no network. A scripted
version is in [`examples/quickstart.sh`](examples/quickstart.sh).

### 1. Init and ingest a baseline

```bash
recall init --source docs
recall ingest
```

```
snap_5cdbd0dc0e65ab00  docs=12 chunks=72
embed_calls=72 new_chunks=72 reused_chunks=0
```

### 2. Generate a golden dataset and eval: green

```bash
recall dataset generate --n 50 --seed 0 --name golden
recall eval golden --snapshot latest
```

```
┏━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Metric      ┃  Value ┃
┡━━━━━━━━━━━━━╇━━━━━━━━┩
│ recall@5    │ 1.0000 │
│ recall@1    │ 0.8600 │
│ mrr         │ 0.9217 │
│ ndcg@5      │ 0.9417 │
└─────────────┴────────┘
```

### 3. Change the chunker and re-ingest

Swap the default `markdown_heading(800, 120)` for an aggressive fixed-token chunker
that fractures sections:

```bash
recall ingest --chunker recall.chunkers.fixed_token \
              --chunk-params '{"max_tokens": 15, "overlap": 0}'
```

```
snap_9c97b0630a2ea8e2  docs=12 chunks=176
embed_calls=176 new_chunks=176 reused_chunks=0
```

### 4. Eval: red, and the gate fails

```bash
recall eval golden --snapshot latest --fail-if "recall@5<0.99"
```

```
│ recall@5 │ 0.9800 │   ← was 1.0000
│ recall@1 │ 0.8200 │   ← was 0.8600
Gate FAIL: recall@5=0.9800 < 0.99
```

The command exits **1**: a red CI check. (`--fail-if` is the raw-threshold mode;
for CI, prefer the statistical gate, `--gate statistical`, which is designed never to
cry wolf.)

### 5. Diff with verified attribution: *why* it broke

```bash
recall diff snap_5cdbd0dc0e65ab00 snap_9c97b0630a2ea8e2 \
       --dataset golden --attribute deep
```

```
        Funnel attribution (regressed queries)
┃ Query ┃ Fate  ┃ Failing stage ┃ Implicated factors ┃
│ q_036 │ split │ fused         │ chunk              │

verified q_036: chunk -> rank 1 (arm_df39b727e7faa029)
  The target section was split across 4 chunks by the chunker; each fragment
  scores lower on both dense and BM25 (heading text no longer in-chunk).
  Reverting the chunker alone restores rank 1 (verified: arm_df39b727e7faa029).
```

That last line is the whole point: RecallOps **executed a counterfactual arm** that
reverts only the chunker (everything else held at the new config) and confirmed the
target returns to rank 1. The cause is `verified`, not guessed. `--format json` gives
the full auditable report:

```json
{
  "query_id": "q_036",
  "classification": "regressed",
  "funnel": {
    "target_chunk_before": "ch_dbbc9abe7b3a2ead",
    "target_in_index_after": true,
    "dense":  {"rank_before": 21, "rank_after": null, "shadow_exact_rank_after": 66},
    "sparse": {"rank_before": 1,  "rank_after": 1},
    "fused":  {"rank_before": 1,  "rank_after": 6},
    "rerank": {"in_candidates_after": true},
    "ann_divergence": false
  },
  "chunk_fate": {"class": "split", "alignment_score": 0.989,
                 "old_chunk": "ch_dbbc9abe7b3a2ead",
                 "new_chunks": ["ch_cb2357c2...", "ch_55a45be9...", "...", "..."]},
  "verified_causes": [
    {"factor": "chunk", "arm_id": "arm_df39b727e7faa029",
     "recovered_rank": 1, "status": "verified"}
  ],
  "hypotheses": [],
  "narrative": "The target section was split across 4 chunks by the chunker; ..."
}
```

Fix the chunker overlap, re-ingest, and the gate goes green again.

> Re-running an identical pipeline performs **zero embedding calls**; a fusion-weight
> change re-uses every chunk and embedding. Try `recall sweep hybrid --dataset golden`
> to tune the BM25 weight with no re-embedding.

---

## How verification works

The product principle, **no explanation without evidence**, is enforced by a
confirmation rule. Every rendered claim is exactly one of two kinds:

| | **Verified cause** | **Hypothesis** |
|---|---|---|
| Emitted when | an arm reverting the factor (others held at the new config) recovers the query to within its original rank ±1 | funnel evidence implicates a factor but **no** arm recovered the query |
| Label | `status: verified`, cites the confirming `arm_id` | `status: unverified`, cites the co-occurring evidence, e.g. *"sparse rank drop co-occurred"* |
| Presented as | fact | explicitly a guess |
| Guarantee | **every cause links to its counterfactual arm** | never merged into the verified set |

**Embedding-model migration is a deliberate special case.** There is no mechanistic
story for a black-box model swap, so RecallOps does not fabricate one; it emits
**characterization** instead: per-tag metric deltas (e.g. exact-term −7%, paraphrase
+11%), which is exactly the migration artifact `recall compare-embeddings` produces.
This is characterization, **not** a verified mechanism; see
[Project status & honest caveats](#project-status--honest-caveats).

Statistical gating is **never-flaky by design** and needs a one-time calibration per
snapshot: `recall calibrate` re-runs an identical snapshot N times (rebuilding the
index where the adapter supports it), derives a near-tie threshold ε, and excludes
near-ties from every regression signal. Stable-query deltas are gated by a bootstrap
95% CI and stable-query flips by McNemar's exact test, corrected across tags with
Benjamini-Hochberg FDR. Because near-ties feed no signal, serving noise never reddens
the gate; the trade-off is that a regression visible only as near-ties is below the
calibrated noise floor and is reported (with the excluded count) rather than gated.

```bash
recall calibrate --snapshot latest --dataset golden
recall eval golden --snapshot latest --gate statistical
```

---

## CLI at a glance

One console script, `recall`, with the following commands (`recall <cmd> --help` for
options):

| Command | What it does |
|---|---|
| `init` | Write `recall.yaml` and create the `.recall` provenance store. |
| `ingest` | Ingest a corpus into a new immutable, content-addressed snapshot. |
| `snapshot` | Inspect snapshots (`list`, `show`). |
| `dataset` | Create and manage golden datasets (`generate`, `curate`, `import`, `mine`, `list`, `show`). |
| `eval` | Evaluate a snapshot against a golden dataset (raw threshold or statistical gate). |
| `calibrate` | Calibrate a snapshot's noise floor, required before statistical gating. |
| `diff` | Diff two snapshots with funnel (fast) or verified (deep) attribution. |
| `attribute` | Run deep counterfactual attribution on a stored diff (the async Phase-2 pass). |
| `compare-embeddings` | Compare two embedding models with per-tag deltas and a recommendation. |
| `compare-chunkers` | Compare two chunkers with per-tag deltas and chunk-fate statistics. |
| `sweep` | Parameter sweeps over stored artifacts with zero re-embedding (`hybrid`). |
| `drift` | Corpus-drift comparison: config held constant, corpus changed. |
| `ci` | Phase-1 CI gate: ingest, eval, diff, funnel; writes `recall-report.md`. |
| `report` | Re-render a stored diff (no re-run) to md / html / json. |
| `gc` | Garbage-collect old snapshots' artifacts (`--keep N`). |
| `scorecard` | Run the attribution-engine self-test on the example corpus. |
| `phase0` | Run the Phase-0 go/no-go against the real serving stack (adapter + provider). |

---

## Where it fits, and what it is not

RecallOps is a **retrieval regression-testing and attribution** tool for the RAG
ingestion path. It complements the tools around it rather than replacing them:

| Category | What those tools do | How RecallOps relates |
|---|---|---|
| **Vector databases** (pgvector, and similar) | Store and serve nearest-neighbor search at query time. | RecallOps is *not* one. It reads/writes through an adapter (built-in exact-KNN, or pgvector) and shadow-scores exactly to separate real regressions from ANN noise. |
| **Observability / tracing** (LLM tracing, request logs) | Watch production traffic and latency at serving time. | RecallOps is *not* one. It works in the ingestion/CI path on golden sets, before a change ships, and can *prove* root cause; tracing observes, it does not run counterfactuals. |
| **Eval-metrics libraries** (Ragas, DeepEval, and similar) | Score generation quality (faithfulness, answer relevance, …). | RecallOps *integrates* with them. It owns retrieval metrics and the CI gate; generation-quality metrics come from those libraries and are never part of the default gate. |

The distinctive job, **verified counterfactual root-cause attribution**, requires
owning the versioned ingestion pipeline, which none of the above do.

---

## Bring your own pipeline (SDK)

If you run a bespoke ingestion pipeline, record the same content-addressed provenance
with the `Recorder` API. SDK snapshots are byte-diffable against managed-mode snapshots
of the same corpus.

```python
import numpy as np
from recallops.recorder import Recorder

rec = Recorder(project="support-rag", store=".recall")

with rec.stage("parse", tool="my_parser", version="1"):
    doc_id = rec.log_document("billing/refunds.md", raw_bytes, parsed_text)

with rec.stage("chunk", tool="acme_chunker", version="3.2", params={"size": 800}):
    chunk_ids = rec.log_chunks(doc_id, [{"text": t, "span": (s, e)} for ...])

rec.log_embeddings(provider="openai", model="text-embedding-3-small",
                   chunk_ids=chunk_ids, vectors=vecs)   # np.ndarray, fp16-cast on write
rec.log_retrieval(query_id="q_014", stage="dense",
                  candidates=[("ch_a01", 0.812), ...])
snapshot_id = rec.commit()
```

---

## CI setup: the two-phase gate

Copy [`examples/github-action/recall-ci.yml`](examples/github-action/recall-ci.yml)
into your repo's `.github/workflows/` and fill in the clearly-marked `TODO(you)`
placeholders (baseline store, docs path, optional provider secret). It implements a
two-phase gate:

- **Phase 1, blocking (< 5 min):** `recall ci` runs eval + diff + funnel attribution,
  posts `recall-report.md` as a PR comment, and fails the check on a statistically
  gated regression. No re-embedding, no counterfactual runs.
- **Phase 2, async (non-blocking):** `recall attribute` runs the deep counterfactual
  pass and edits the same PR comment with verified causes. Deep attribution is never
  promised instantly.

RecallOps's own repository CI lives in
[`.github/workflows/ci.yml`](.github/workflows/ci.yml) (the badge above tracks it).

---

## Honest at scale

Attribution *fidelity* (not storage) is what degrades at scale, and degradation is
always **labeled, never silent**:

| Tier | Corpus size | Attribution behavior |
|---|---|---|
| **S** | ≤ 500k chunks | Full fidelity: complete arm lattice (k ≤ 3), exact Shapley, full-corpus re-embeds within the cost gate. |
| **M** | 500k to 5M | Full arms for zero-embed factors (fusion, rerank, retrieve); embedding/chunker arms use exact-KNN over the candidate union + a stratified background sample; **embedding-model swaps require a full, cost-gated re-embed** (sampling is unsound for model swaps and is never silently substituted). |
| **L** | > 5M | Factor-level attribution by default; embedding-swap verification only with explicit full re-embed approval; any sampled result labeled **"unverified at corpus scale"** in every output. |

Any provider-billed operation prints a cost estimate and requires `--yes` or a
`--max-cost` budget. The local provider is $0 and auto-approves.

Storage is fp16 columnar and grows as `chunks × dims × 2` bytes per model version;
see [`docs/sizing.md`](docs/sizing.md) for the full table (10k → 10M chunks) and the
≤ 2.5× corpus-text overhead target. Retention is managed with `recall gc --keep N`.

---

## Metrics & determinism

Native retrieval metrics only (deterministic given rankings): `recall@k`, `MRR`,
`nDCG@k`, `hit_rate@k`, per-query and aggregate. Generation-quality metrics come from
Ragas/DeepEval integrations and are never part of the default CI gate. Identical inputs
produce byte-identical manifests; every stochastic step takes an explicit seed. In
local mode, customer data (documents, chunks, embeddings) never leaves
customer-controlled storage.

---

## Project status & honest caveats

RecallOps is **alpha** (`Development Status :: 3 - Alpha`). The design and mechanism
have been validated on real infrastructure; the empirical product claim on *real
partner regressions* has not. Read this before relying on it:

- **Verification is strong for chunker, parser, fusion-weight, reranker, and
  corpus-drift changes.** For those, a cause is emitted as `verified` only if a
  counterfactual arm reverting that factor actually recovers the query.
- **Embedding-model *migration* yields per-category characterization, not a verified
  mechanism.** A black-box model swap has no ablatable internal cause, so RecallOps
  reports per-tag metric deltas and labels them as characterization; it does not
  claim a verified root cause for the migration itself.
- **On real, dense embeddings, config-change regressions are rarer than on toy
  corpora.** Production embeddings are robust, so the dramatic per-query breaks that
  are easy to manufacture on a small example corpus occur less often in practice.
- **No coverage number on real regressions yet.** The build-machine validation
  confirmed the *infrastructure and mechanism*, including that a real pgvector
  ivfflat index at default settings hides most true top-1 recall while RecallOps's
  exact shadow scoring recovers it, and that the never-flaky gate held against a noisy
  index (see [`docs/phase0-validation.md`](docs/phase0-validation.md)). The empirical
  go/no-go (verified-attribution coverage on a real corpus, real production golden set,
  and a real breaking change, over a 30-day PR-gate window) still requires a design
  partner. `recall phase0` runs exactly that validation.

If you have a real RAG pipeline and a regression you want to attribute, that is the
most useful thing you can bring.

---

## Docs & examples

- [`docs/sizing.md`](docs/sizing.md): storage footprint (10k → 10M chunks) and the ≤ 2.5× overhead target.
- [`docs/phase0-validation.md`](docs/phase0-validation.md): the real-infrastructure go/no-go runbook and build-machine findings.
- [`examples/quickstart.sh`](examples/quickstart.sh): the 5-minute quickstart as a runnable script ($0, no network).
- [`examples/corpus/`](examples/corpus/): the 12-doc example corpus the quickstart uses.
- [`examples/github-action/recall-ci.yml`](examples/github-action/recall-ci.yml): a ready-to-copy two-phase CI workflow.
- [`ARCHITECTURE.md`](ARCHITECTURE.md): design of the ingestion path, provenance store, and attribution engine.

## Contributing

Contributions are welcome. Set up a dev environment with `pip install -e ".[dev]"`
(or `uv pip install -e ".[dev]"`), then run `pytest` and `ruff check` before opening a
PR. Please open an issue at
[github.com/skundu42/recallops/issues](https://github.com/skundu42/recallops/issues) to
discuss larger changes first. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full
guide.

## License

Apache-2.0. See [`LICENSE`](LICENSE).

## Acknowledgements

RecallOps builds on [NumPy](https://numpy.org/), [PyArrow](https://arrow.apache.org/),
[Click](https://click.palletsprojects.com/), and [Rich](https://github.com/Textualize/rich),
and integrates with [pgvector](https://github.com/pgvector/pgvector) for the opt-in
adapter. The statistical gate draws on standard techniques: bootstrap confidence
intervals, McNemar's exact test, and the Benjamini-Hochberg FDR correction. Generation-
quality evaluation is delegated to projects such as Ragas and DeepEval.
