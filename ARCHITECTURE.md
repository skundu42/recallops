# RecallOps architecture

RecallOps is **retrieval regression testing with verified counterfactual
root-cause attribution for RAG pipelines**. It lives in the *ingestion path*,
the only place where the factors that move retrieval (parser, chunker, embedding
model, index, fusion weights, reranker, and the corpus itself) are all
observable, and answers two questions about any change to that path:

1. *Did retrieval get better or worse?* A golden-set eval with a
   statistically gated, never-flaky pass/fail.
2. *Exactly why?* A two-phase attribution that emits a cause as **verified**
   only when a counterfactual arm reverting that factor actually recovers the
   query. Everything else is a labeled hypothesis.

It is **not** a vector database, **not** an observability/tracing tool, and
**not** an eval-metrics library; it integrates with those. What it owns is
provenance + causal attribution.

The system is four layers over a content-addressed store. Everything below is
accurate to the modules under [`recallops/`](recallops/); a per-module map is at
the end.

```
        corpus (docs/)                              golden dataset
             │                                            │
   ┌─────────▼────────────────────────────────────────────▼──────────┐
   │  LAYER 0 - PROVENANCE STORE + RETRIEVAL ENGINE                    │
   │                                                                  │
   │   parse ─▶ chunk ─▶ embed ─▶ index ─▶ retrieve ─▶ rerank         │  ingestion
   │     │        │        │         │          │          │          │    path
   │  doc_id  chunk_id  emb_key  collection   dense/sparse/fused/final │
   │     └──────────── content-addressed Merkle DAG ─────────┘        │
   │        SQLite (metadata) + Parquet (chunk sets, fp16 emb)        │
   │                                                                  │
   │   replay = exact KNN over stored emb   ‖   live = vector adapter │
   │   shadow scorer: full-corpus exact cosine, adapter-independent   │
   └───────────────┬──────────────────────────────────────────────────┘
                   │  snapshot manifest (snap_…)  +  per-stage candidates
       diff A→B    ▼
   ┌────────────────────────────┐      ┌──────────────────────────────────┐
   │ LAYER 1 - FUNNEL (obs.)    │      │ STATISTICAL GATE                 │
   │ tracer chunk; stage-wise   │      │ calibrate ε · bootstrap CI       │
   │ before/after ranks;        │      │ McNemar · per-tag BH-FDR         │
   │ ANN-divergence quarantine  │      │ near-tie exclusion (never-flaky) │
   └────────────┬───────────────┘      └──────────────────────────────────┘
                │  implicated factors
                ▼
   ┌─────────────────────────────────────────────────┐
   │ LAYER 2 - COUNTERFACTUAL ABLATION (causal)       │
   │ memoized arms · exact-KNN replay scoring         │
   │ Shapley (k ≤ 3) · CONFIRMATION RULE:             │
   │   a factor is verified  ⟺  its revert arm        │
   │   recovers the query (rank + metric restored)    │
   └────────────┬─────────────────────────────────────┘
                │  verified causes + hypotheses
                ▼
   ┌─────────────────────────────────────────────────┐
   │ LAYER 3 - RENDERER (evidence-constrained)        │
   │ deterministic narratives · md / html / json      │
   │ faithfulness audit: no claim without evidence    │
   └─────────────────────────────────────────────────┘
```

---

## Layer 0: Provenance store + retrieval engine

### The content-addressed Merkle DAG

The ingestion pipeline is a linear DAG of stages, canonically ordered
`parse → chunk → embed → index → retrieve → rerank`. Every artifact along it is
keyed by a hash of its *content*, never by time or randomness
([`hashing.py`](recallops/hashing.py); `h()` is `sha256` truncated to 16 hex):

| Identifier | Derivation |
|---|---|
| `doc_id` | `doc_` + hash of the raw document bytes |
| `text_hash` | `tx_` + hash of parsed/chunk text |
| `chunk_id` | `ch_` + hash of `(doc_id, span_start, span_end, text)` |
| `params_hash` | `ph_` + hash of a stage's canonical-JSON params |
| `embedding_key` | `emb_` + hash of `(text_hash, provider, model, dims, params_hash)` |
| `merkle_root` | `mr_` + hash of the sorted `(source_path, doc_id)` set |
| `snapshot_id` | `snap_` + hash of the manifest **core** (pipeline DAG + corpus info + artifact URIs) |

Two derived keys tie the DAG together:

- **`chunkset_key`** = `h(merkle_root, parse_tool, parse_params_hash,
  chunk_tool, chunk_params_hash)`: the identity of a set of chunks. Identical
  corpus + parse + chunk config always resolves to the same stored chunk set.
- **`collection_name`** = `col_` + hash of the (optional project namespace, the)
  corpus `merkle_root`, and the `parse`, `chunk`, `embed`, `index` stage dicts:
  the serving-index identity. Including the merkle root means every distinct
  chunk set gets its own collection, so re-ingesting an edited corpus under the
  same pipeline can never keep serving chunks of deleted documents (upserts do
  not delete). Retrieve-stage changes still never force an index rebuild, and
  two projects sharing one vector DB never collide (the namespace is stored in
  the project meta, not in any content hash).

A **`SnapshotManifest`** ([`models.py`](recallops/models.py)) binds one
pipeline DAG, its `CorpusInfo` (`doc_count`, `chunk_count`, `merkle_root`), and
the artifact URIs. Its `snapshot_id` is the hash of that core, so committing the
same inputs twice is idempotent and re-computes the same id on any machine
(`created_at` is deliberately left empty).

### The store

[`store.py`](recallops/store.py), `ProjectStore`, persists everything under
`.recall/`:

- **SQLite** (`db.sqlite`) holds metadata and small JSON artifacts: `docs`,
  `corpus_states`, `chunksets`, an `embeddings` cache index, `snapshots`,
  `datasets`, `evals`, a generic `json_blobs` table (diffs, attributions,
  calibration, ablation checkpoints), and `meta`.
- **Parquet** under `.recall/artifacts/` holds the bulk data: chunk sets
  (`chunks/`) and **fp16** embedding parts (`emb/`, a `FixedSizeList` vector
  column). fp16 halves disk and cache pressure without moving rankings; see
  [`docs/sizing.md`](docs/sizing.md).

Because keys are content hashes, `put_chunks`, `put_embeddings`, and
`commit_snapshot` are all **memoizing and idempotent**: identical inputs resolve
to existing artifacts with zero recomputation (FR-1.5). An in-process
**embedding cache** (keyed by `embedding_key`) plus an in-memory query-vector
cache mean eval / calibration / ablation loops that re-query the same golden
questions never re-embed or re-bill. `gc(keep_last, pinned)` prunes chunk and
embedding artifacts no longer referenced by a retained snapshot.

Ingestion has two front doors that produce **byte-identical** provenance for the
same corpus and config, so their snapshots are directly diffable:

- **Managed mode**: [`ingest.py`](recallops/ingest.py) drives
  parse → chunk → embed → index with dedup at every step and optional
  write-through to a vector adapter.
- **Recorder SDK**: [`recorder.py`](recallops/recorder.py) lets a
  bring-your-own pipeline log the same docs / chunks / embeddings / retrieval
  candidates under the same content addressing.

### The retrieval engine

[`retrieval.py`](recallops/retrieval.py), `RetrievalEngine`, executes a query
against a snapshot and **captures the candidate list at every stage** into a
`QueryRun` (`dense`, `sparse`, `fused`, `reranked`, `final`), which is what
Layer 1 later reads:

- **dense**: replay mode (`adapter=None`) scores exact cosine over the
  snapshot's stored fp16 embeddings; live mode queries the adapter's collection.
  Both L2-normalize and share the `(-score, chunk_id)` tie-break, so replay
  reproduces an exact live index bit-for-bit.
- **sparse**: the in-package BM25 index ([`bm25.py`](recallops/bm25.py),
  Lucene-style idf) over chunk texts. Sparse candidates are always computed so
  funnel attribution has every stage even for a dense-only pipeline.
- **fused**, [`fusion.py`](recallops/fusion.py): `weighted` (per-list min-max
  normalization with a `bm25_weight`) or `rrf` (reciprocal-rank fusion), with
  deterministic tie-breaks. A pipeline with no `hybrid` block leaves `fused`
  in dense order.
- **rerank**: [`rerankers.py`](recallops/rerankers.py) reorders the fused
  top-k (e.g. `recall.rerankers.overlap`) and its output becomes `final`.

Dense and sparse each retrieve `max(top_k·4, 20)` candidates before fusion.
Crucially, `exact_dense_ranks()` is the **shadow scorer** (FR-6.2): a
full-corpus exact-cosine ranking computed from stored embeddings regardless of
the serving adapter. It is the ground truth that lets attribution tell a real
config regression apart from ANN/index noise (see Layer 1).

Metrics ([`evalrunner.py`](recallops/evalrunner.py)) are deterministic functions
of the ranked doc list and expected sources: `recall@k`, `hit_rate@k`, `MRR`,
`nDCG@k`. `evaluate()` runs a golden set through the engine (replay or live) and
persists a content-addressed `EvalResult`.

---

## Layer 1: Observational funnel attribution

[`funnel.py`](recallops/funnel.py) localizes *where* a regressed query broke,
purely observationally: **no re-embedding and no counterfactual runs**. For
each stable regressed query it:

1. Identifies the **tracer**: the chunk A used to win, the top-ranked chunk in
   A's final list whose source document is one of the expected sources.
2. **Resolves the tracer's identity in B** through the chunk-fate alignment from
   [`diffing.py`](recallops/diffing.py): a *split* tracer is followed to all its
   descendant fragments; any other aligned tracer to its best descendant (an
   intact chunk may have a new content-addressed id after a boundary shift); an
   unaligned tracer stays itself.
3. Reads the tracer's **before/after rank at each stage** (`dense`, `sparse`,
   `fused`, `rerank`) straight from the captured candidate lists (1-based; absent
   ⇒ `None`), plus `shadow_exact_rank_after` from B's exact-cosine shadow.
4. Flags **ANN divergence**: when an optional live run's dense rank differs from
   the exact shadow rank by more than `ANN_DIVERGENCE_POSITIONS` (2), the
   serving index, not the config, moved the target. This is its own failure
   class, quarantined from the config factors.

`failing_stage()` scans `index → dense → sparse → fused → rerank`, returns `ann`
when the exact shadow is still in top-k but a live run diverged, and otherwise
falls back to the stage with the largest rank degradation. `implicated_factors()`
maps that stage to the config factors that actually changed (`STAGE_FACTORS ∩
config_diff`). Those implicated factors are the *hypotheses* Layer 2 then tries
to confirm.

---

## Layer 2: Causal counterfactual ablation

[`ablation.py`](recallops/ablation.py) turns "which factors changed" into
"which factor *caused* the regression" by running counterfactuals.

**Factors.** The changed pipeline stages become `kind="stage"` factors; a
synthetic `kind="corpus"` factor is added when the corpus content changed
(FR-7.6).

**Arms.** A factor set of size `k` defines a lattice of *arms*: manifests where
each factor is independently pinned to its A or B state. With `k ≤ 3` (or
`--arms full`) the **full lattice** is enumerated; with `k > 3` under `--arms
auto` the set is one-factor-at-a-time (each flippable factor at A, the rest at
B) plus registered interaction pairs and the all-A / all-B endpoints.
`pruned_to` (the funnel's implicated factors) can restrict which factors are
flipped. Every arm id is content-addressed from its assignment.

**Materialization is memoized.** An arm is rebuilt entirely from the provenance
store: the chosen corpus merkle resolves to its documents (raw bytes included),
which are re-parsed, re-chunked, and embedded under the arm's composed pipeline.
Because every id is content-addressed, an arm that recomposes an
already-materialized state **resolves to the existing snapshot with zero new
embedding calls**, and arms sharing a chunk set or embedding share the cache
(FR-7.1). `plan_arms()` is a dry-run cost gate (counts only the still-missing
embedding keys, deduplicated); `run_arms()` is **resumable** via a checkpoint
so an interrupted job never repeats completed work.

**Arm scoring always runs in replay mode**, exact KNN over stored embeddings,
so ANN noise can never enter causal analysis (FR-7.2). `shapley()` gives an
exact Shapley attribution over the boolean factor lattice when the full lattice
is present.

### The confirmation rule (the core invariant)

[`confirm.py`](recallops/confirm.py) is where "verified" is earned. For each
factor, the **single-revert arm**, that factor at A, every other factor held at
B, is looked up, and the factor is emitted as a `VerifiedCause` **iff** that arm
*recovers* the query:

- the target document returns within its original rank (`rank1`, default) or
  within top-k (`topk`), **and**
- the primary metric is restored to at least its pre-change value
  (`_metric_restored`, a necessary guard so a second evicted source can't be
  hidden by a single-source rank check).

Funnel-implicated factors whose revert arm did **not** recover the query become
`Hypothesis` objects carrying their co-occurring evidence. **No factor is ever
both.** `fidelity_check()` re-verifies every emitted cause against its arm eval
(the §13 fidelity gate is 1.0 by construction; < 1.0 flags a false cause).

### Statistical gating (never-flaky)

Deciding whether a *diff* fails a release gate is a separate concern, handled by
[`gating.py`](recallops/gating.py) + [`stats.py`](recallops/stats.py) over the
per-query classification in [`diffing.py`](recallops/diffing.py):

- **`calibrate`** re-runs an identical snapshot `n_runs` times (rebuilding the
  index between runs where the adapter supports it) and records the serving
  noise floor: per-metric std, and **`epsilon`** = the 95th percentile of the
  target-vs-kth score-gap fluctuations, taken at the gated metric's `k`.
- **`classify_query`** labels each query `regressed / improved / changed-top-k /
  unchanged` by the primary-metric delta, and marks it **`unstable`** (a
  near-tie) when the after-run score gap is below `epsilon`.
- **`evaluate_gate` (statistical mode)** combines three signals: a bootstrap 95%
  CI on the **stable-query** primary-metric deltas (fail if the upper bound < 0),
  **McNemar's exact test** on stable-query hit flips, and **per-tag McNemar**
  corrected with **Benjamini-Hochberg FDR**. The overall McNemar is judged on
  its own `q`; BH corrects only the per-tag subgroup family (so finer tagging
  never makes a global regression harder to detect).

**Near-tie (`unstable`) queries are excluded from every red signal** — the CI
and both flip counts — so pure serving noise can never turn a gate red, the
never-flaky invariant. The cost is that a regression small enough to show up
only as near-ties sits below the calibrated noise floor and is not flagged; the
excluded count is surfaced in the gate details so a reviewer can see it. A `raw`
`--fail-if` threshold exists only as an escape hatch; the docs steer callers to
the statistical gate.

---

## Layer 3: Evidence-constrained renderer

[`narrative.py`](recallops/narrative.py) renders a **deterministic,
template-based** narrative for each attribution report: no LLM, no fabrication.
Sentences are keyed on structured signals only: the chunk-fate class, the
verified-cause factor, ANN divergence, and a funnel-only fallback when nothing
is verified. Verified sentences cite their arm id as `(verified: <arm_id>)`;
every hypothesis is prefixed `Unverified:` with its evidence.
`narrative_faithfulness_audit()` scans a rendered narrative for controlled
tokens (arm ids, chunk-fate classes, stage names, factor names) not backed by
the report's structured evidence and returns them as violations; a faithful
narrative returns `[]` (the §13 narrative gate).

[`report.py`](recallops/report.py) renders the same structured artifacts into
delivery formats with **no re-execution**: the §10.4 attribution JSON, a
PR-comment Markdown summary, a self-contained static HTML report, the migration
comparison Markdown, `rich` tables, and the stable `--format json` diff schema.

---

## Module map

Every module under [`recallops/`](recallops/), one line each. Sub-package
`__init__.py` files (`adapters/`, `pipeline/`) are empty package markers.

### `recallops/` (core)

| Module | Responsibility |
|---|---|
| `__init__.py` | Public API surface (PRD §11): re-exports `Recorder`, `ProjectStore`, `RetrievalEngine`, adapters, providers, `evaluate`, `build_pipeline`, and the core models. |
| `hashing.py` | Content addressing for every artifact: doc / text / chunk / params / embedding / merkle / snapshot ids + canonical JSON. |
| `models.py` | Core dataclasses (PRD §10 schemas): `StageSpec`, `PipelineDAG`, `SnapshotManifest`, `ChunkRecord`, `GoldenCase/Dataset`, `QueryRun/Eval`, `DiffResult`, `FunnelReport`, `Arm/ArmResult`, `VerifiedCause/Hypothesis`, `AttributionReport`, `CalibrationRecord`, `GateResult`: all round-trip to plain JSON. |
| `store.py` | `ProjectStore`: content-addressed provenance store, SQLite metadata + Parquet chunk sets / fp16 embeddings, embedding cache, snapshot/dataset/eval/JSON persistence, `gc` retention. |
| `config.py` | `recall.yaml` (dependency-free YAML subset) → `ProjectConfig`; `build_adapter` / `build_provider`; embedding-spec parsing. |
| `ingest.py` | Managed ingestion: `build_pipeline` DAG, `chunkset_key` / `collection_name`, parse→chunk→embed→index with dedup, optional adapter write-through, snapshot commit. |
| `recorder.py` | Recorder SDK (bring-your-own pipeline) that logs docs / chunks / embeddings / retrieval with identical content addressing. |
| `retrieval.py` | `RetrievalEngine`: per-stage candidate capture; replay (exact KNN over stored emb) vs live (adapter); exact-cosine shadow scorer (FR-6.2). |
| `bm25.py` | In-package BM25 sparse index (Lucene-style idf) over chunk texts. |
| `fusion.py` | Dense/sparse candidate fusion: `weighted` (min-max) and `rrf`, deterministic tie-breaks. |
| `rerankers.py` | Reranker registry; `recall.rerankers.overlap` token-overlap reranker. |
| `evalrunner.py` | Native retrieval metrics (`recall@k`, `hit_rate@k`, `MRR`, `nDCG@k`) + `evaluate()` over a golden set, replay or live. |
| `diffing.py` | Snapshot diff (FR-5): config diff, metric deltas, per-query classification + stability, chunk-fate alignment (span-overlap; fuzzy Jaccard on parser change). |
| `funnel.py` | Layer 1 observational funnel: tracer resolution, stage-wise before/after ranks, failing-stage localization, ANN-divergence quarantine. |
| `ablation.py` | Layer 2 counterfactual engine: factor enumeration, arm-lattice construction (`k ≤ 3` full / auto pruned), store-materialized memoized arms, replay scoring, exact Shapley, resumable runs. |
| `confirm.py` | Confirmation rule (FR-8.1): single-revert-arm recovery + metric restoration → `VerifiedCause`; unrecovered implicated factors → `Hypothesis`; fidelity check. |
| `gating.py` | Noise-floor calibration (`epsilon`) and the statistical release gate: bootstrap CI, McNemar, per-tag BH-FDR, near-tie exclusion; raw `--fail-if` escape hatch. |
| `stats.py` | Statistical primitives: percentile bootstrap CI, exact McNemar p, Benjamini-Hochberg FDR, epsilon derivation. |
| `dataset.py` | Golden dataset bootstrap (FR-3): offline heuristic generator, JSON/JSONL import + trace mining, curation, stratification. |
| `narrative.py` | Layer 3 evidence-constrained narrative renderer (deterministic, no LLM) + faithfulness audit. |
| `report.py` | Reporting artifacts: attribution JSON (§10.4), PR-comment Markdown, self-contained HTML, migration-comparison Markdown, `rich` tables, stable JSON diff schema. |
| `phase0.py` | Phase-0 real-stack go/no-go harness: ANN effect + real noise floor + attribution quality, provenance-stamped. |
| `scorecard.py` | Attribution-engine self-test on the example corpus (four known-cause scenarios + noise floor) against the §13 gates. |
| `cli.py` | The `recall` Click CLI: wires every engine module end-to-end over a `ProjectStore` + `recall.yaml`, with cost gating. See [`docs/cli.md`](docs/cli.md). |

### `recallops/adapters/` (vector DB contract)

| Module | Responsibility |
|---|---|
| `base.py` | `VectorAdapter` ABC + `Capability` descriptor: the adapter contract (FR-12). |
| `local.py` | `LocalIndexAdapter`: built-in exact-KNN adapter (npz per collection); `ann_mode` noise overlay for calibration / divergence. |
| `pgvector.py` | `PgVectorAdapter`: pgvector serving index (cosine `<=>`), connection-free SQL builders, tunable ivfflat `probes`. |

### `recallops/pipeline/` (ingestion primitives)

| Module | Responsibility |
|---|---|
| `parsers.py` | Document parsers (`text-v1`, `markdown-v2`) producing parsed text + lineage. |
| `chunkers.py` | Chunkers (`fixed_token`, `markdown_heading`, `sentence`) producing char-span chunk records (the span invariant that makes fate alignment possible). |
| `providers.py` | Embedding providers: `LocalHashProvider` ($0 offline feature-hashing), `OpenAIProvider` (real), + cost estimation. |

---

## Key invariants

These three properties are what the whole design defends.

### 1. Content-addressing ⇒ determinism and memoization

Every identifier derives from content, never from time or randomness; manifests
carry an empty `created_at`. Identical inputs produce **byte-identical
manifests** on any machine, re-running an identical pipeline performs **zero
embedding calls**, and a counterfactual arm that recomposes an existing state
resolves to that snapshot for free. This is what makes reproducible reports and
memoized ablation possible at all.

### 2. Verified ⟺ recovered

No cause is emitted as **verified** unless the counterfactual arm that reverts
that single factor (all others held at B) *recovers the query*: the target
returns within its original rank **and** the primary metric is restored
(`confirm.py`). Every other signal is a labeled **hypothesis** with its evidence.
A factor is never both. This is the product's central honesty guarantee, and the
renderer's faithfulness audit enforces that no narrative can assert more than the
structured evidence supports.

### 3. Never-flaky

Calibration derives a per-snapshot near-tie threshold `epsilon` from real
serving noise, and near-tie queries are excluded from every gate flip count, so
pure serving/ANN variance can never turn a gate red. Independently, all
counterfactual arms and the shadow scorer run in **exact replay**, so ANN noise
never enters causal analysis. The `LocalIndexAdapter`'s `ann_mode` exists
specifically to inject controlled serving noise so this can be validated.

---

## Adapter contract + capability descriptor

A vector DB integrates by implementing [`VectorAdapter`](recallops/adapters/base.py):

```python
class VectorAdapter(ABC):
    name: str
    def capabilities(self) -> Capability: ...
    def ensure_collection(self, collection: str, dims: int) -> None: ...
    def upsert(self, collection, ids, vectors, payloads) -> None: ...
    def query_dense(self, collection, vector, top_k) -> list[tuple[str, float]]: ...
    def count(self, collection: str) -> int: ...
    def drop(self, collection: str) -> None: ...
    def rebuild(self, collection: str, seed: int = 0) -> None: ...   # default: no-op
```

`query_dense` returns `(chunk_id, score)` pairs with **higher = more similar**
(the pgvector adapter returns `1 - cosine_distance` to match the local adapter's
cosine). `rebuild` is the calibration hook: adapters that can re-seed their index
implement it so `calibrate` can measure real serving noise.

The **`Capability` descriptor** is load-bearing: it tells the funnel layer what
a given adapter exposes so shadow re-scoring can fill the gaps:

```python
@dataclass(frozen=True)
class Capability:
    name: str
    exposes_dense_scores: bool
    exposes_sparse: bool
    supports_rebuild: bool
```

Both shipped adapters advertise `exposes_dense_scores=True`,
`exposes_sparse=False`, `supports_rebuild=True`. Because RecallOps computes
sparse and shadow-dense rankings itself from stored artifacts, an adapter is
never *required* to expose scores or sparse retrieval for attribution to work;
the capability descriptor just lets the engine skip its own shadow work when the
adapter can answer directly.

---

## Scope and honest limits

Verification is strongest for **chunker, parser, fusion/retrieve, and
corpus-drift** changes, where a revert arm produces a crisp recover / not-recover
signal. An **embedding-model migration** is deliberately treated as a black box:
the renderer emits per-category *characterization* (per-tag metric deltas, e.g.
via `compare-embeddings`), not a verified mechanism: a model swap that recovers
a query tells you *that* the model mattered, not *why*.

The engine is validated on real serving infrastructure (the pgvector Phase-0 run
in [`docs/phase0-validation.md`](docs/phase0-validation.md)): shadow scoring
recovers the true ranking a mis-tuned ANN index hides, and the never-flaky gate
holds against a noisy index. But real dense embeddings are robust, so
config-change regressions are *rarer* on real corpora than on toy ones, and the
product does **not yet** have a verified-attribution coverage number on real
partner regressions. Establishing that number is exactly what `recall phase0` is
built to do; see the validation runbook.
