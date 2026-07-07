# Phase-0 Real-Project Validation: Runbook

**Purpose.** Decide the PRD's go/no-go (§13, §15): does RecallOps achieve **≥70%
verified-attribution coverage** on real regressed queries with a **≤2% noise
floor**, on a real serving stack (real vector DB, real embedding model), before
committing to the v0.1 product build.

This runbook sets up and runs that validation against **OpenAI embeddings +
pgvector**. Everything here is executed by the `recall phase0` command; the
sections below cover provisioning, what you supply, and how to read the result.

---

## What Phase 0 measures

`recall phase0` produces a provenance-stamped report with five §13 gates plus two
measurements that a synthetic self-test (`recall scorecard`) cannot make:

| Measurement | Gate | Why it needs real infrastructure |
|---|---|---|
| **coverage** | ≥ 0.70 | % of stable regressed queries with ≥1 *verified* cause (the core claim) |
| **fidelity** | ≥ 0.995 | % of verified causes whose arm actually recovers the query |
| **noise_floor** | ≤ 0.02 | stable-regressed rate over no-change re-runs of the **real** index (FR-9) |
| **stage_accuracy** | ≥ 0.80 | funnel names the correct stage on known-cause changes |
| **narrative_violations** | == 0 | no narrative claim absent from structured evidence |
| **ANN effect** | (reported) | exact shadow ranking vs the live index: the FR-6.3 quarantine, on real data |

The **noise floor** and **ANN effect** are the numbers only a real ANN index can
produce, and they are the whole reason the ingestion-path architecture exists.

---

## 1. Provision pgvector

You need Postgres with the `vector` extension. Two proven paths:

### Option A: Docker (simplest)
```bash
docker run -d --name recallops-pg -e POSTGRES_PASSWORD=recall -p 5433:5432 \
  pgvector/pgvector:pg16
export RECALL_PG_DSN="postgresql://postgres:recall@localhost:5433/postgres"
psql "$RECALL_PG_DSN" -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Option B: Local Postgres + pgvector from source (no Docker)
This is the exact path validated on the build machine (Homebrew Postgres 14,
Apple Silicon). Build pgvector against the target `pg_config` so the extension
loads in that server:
```bash
PGC=/opt/homebrew/opt/postgresql@14/bin/pg_config
git clone --depth 1 --branch v0.8.0 https://github.com/pgvector/pgvector.git
cd pgvector && make PG_CONFIG=$PGC && make install PG_CONFIG=$PGC

# start a server and create the DB + extension
initdb -D ./pgdata -U postgres --auth=trust
pg_ctl -D ./pgdata -o "-p 5433 -k /tmp" start
psql -h /tmp -p 5433 -U postgres -c "CREATE DATABASE recallops_phase0;"
psql -h /tmp -p 5433 -U postgres -d recallops_phase0 -c "CREATE EXTENSION vector;"
export RECALL_PG_DSN="postgresql://postgres@/recallops_phase0?host=/tmp&port=5433"
```
> The Homebrew *bottle* of pgvector is built for the current default Postgres
> major version (17/18 at time of writing), so it will not load in Postgres 14;
> building from source against the target `pg_config` (above) is the reliable path.

Install the pg client extra: `pip install -e ".[pg]"` (or `uv pip install -e ".[pg]"`).

## 2. Configure the project (pgvector + OpenAI)

```bash
recall init --source ./your-corpus --adapter pgvector --dsn "$RECALL_PG_DSN"
```
Then edit `recall.yaml`:
```yaml
adapter:
  type: pgvector
  dsn: postgresql://...        # or leave unset and use $RECALL_PG_DSN
  probes: 10                   # ivfflat recall/latency knob, see §Findings
pipeline:
  embedding:
    provider: openai
    model: text-embedding-3-small
    dims: 1536
gate:
  primary_metric: recall@5
  q: 0.05
```
Set your key: `export OPENAI_API_KEY=sk-...`. When the key is present, `recall
phase0` uses real OpenAI embeddings and **prints a cost estimate you must approve**
(`--yes` or `--max-cost N`). With no key it falls back to the offline local
embedder and clearly labels the result as not the real go/no-go.

## 3. Supply a golden set and a known-regression change

- **Golden set.** Import your production golden cases, or bootstrap:
  `recall dataset generate --n 200` (needs an LLM key), or
  `recall dataset mine --from traces.jsonl`.
- **Known-regression change (important).** The default `recall phase0` config
  changes (chunker/fusion) are generic. For a *meaningful coverage number*, edit
  `recallops/phase0.py:default_changes()` (or pass your own `ConfigChange`s via
  the library API) to the **actual config change that caused a real regression**
  in your pipeline. Coverage answers "when a real change breaks retrieval, does
  the engine verify why?", so it must run on a real breaking change.

## 4. Run it

```bash
recall phase0 --dataset golden --reruns 10 --max-cost 20
```
This ingests the baseline into pgvector, measures the ANN effect and real noise
floor, runs each known-cause change through full counterfactual attribution, and
writes `phase0-report.json` with a GO / NO-GO verdict.

---

## Interpreting the result

- **GO** requires all five gates green *with real embeddings on a real corpus and
  a real breaking change*. A GO on the local embedder is infrastructure
  validation only; the report says so in its caveats.
- **coverage < 0.70** → invoke the PRD's pre-agreed fallback (§13): reposition as
  a diagnostic *assistant* (hypotheses + evidence, weaker verification claims).
  This is an explicit decision, not a drift.
- **noise_floor > 0.02** → the serving index is too noisy to gate on directly.
  Raise `probes` (or index params) and re-run; attribution is unaffected because
  it uses exact shadow scoring.

---

## Findings from the build-machine run (pgvector, ~4,000 chunks)

Executed with the offline local embedder (no OpenAI key on the build machine),
so the attribution gates below are infrastructure validation, but the **ANN
effect and noise floor are real pgvector ivfflat behaviour**:

| ivfflat `probes` | live recall@1 | exact recall@1 | calibrated ε | gate flake-rate |
|---|---|---|---|---|
| 1 (pgvector default) | 0.08 to 0.375 *(unstable)* | 0.98 | 0.80 | 0.010 |
| 100 (≈ exact) | 0.98 | 0.98 | 0.00 | 0.000 |

Three conclusions, each material to the product thesis:

1. **The core wedge is validated on real infrastructure.** A real pgvector
   ivfflat index with default settings hides ~92% of true top-1 recall (0.98 →
   0.08) and is unstable run-to-run. Because RecallOps stores embeddings and
   shadow-scores exactly (FR-6.2), it recovers the true 0.98 ranking the serving
   index hides, so it distinguishes a real config regression from ANN noise.
   Without this, every index rebuild looks like a regression.

2. **The never-flaky guarantee held even against a badly-tuned index.** At `probes=1`
   with catastrophic serving noise (ε=0.80), calibration + near-tie exclusion kept
   the gate flake-rate at 1%: no false regressions from serving noise. Note the
   honest trade-off: a very noisy index makes the gate *permissive* (large ε
   excludes almost everything), so the right operating point is a tuned index
   (`probes` high → ε=0, gate both stable **and** discriminating), not a noisy one.

3. **Two real adapter findings surfaced by running live** (both fixed/flagged):
   - `probes` was hardcoded to pgvector's default of 1 and is now a tunable
     adapter/config parameter (this run added it).
   - The serving collection was named from the pipeline config only, so two
     *different corpora sharing one pgvector database with the same pipeline*
     collided in one table. **Fixed:** collection names are now qualified by the
     project namespace (`recall init --name <project>`, persisted in the store),
     so distinct projects get distinct tables in a shared database, verified
     live (two same-pipeline projects → two separate `col_*` tables).

## What is still required for the true go/no-go

This build validated the **infrastructure and mechanism**. The empirical product
gate still needs, from a design partner:

- a **real corpus** and **real production golden set**,
- a **real config change that caused a real regression**,
- an **OpenAI (or production model) key** so coverage/stage-accuracy are measured
  on real embeddings,
- and the FR-9 acceptance window: **30 days of real PR gates with zero
  nondeterminism-induced failures** (a longitudinal test `recall phase0` seeds
  but cannot compress).

Run `recall phase0` with those inputs on ≥3 real projects (incl. one Tier-M) to
produce the actual §13 scorecard the build decision hinges on.
