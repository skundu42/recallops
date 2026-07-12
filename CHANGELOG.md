# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the project is pre-1.0 (`0.x`), minor versions may include breaking
changes; these will always be called out under **Changed** or **Removed**.

## [Unreleased]

### Added
- Qdrant adapter (`recallops[qdrant]`): server or embedded local mode.
- Chroma adapter (`recallops[chroma]`): embedded.
- LanceDB adapter (`recallops[lancedb]`): embedded, exact flat scan.
- Shared adapter behavioral contract test suite; all adapters must match the
  exact local reference's cosine-similarity score semantics.
- sentence-transformers embedding provider (`recallops[st]`): real semantics
  at $0, CPU-pinned for determinism.
- Cohere and Voyage embedding providers (no extra needed; raw HTTP), with
  query/document input types and cost gating.
- `EmbeddingProvider.embed_queries` hook (backward compatible: defaults to
  `embed`).
- `recallops[all]` convenience extra.
- Composite GitHub Action (`uses: skundu42/recallops@<ref>`): two-phase gate
  with a single self-updating PR comment (`recallops.pr_comment`); the example
  workflow now consumes the action.
- LLM-backed golden-dataset generation: `recall dataset generate --llm
  openai[:model]`, cost-gated, using the caller's `OPENAI_API_KEY`.
- Dataset curation edits: `recall dataset curate --edit-file edits.jsonl`.
- `recall snapshot pin`/`unpin`: persistent pins that `recall gc` honors.
- Live eval and calibrate warn when the serving collection is empty instead
  of silently scoring zeros.

### Fixed

- Every provider HTTP call now has a timeout (`RECALL_HTTP_TIMEOUT`, default
  60 s); a stalled embedding API can no longer hang ingest or CI forever.
  OpenAI now shares the same HTTP path as Cohere/Voyage.
- Query embeddings are persisted in the store (fp32), so a new process (every
  CI run) no longer re-embeds and re-bills the whole query set.

### Unchanged
- Snapshot ids are byte-identical to 0.1.0 (pinned by test).

## [0.1.0] - 2026-07-09

Initial engine. RecallOps is retrieval regression testing with **verified**
counterfactual root-cause attribution for RAG pipelines: it versions the
ingestion path, tests retrieval in CI with a never-flaky statistical gate, and
attributes regressions to a cause only when a counterfactual arm reverting that
factor actually recovers the query.

### Added

- **Content-addressed provenance store**: versioned, byte-diffable manifests of
  documents, chunks, embeddings, and config; identical inputs yield identical
  snapshot ids, and re-running an identical pipeline performs zero embedding
  calls. Embeddings persisted as fp16 columnar data.
- **Managed ingestion + Recorder SDK**: `recall ingest` for the built-in
  pipeline, plus a `Recorder` API to record the same content-addressed
  provenance from a bring-your-own ingestion pipeline. SDK snapshots are
  byte-diffable against managed-mode snapshots of the same corpus.
- **Retrieval**: dense (exact-KNN), sparse (BM25), hybrid fusion, and reranker
  stages, with exact shadow scoring that recovers true rankings an ANN index may
  hide.
- **Golden datasets**: generate, mine, and manage golden query sets
  (`recall dataset`).
- **Evaluation**: native, deterministic retrieval metrics (`recall@k`, `MRR`,
  `nDCG@k`, `hit_rate@k`), per-query and aggregate, with raw-threshold and
  statistical gating (`recall eval`).
- **Diff + chunk alignment**: snapshot-to-snapshot diffing that classifies each
  regressed query and aligns old/new chunks (split/merge/move detection).
- **Funnel attribution**: observational stage-by-stage funnel analysis that
  localizes the failing retrieval stage and implicates candidate factors.
- **Counterfactual ablation + confirmation rule**: memoized counterfactual arm
  lattice; a cause is labeled `verified` only when the arm reverting that single
  factor (others held at B) recovers the query, and it cites the confirming
  `arm_id`. Everything else is a labeled hypothesis. Embedding-model swaps emit
  per-tag characterization rather than a verified mechanism.
- **Statistical gating**: per-snapshot calibration (`recall calibrate`) derives
  a near-tie threshold and excludes near-ties; aggregate deltas gated by
  bootstrap 95% CI, per-query flips by McNemar's exact test, corrected across
  tags with Benjamini-Hochberg FDR. Designed to be never-flaky.
- **Narratives**: human-readable explanations constrained to structured
  evidence; no narrative claim is emitted that is not backed by the report.
- **CLI**: 17 commands: `init`, `ingest`, `snapshot`, `dataset`, `eval`,
  `calibrate`, `diff`, `attribute`, `compare-embeddings`, `compare-chunkers`,
  `sweep`, `drift`, `ci`, `report`, `gc`, `scorecard`, `phase0`.
- **Adapters**: built-in local exact-KNN adapter (offline, $0) and an opt-in
  pgvector adapter (`recallops[pg]`), with project-namespaced collection names
  so distinct projects get distinct tables in a shared database.
- **GitHub Action**: a two-phase CI gate (blocking eval/diff + async deep
  attribution) shipped as an example workflow.
- **Scorecard**: `recall scorecard` self-test that exercises the engine on a
  synthetic corpus and reports the §13 fidelity metrics that do not need real
  infrastructure.
- **Phase-0 harness**: `recall phase0` runbook and command to measure verified
  attribution coverage, fidelity, noise floor, stage accuracy, and the ANN
  effect against real infrastructure (OpenAI embeddings + pgvector).
- **Provider cost gating**: any provider-billed operation prints a cost
  estimate and requires explicit approval (`--yes` / `--max-cost`); the local
  provider is $0 and auto-approves.
- **Packaging & release**: Apache-2.0 licensed, ships `py.typed`, supports
  Python 3.11 to 3.13, PyPI-metadata-valid (hatchling backend). The version is
  tag-derived (hatch-vcs), and a `vX.Y.Z` tag publishes to PyPI automatically
  via GitHub Actions Trusted Publishing (OIDC, no stored token). See
  `docs/PUBLISHING.md`.

### Fixed

Pre-release hardening from an adversarial review of the engine:

- `recall gc` no longer deletes snapshots it was told to keep: with fewer
  snapshots than `--keep` the retention slice wrapped negative and pruned the
  oldest snapshots. It now also removes index rows before unlinking artifact
  files, so an interrupted `gc` cannot poison the embedding cache.
- Serving collections now include the corpus identity, so re-ingesting an edited
  corpus under the same pipeline no longer keeps serving chunks of deleted
  documents (which crashed reranked live evals and skewed live metrics).
- Byte-identical source files (which share one content-addressed `doc_id`) no
  longer duplicate chunk records / inflate `chunk_count`, and a golden case that
  names any one of the identical paths now scores correctly instead of `0`.
- Statistical gate: the bootstrap CI is computed over stable queries only, so
  near-tie serving noise can no longer turn a gate red (the never-flaky
  invariant); the excluded near-tie count is surfaced in the gate details.
- SDK `Recorder`: `log_embeddings` raises on a conflicting model/param instead of
  silently dropping vectors; `log_chunks` de-duplicates byte-identical documents.
- Phase-0 arm checkpoints are content-addressed by diff id, so re-runs no longer
  reuse stale arm evals from a different corpus/config.
- Retrieval replays Recorder-logged candidates for a bespoke rerank stage instead
  of crashing every eval; managed-mode unknown reranker tools get a clear error.

[Unreleased]: https://github.com/skundu42/recallops/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/skundu42/recallops/releases/tag/v0.1.0
