# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the project is pre-1.0 (`0.x`), minor versions may include breaking
changes; these will always be called out under **Changed** or **Removed**.

## [Unreleased]

_Nothing yet. Add user-facing changes here under the appropriate heading
(`Added`, `Changed`, `Fixed`, `Removed`, `Deprecated`, `Security`)._

## [0.1.0] - 2026-07-07

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
- **CLI**: 16 commands: `init`, `ingest`, `snapshot`, `dataset`, `eval`,
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
- **Packaging**: Apache-2.0 licensed, ships `py.typed`, supports Python
  3.11 to 3.13, PyPI-metadata-valid (hatchling backend).

[Unreleased]: https://github.com/OWNER/recallops/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/OWNER/recallops/releases/tag/v0.1.0
