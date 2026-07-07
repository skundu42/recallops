# RecallOps documentation

RecallOps is retrieval regression testing with **verified counterfactual
root-cause attribution** for RAG pipelines. It versions the ingestion path,
tests retrieval in CI with a never-flaky statistical gate, and attributes
regressions to a cause only when a counterfactual arm reverting that factor
actually recovers the query.

## Start here

- **New to RecallOps?** Read the top-level [`README.md`](../README.md) for the
  positioning and the 5-minute quickstart, then run
  [`examples/quickstart.sh`](../examples/quickstart.sh) to catch and explain a
  regression end-to-end at $0 (no API keys).
- **Want to understand how it works?** Read
  [`ARCHITECTURE.md`](../ARCHITECTURE.md): the four-layer design and the key
  invariants.
- **Looking for a specific command?** Jump to the [CLI reference](cli.md).

## Contents

| Document | What it covers |
|---|---|
| [`../README.md`](../README.md) | Project overview, positioning, install, and the 5-minute quickstart (catch a regression, see a verified cause). |
| [`../ARCHITECTURE.md`](../ARCHITECTURE.md) | The layered design (provenance store + retrieval engine, funnel attribution, counterfactual ablation, statistical gating, renderer), the per-module map, the key invariants, and the adapter contract. |
| [`cli.md`](cli.md) | Reference for every `recall` command, grouped by workflow: purpose and key flags for each. |
| [`sizing.md`](sizing.md) | Storage sizing: the fp16 embedding footprint, the ≤ 2.5× corpus-text overhead target, and how retention (`recall gc`) bounds it. |
| [`phase0-validation.md`](phase0-validation.md) | The Phase-0 real-project go/no-go runbook: provision pgvector, supply a corpus + golden set + a real breaking change, run `recall phase0`, and interpret the result. |

## Examples

| Path | What it is |
|---|---|
| [`../examples/quickstart.sh`](../examples/quickstart.sh) | End-to-end Journey J1 script: ingest → golden set → green eval → chunker regression → **verified** cause → revert to green. Runs offline at $0. |
| [`../examples/corpus/`](../examples/corpus/) | The bundled 12-document markdown corpus used by the quickstart, scorecard, and docs. |
| [`../examples/github-action/recall-ci.yml`](../examples/github-action/recall-ci.yml) | A drop-in GitHub Actions workflow implementing the two-phase CI gate (Phase 1 blocking eval + diff + funnel; Phase 2 async deep attribution). |
