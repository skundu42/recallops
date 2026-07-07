# Contributing to RecallOps

Thanks for your interest in improving RecallOps. This document covers local
setup, how we run tests and lint, our branch/PR conventions, and, most
importantly, the engine invariants that every change must preserve.

By participating you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md).

## Ground rules

- **New code ships with tests.** Bug fixes come with a regression test that
  fails before the fix; features come with coverage of the new behavior.
- **Every change stays `ruff`-clean.** CI runs `ruff check`; a red lint blocks
  the merge.
- **The engine invariants below are not negotiable.** They are the product, and
  a change that weakens them will not be merged even if every test is green.

## Development setup

RecallOps targets Python 3.11, 3.12, and 3.13. Clone the repo and install in
editable mode with the dev extra:

```bash
pip install -e ".[dev]"
# or, with uv:
uv pip install -e ".[dev]"
```

The default stack (local hash embeddings + the built-in exact-KNN adapter) runs
fully offline at $0 (no API keys, no server) so the full unit suite needs no
external services. OpenAI and pgvector are opt-in.

To work on the pgvector adapter, also install the `pg` extra:

```bash
pip install -e ".[dev,pg]"
```

## Running the tests

```bash
pytest
```

The suite is deterministic and network-free by default. The pgvector live tests
**auto-skip** unless `RECALL_PG_DSN` is set, so a plain `pytest` run is green on
any machine.

### Optional: pgvector integration tests

To exercise the pgvector adapter against a real Postgres server, provision one
and export its DSN:

```bash
export RECALL_PG_DSN="postgresql://postgres:recall@localhost:5433/postgres"
pytest
```

[`docs/phase0-validation.md`](docs/phase0-validation.md) has the full pgvector
provisioning steps: a one-line Docker path (`pgvector/pgvector:pg16`) and a
build-from-source path, plus how to create the `vector` extension. When
`RECALL_PG_DSN` is unset the same tests skip automatically; please do not remove
that skip guard.

Tests that would hit a paid provider (OpenAI embeddings) are gated behind an API
key in the same way and must never run (or cost money) in the default suite.

## Linting and formatting

```bash
ruff check
```

The `ruff` config lives in `pyproject.toml` (`select = E4,E7,E9,W,F,I,UP`,
line length 110, target `py311`). Run `ruff check --fix` to auto-apply the
import-sorting and pyupgrade fixes. Keep imports sorted and code within the
110-column limit.

## Branch and PR conventions

1. **Branch off the default branch.** Do not commit directly to it. Use a short,
   descriptive branch name, optionally prefixed by type:
   `fix/pgvector-collection-namespace`, `feat/compare-rerankers`,
   `docs/publishing-runbook`.
2. **Keep PRs focused.** One logical change per PR. Unrelated refactors belong
   in their own PR.
3. **Fill in the PR template.** The checklist
   ([`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md))
   is a gate, not a formality: tests added/passing, `ruff` clean, docs updated,
   invariants preserved, and `CHANGELOG.md` updated under **Unreleased**.
4. **CI must be green** before review. That means `pytest` passes on 3.11 to 3.13
   and `ruff check` is clean.
5. **Link the issue** the PR closes (`Closes #123`) when there is one.

## Commit style

We use [Conventional Commits](https://www.conventionalcommits.org):

```
type(scope): short imperative summary

Optional body explaining what and why (not how). Wrap at ~72 columns.

Closes #123
```

Common types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `build`,
`ci`, `chore`. Scopes track the module or subsystem, e.g. `ingest`, `funnel`,
`ablation`, `gating`, `pgvector`, `cli`. Write the summary in the imperative
mood ("add", not "added"). Keep the subject line under ~72 characters.

## User-facing changes: update the CHANGELOG

Any change a user would notice (a new flag, a changed default, a bug fix, a new
CLI command) gets a bullet under the **Unreleased** section of
[`CHANGELOG.md`](CHANGELOG.md), in Keep a Changelog format (`Added`, `Changed`,
`Fixed`, `Removed`, `Deprecated`, `Security`). Purely internal refactors that
change no observable behavior do not need an entry.

## Engine invariants (must be preserved)

RecallOps's credibility rests on a small set of invariants. A change that breaks
any of them is a regression in the product, not just the code, and will be
rejected. If you believe an invariant needs to change, open an issue to discuss
it first; do not smuggle it into a feature PR.

### 1. No verified cause without a confirming arm

This is the core promise (FR-8.1, the "no explanation without evidence"
principle). A cause may be labeled **`verified`** *only* if a counterfactual arm
that reverts that single factor (with all other factors held at the post-change
state B) actually recovers the query to within its original rank ±1, and the
emitted cause cites that confirming `arm_id`. Everything else is a labeled
**hypothesis** (`unverified`) and must never be merged into the verified set or
presented as fact.

- Embedding-**model** swaps are the documented exception (FR-8.4): there is no
  mechanistic arm for a black-box model change, so the engine emits per-tag
  **characterization**, not a verified mechanism. Do not relabel
  characterization as verification.
- Any sampled/approximate attribution (Tier-M/L behavior) must carry its
  "unverified at corpus scale" label into every output. Degradation is always
  labeled, never silent.

If you touch `ablation.py`, `confirm.py`, `funnel.py`, `narrative.py`, or
`report.py`, re-read this section and make sure the confirmation rule still
holds. Narrative text must never assert a claim that is not backed by structured
evidence in the same report.

### 2. Determinism and content-addressing

Identical inputs must produce byte-identical manifests and identical
content-addressed ids (docs, chunks, embeddings, snapshots, arms). Concretely:

- Every stochastic step takes an **explicit seed**; never seed from wall-clock,
  PID, hash randomization, or unordered iteration.
- Ids are derived from content via `hashing.py`. Do not fold non-deterministic
  or environment-dependent data (timestamps, absolute paths, dict ordering)
  into a hashed payload.
- Re-running an identical pipeline must perform **zero** embedding calls and
  reuse every chunk and embedding.
- Embeddings are stored fp16 columnar; shadow scoring is exact. Do not introduce
  a code path where a config change silently re-embeds unchanged chunks or where
  ANN noise is mistaken for a real regression.

There are tests that assert snapshot-id stability and zero re-embedding; if your
change moves those numbers, that is a signal to stop and understand why, not to
update the expected values.

### 3. Cost is always gated and honest

Any provider-billed operation must print a cost estimate and require explicit
approval (`--yes` or `--max-cost`). The local provider is $0 and auto-approves.
Never add a code path that bills a provider without that gate.

## Reporting bugs and requesting features

Use the issue templates under
[`.github/ISSUE_TEMPLATE`](.github/ISSUE_TEMPLATE). For anything security- or
data-exposure-related, follow [SECURITY.md](SECURITY.md) instead of opening a
public issue.

## License

By contributing, you agree that your contributions are licensed under the
project's [Apache-2.0](LICENSE) license.
