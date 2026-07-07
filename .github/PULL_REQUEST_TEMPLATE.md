<!--
Thanks for contributing to RecallOps! Please read CONTRIBUTING.md first.
Keep PRs focused on one logical change. Link the issue this closes.
-->

## Summary

What does this PR change, and why?

Closes #<!-- issue number, if any -->

## Type of change

- [ ] Bug fix (non-breaking)
- [ ] New feature (non-breaking)
- [ ] Breaking change (behavior/API/output changes)
- [ ] Docs / tooling only

## Checklist

- [ ] **Tests added/updated and passing** (`pytest` is green on 3.11 to 3.13; a bug
      fix includes a regression test that fails without the fix).
- [ ] **Lint clean** (`ruff check` passes with no new warnings).
- [ ] **Docs updated** where behavior, flags, or interfaces changed (README /
      `docs/` / docstrings).
- [ ] **CHANGELOG updated**: a bullet under **Unreleased** in `CHANGELOG.md` for
      any user-facing change.
- [ ] **Engine invariants preserved** (see below).

## Invariants preserved

Confirm the change does not weaken any core invariant (see CONTRIBUTING.md):

- [ ] **No verified cause without a confirming arm**: nothing is labeled
      `verified` unless a counterfactual arm reverting that single factor
      actually recovers the query and cites its `arm_id`; everything else stays a
      labeled hypothesis. (Embedding-model swaps remain characterization, not a
      verified mechanism.)
- [ ] **Narratives stay evidence-bound**: no narrative claim is emitted that is
      not backed by structured evidence in the same report.
- [ ] **Determinism & content-addressing**: identical inputs still produce
      byte-identical manifests/ids; stochastic steps take an explicit seed; an
      identical pipeline re-run performs zero embedding calls.
- [ ] **Cost is gated**: any provider-billed operation prints an estimate and
      requires explicit approval; the local provider stays $0.
- [ ] Not applicable: this PR does not touch attribution, gating, ingestion, or
      reproducibility.

## How was this tested?

Commands run, adapters exercised (local / pgvector), and any manual verification.
If this touches the pgvector path, note whether the live tests were run with
`RECALL_PG_DSN` set.

## Notes for reviewers

Anything else reviewers should know: trade-offs, follow-ups, or areas you want
scrutiny on.
