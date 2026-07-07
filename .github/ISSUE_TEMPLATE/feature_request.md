---
name: Feature request
about: Suggest an enhancement or new capability for RecallOps
title: "[feat] "
labels: ["enhancement", "triage"]
assignees: []
---

<!--
Please search existing issues and discussions first to avoid duplicates.
Keep in mind what RecallOps is (retrieval regression testing with verified
root-cause attribution for RAG pipelines) and what it is not (a vector database,
an observability/tracing tool, or an eval-metrics library; it integrates with
those).
-->

## Problem / motivation

What problem are you trying to solve? What can't you do today, or what is
painful? Concrete scenarios and real pipelines help a lot.

## Proposed solution

What you'd like RecallOps to do. If it's a new CLI command or flag, sketch the
interface:

```bash
recall <command> --new-flag ...
```

## Alternatives considered

Other approaches, workarounds you're using today, or existing commands that
partially cover this.

## Scope / fit

- [ ] This fits RecallOps's mission (versioning the ingestion path, testing
      retrieval in CI, or attributing regressions to causes).
- [ ] This is not primarily a job for a vector DB, an observability tool, or an
      eval-metrics library.

### Invariant impact

Would this touch attribution, gating, or reproducibility? If so, describe how it
preserves the core invariants: no cause labeled `verified` without a confirming
counterfactual arm, narratives constrained to structured evidence, deterministic
content-addressed results, and cost-gated provider calls. Features that would
weaken these are unlikely to be accepted.

## Additional context

Links, prior art, references to the PRD/docs, or anything else.
