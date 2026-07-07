---
name: Bug report
about: Report a defect in RecallOps so we can reproduce and fix it
title: "[bug] "
labels: ["bug", "triage"]
assignees: []
---

<!--
Before filing: please search existing issues first.
For anything security- or data-exposure-related, do NOT open a public issue:
follow SECURITY.md instead.
-->

## Summary

A clear, concise description of the bug.

## Steps to reproduce

Exact commands or a minimal script. Prefer the offline default stack (local hash
embeddings + built-in exact-KNN adapter) if you can reproduce it there.

```bash
recall init --source ...
recall ingest ...
# ...
```

## Expected behavior

What you expected to happen.

## Actual behavior

What actually happened. Paste the full error / traceback and any relevant CLI
output inside a code block.

```
<paste output here>
```

## Attribution / determinism (if applicable)

<!-- Fill this in only if the bug is about attribution results or reproducibility. -->

- [ ] A cause was labeled `verified` but no confirming arm was cited / the arm
      does not recover the query.
- [ ] A narrative claim was not backed by structured evidence in the report.
- [ ] Identical inputs produced different snapshot ids / manifests / results.
- [ ] An identical pipeline re-run performed embedding calls (expected: zero).

If so, include the snapshot ids, dataset name, and (if safe to share) the JSON
report (`--format json`).

## Environment

- RecallOps version: <!-- `recall --version` or the commit SHA -->
- Python version: <!-- 3.11 / 3.12 / 3.13 -->
- OS: <!-- macOS 14 / Ubuntu 22.04 / ... -->
- Adapter: <!-- local (default) / pgvector -->
- Embedding provider: <!-- local (default) / openai / other -->
- Installed via: <!-- pip / uv / from source -->

## Additional context

Anything else: corpus size/tier, config snippet (`recall.yaml`), screenshots.
Please redact any secrets, document text, or proprietary data.
