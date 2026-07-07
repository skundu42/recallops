# Publishing / Release Runbook

This is the maintainer runbook for cutting a RecallOps release and publishing it
to PyPI. It assumes the build backend and metadata already in `pyproject.toml`
(hatchling, Apache-2.0, `py.typed`, Python 3.11 to 3.13) and that `twine check`
passes on the built artifacts.

> Throughout this document, replace the `OWNER` placeholder with the real GitHub
> org/user that owns the repository, everywhere it appears (the repo slug is
> `OWNER/recallops`). The same `OWNER` placeholder is used in `pyproject.toml`
> URLs, `CHANGELOG.md`, `SECURITY.md`, and `CODE_OF_CONDUCT.md`; replace them
> consistently.

## 0. Distribution name

The distribution + import name is **`recallops`**. It was chosen because the
original name `recallkit` was already taken on PyPI (PRD open question #7);
`recallops` was free on PyPI at rename time.

Before the first public release:

- **Re-verify** `recallops` is still available on PyPI (`pip index versions
  recallops` or check <https://pypi.org/project/recallops/>): an unrelated
  project could claim it before you publish.
- Clear the name for **trademark** if you intend to brand around it.
- If you must change it after all, update `name` in `pyproject.toml`, the
  install instructions in `README.md`, and any `pip install` references, but
  leave the `import recallops` package and the `recall` console script untouched.

TestPyPI dry-runs (Section 4) are recommended before the first real publish.

## 1. Trusted Publishing (OIDC): one-time setup

RecallOps publishes to PyPI via **Trusted Publishing** (OpenID Connect), so no
long-lived PyPI API token is ever stored in the repo. GitHub Actions mints a
short-lived OIDC token that PyPI exchanges for an upload grant.

One-time configuration, done by a PyPI project owner:

1. Sign in to PyPI and open the project's **Settings → Publishing** (for a brand
   new name, use **Your projects → Publishing → Add a pending publisher**).
2. Add a **GitHub Actions** trusted publisher with exactly:
   - **Owner / Repository**: `OWNER/recallops`
   - **Workflow name**: `release.yml`
   - **Environment name**: `pypi`
3. Repeat on **TestPyPI** (<https://test.pypi.org>) with the same values plus the
   environment name your test job uses (e.g. `testpypi`) if you want OIDC dry-runs
   there too.

On the GitHub side, the `release.yml` workflow job must:

- run in the GitHub environment named **`pypi`** (matching step 2),
- request `permissions: id-token: write` (required to mint the OIDC token),
- and publish with `pypa/gh-action-pypi-publish` (no username/password/token).

> Managing the `release.yml` workflow itself is out of scope for this doc (the
> workflow files are owner-managed). This runbook only covers the PyPI-side
> configuration and the release procedure. Keep the workflow filename `release.yml`
> and the environment name `pypi` in sync with the trusted-publisher entry above;
> if you rename either, update the PyPI configuration to match or publishing will
> be rejected.

## 2. Pre-release checklist

Before tagging, on a clean checkout of the default branch:

```bash
pip install -e ".[dev]"
ruff check
pytest
python -m build
python -m twine check dist/*
```

All four must pass: lint clean, tests green (the 556-test suite; pgvector live
tests auto-skip unless `RECALL_PG_DSN` is set), a clean `sdist` + wheel build,
and a passing `twine check`.

## 3. Cut a release

1. **Bump the version.** Edit `version` in `pyproject.toml` (e.g. `0.1.0` →
   `0.1.1`). Follow SemVer; while pre-1.0, breaking changes are allowed in a
   minor bump but must be called out in the changelog.
2. **Update `CHANGELOG.md`.** Rename the **Unreleased** section to the new
   version with today's date (`## [X.Y.Z] - YYYY-MM-DD`), leave a fresh empty
   **Unreleased** section above it, and update the comparison links at the bottom
   of the file.
3. **Commit** on a release branch and open a PR:

   ```bash
   git switch -c release/vX.Y.Z
   git commit -am "chore(release): vX.Y.Z"
   ```

   Merge it once CI is green.
4. **Tag** the merge commit and push the tag:

   ```bash
   git tag -a vX.Y.Z -m "vX.Y.Z"
   git push origin vX.Y.Z
   ```

   The tag must be `vX.Y.Z` (leading `v`) and match `pyproject.toml`'s `version`.
5. **Create a GitHub Release** from that tag (title `vX.Y.Z`, body = the
   changelog section for this version). Publishing the GitHub Release is what
   triggers `release.yml`, which builds the artifacts and publishes them to PyPI
   via Trusted Publishing.
6. **Verify** the release landed:

   ```bash
   pip install "recallops==X.Y.Z"
   recall --version
   ```

   (Substitute the final distribution name if it differs from `recallops`, see
   Section 0.)

## 4. Test against TestPyPI first

For any release you are unsure about (and strongly recommended for the first
one), publish to **TestPyPI** before the real thing.

Manual dry-run from a clean build:

```bash
python -m build
python -m twine check dist/*
python -m twine upload --repository testpypi dist/*
```

Then install from TestPyPI in a throwaway virtualenv, pulling runtime
dependencies from real PyPI:

```bash
python -m venv /tmp/rk-test && source /tmp/rk-test/bin/activate
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            "recallops==X.Y.Z"
recall --version
recall scorecard   # quick offline smoke test of the engine
deactivate
```

Note that **TestPyPI is a separate namespace** with the same name-collision
caveat as Section 0, and a given `version` can only be uploaded once per index;
bump or use a local dev suffix if you need to re-test.

## 5. Post-release

- Confirm the new version resolves on PyPI and the GitHub Release notes match
  `CHANGELOG.md`.
- Announce as appropriate.
- Open the next development cycle by leaving the empty **Unreleased** section in
  `CHANGELOG.md` ready for the next change.
