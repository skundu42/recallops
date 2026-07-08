# Publishing / Release Runbook

This is the maintainer runbook for cutting a RecallOps release and publishing it
to PyPI. It assumes the build backend and metadata already in `pyproject.toml`
(hatchling + **hatch-vcs tag-derived versioning**, Apache-2.0, `py.typed`,
Python 3.11 to 3.13) and that `twine check` passes on the built artifacts.

> The repo slug is `skundu42/recallops`. The only remaining `OWNER` placeholders
> are the contact **email** addresses in `SECURITY.md` (`security@OWNER`) and
> `CODE_OF_CONDUCT.md` (`conduct@OWNER`); replace those with real addresses.

## Versioning: the tag is the source of truth

The version is **derived from the git tag** by hatch-vcs — there is no `version`
field to edit in `pyproject.toml`, and `recallops.__version__` (and `recall
--version`) read it from the installed package metadata. Pushing a `vX.Y.Z` tag
is what sets the version; a build from any other commit is a dev version
(`X.Y.Z.devN+g<sha>`). Never hand-edit a version anywhere.

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
   - **Owner / Repository**: `skundu42/recallops`
   - **Workflow name**: `release.yml`
   - **Environment name**: `pypi`
3. Repeat on **TestPyPI** (<https://test.pypi.org>) with the same values plus the
   environment name your test job uses (e.g. `testpypi`) if you want OIDC dry-runs
   there too.

On the GitHub side, the `release.yml` workflow job must:

- run in the GitHub environment named **`pypi`** (matching step 2),
- request `permissions: id-token: write` (required to mint the OIDC token),
- and publish with `pypa/gh-action-pypi-publish` (no username/password/token).

> The `release.yml` workflow is already wired for this: its publish job runs in
> the `pypi` environment with `id-token: write` and publishes via
> `pypa/gh-action-pypi-publish` (no token). Keep the workflow filename
> `release.yml` and the environment name `pypi` in sync with the trusted-publisher
> entry above; the OIDC identity uses the canonical repo slug `skundu42/recallops`,
> so if the repo is renamed, update the PyPI registration or uploads are rejected.

## 2. Pre-release checklist

Before tagging, on a clean checkout of the default branch:

```bash
pip install -e ".[dev]"
ruff check
pytest
python -m build
python -m twine check dist/*
```

All four must pass: lint clean, tests green (pgvector live tests auto-skip
unless `RECALL_PG_DSN` is set), a clean `sdist` + wheel build, and a passing
`twine check`. Because the version is tag-derived, a build off an untagged
working tree reports a dev version (`X.Y.Z.devN+...`) — that is expected here;
the clean version only appears on the tagged commit the release workflow builds.

## 3. Cut a release

There is **no version to bump** — the tag sets it. On the default branch, with
CI green:

1. **Update `CHANGELOG.md`.** Rename the **Unreleased** section to the new
   version with today's date (`## [X.Y.Z] - YYYY-MM-DD`), leave a fresh empty
   **Unreleased** section above it, and update the comparison links at the bottom
   of the file. Commit and merge this (a small `docs(changelog)` PR, or directly
   if you have push rights). The workflow reads this section for the release notes.
2. **Tag and push** the release commit:

   ```bash
   git tag -a vX.Y.Z -m "vX.Y.Z"     # leading `v`; SemVer (pre-1.0: 0.x)
   git push origin vX.Y.Z
   ```

   That is the entire release action. The pushed tag triggers `release.yml`,
   which builds the artifacts, verifies the built version equals the tag,
   publishes to PyPI via Trusted Publishing, and **creates the GitHub Release**
   from the matching `CHANGELOG.md` section. If the `pypi` environment requires a
   reviewer, approve the deployment when prompted.
3. **Verify** the release landed:

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
caveat as Section 0, and a given `version` can only be uploaded once per index.
With tag-derived versioning this is easy: an untagged dry-run build already
carries a unique dev suffix (`X.Y.Z.devN+g<sha>.d<date>`), so repeated TestPyPI
uploads never collide. To rehearse the exact release version instead, build on
the `vX.Y.Z` tag (`git checkout vX.Y.Z`).

## 5. Post-release

- Confirm the new version resolves on PyPI and the GitHub Release notes match
  `CHANGELOG.md`.
- Announce as appropriate.
- Open the next development cycle by leaving the empty **Unreleased** section in
  `CHANGELOG.md` ready for the next change.
