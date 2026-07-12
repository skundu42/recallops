"""Create-or-update a RecallOps report comment on a GitHub pull request.

The composite GitHub Action (action.yml) calls this after ``recall ci``
(phase 1) and ``recall attribute`` (phase 2) so BOTH phases share ONE PR
comment that updates in place (PRD FR-9.1) instead of appending noise on
every push. The comment is re-found via an invisible HTML marker, so the
operation is idempotent across re-runs.

Stdlib only (urllib); authenticates with a token from ``GITHUB_TOKEN``.
GitHub API and network failures exit 1 with a one-line error on stderr.

Usage: python -m recallops.pr_comment --repo OWNER/NAME --pr N \
           --body-file recall-report.md [--slug report]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_API_URL = "https://api.github.com"
_MARKER = "<!-- recallops:{slug} -->"
_PAGE_SIZE = 100
_TIMEOUT_S = 60.0


def marked_body(body: str, slug: str) -> str:
    return f"{_MARKER.format(slug=slug)}\n{body}"


def find_comment_id(comments: list[dict], slug: str) -> int | None:
    marker = _MARKER.format(slug=slug)
    for c in comments:
        if marker in (c.get("body") or ""):
            return int(c["id"])
    return None


def _github_json(method: str, url: str, token: str, body: dict | None = None):
    req = urllib.request.Request(
        url,
        data=None if body is None else json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method=method,
    )
    with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
        return json.loads(resp.read().decode("utf-8"))


def upsert_comment(repo: str, pr: int, body: str, slug: str, token: str,
                   api_url: str = DEFAULT_API_URL) -> dict:
    """POST a new marked comment, or PATCH the existing one bearing the
    slug's marker. Returns {"action": "created"|"updated", "id": ...}."""
    comments: list[dict] = []
    page = 1
    while True:
        batch = _github_json(
            "GET",
            f"{api_url}/repos/{repo}/issues/{pr}/comments?per_page={_PAGE_SIZE}&page={page}",
            token,
        )
        comments.extend(batch)
        if len(batch) < _PAGE_SIZE:
            break
        page += 1
    payload = {"body": marked_body(body, slug)}
    existing = find_comment_id(comments, slug)
    if existing is None:
        created = _github_json(
            "POST", f"{api_url}/repos/{repo}/issues/{pr}/comments", token, payload
        )
        return {"action": "created", "id": int(created["id"])}
    _github_json(
        "PATCH", f"{api_url}/repos/{repo}/issues/comments/{existing}", token, payload
    )
    return {"action": "updated", "id": existing}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m recallops.pr_comment",
        description="Create or update the RecallOps report comment on a PR.",
    )
    parser.add_argument("--repo", required=True, help="OWNER/NAME")
    parser.add_argument("--pr", required=True, type=int)
    parser.add_argument("--body-file", required=True)
    parser.add_argument("--slug", default="report")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    args = parser.parse_args(argv)

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("error: GITHUB_TOKEN is not set", file=sys.stderr)
        return 2
    try:
        body = Path(args.body_file).read_text(encoding="utf-8")
    except OSError as exc:
        print(f"error: cannot read body file {args.body_file}: {exc}", file=sys.stderr)
        return 2
    try:
        result = upsert_comment(args.repo, args.pr, body, args.slug, token,
                                api_url=args.api_url)
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = json.loads(exc.read().decode("utf-8")).get("message", "")
        except Exception:
            pass
        msg = (f"error: GitHub API returned {exc.code} for {args.repo}#{args.pr}: "
               f"{exc.reason}")
        if detail:
            msg += f": {detail}"
        print(msg, file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"error: cannot reach the GitHub API: {exc.reason}", file=sys.stderr)
        return 1
    print(f"{result['action']} comment {result['id']} on {args.repo}#{args.pr}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
