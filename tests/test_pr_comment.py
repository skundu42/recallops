from __future__ import annotations

import pytest  # noqa: F401

from recallops import pr_comment as pc


def test_marked_body_and_find():
    body = pc.marked_body("## Report", "report")
    assert body.startswith("<!-- recallops:report -->")
    comments = [
        {"id": 1, "body": "unrelated"},
        {"id": 2, "body": pc.marked_body("old", "report")},
        {"id": 3, "body": pc.marked_body("other slug", "deep")},
    ]
    assert pc.find_comment_id(comments, "report") == 2
    assert pc.find_comment_id(comments, "deep") == 3
    assert pc.find_comment_id(comments, "absent") is None


def _transport(pages, log):
    """Fake _github_json: serves list pages for GET, records writes."""
    def fake(method, url, token, body=None):
        log.append((method, url, body))
        if method == "GET":
            page = int(url.split("&page=")[1])
            return pages.get(page, [])
        return {"id": 99}
    return fake


def test_upsert_creates_when_absent(monkeypatch):
    log = []
    monkeypatch.setattr(pc, "_github_json", _transport({1: []}, log))
    out = pc.upsert_comment("o/r", 5, "## Report", "report", token="t")
    assert out["action"] == "created"
    methods = [m for m, _, _ in log]
    assert methods == ["GET", "POST"]
    post_url = log[-1][1]
    assert post_url.endswith("/repos/o/r/issues/5/comments")
    assert log[-1][2]["body"].startswith("<!-- recallops:report -->")


def test_upsert_updates_in_place(monkeypatch):
    log = []
    existing = [{"id": 42, "body": pc.marked_body("old", "report")}]
    monkeypatch.setattr(pc, "_github_json", _transport({1: existing}, log))
    out = pc.upsert_comment("o/r", 5, "new report", "report", token="t")
    assert out == {"action": "updated", "id": 42}
    method, url, body = log[-1]
    assert method == "PATCH"
    assert url.endswith("/repos/o/r/issues/comments/42")
    assert "new report" in body["body"]


def test_upsert_paginates(monkeypatch):
    log = []
    page1 = [{"id": i, "body": "x"} for i in range(100)]
    page2 = [{"id": 200, "body": pc.marked_body("old", "report")}]
    monkeypatch.setattr(pc, "_github_json", _transport({1: page1, 2: page2}, log))
    out = pc.upsert_comment("o/r", 5, "new", "report", token="t")
    assert out["action"] == "updated"
    assert out["id"] == 200


def test_main_requires_token(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    f = tmp_path / "r.md"
    f.write_text("hi", encoding="utf-8")
    rc = pc.main(["--repo", "o/r", "--pr", "5", "--body-file", str(f)])
    assert rc == 2
    assert "GITHUB_TOKEN" in capsys.readouterr().err


def test_main_happy_path(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("GITHUB_TOKEN", "t")
    calls = []
    monkeypatch.setattr(pc, "upsert_comment",
                        lambda repo, pr, body, slug, token, api_url=pc.DEFAULT_API_URL:
                        calls.append((repo, pr, body, slug)) or {"action": "created", "id": 1})
    f = tmp_path / "r.md"
    f.write_text("## Report", encoding="utf-8")
    rc = pc.main(["--repo", "o/r", "--pr", "5", "--body-file", str(f)])
    assert rc == 0
    assert calls == [("o/r", 5, "## Report", "report")]
    assert "created" in capsys.readouterr().out


def test_main_reports_api_errors_cleanly(monkeypatch, tmp_path, capsys):
    import urllib.error

    monkeypatch.setenv("GITHUB_TOKEN", "t")

    def boom(repo, pr, body, slug, token, api_url=pc.DEFAULT_API_URL):
        raise urllib.error.HTTPError("u", 401, "Unauthorized", None, None)

    monkeypatch.setattr(pc, "upsert_comment", boom)
    f = tmp_path / "r.md"
    f.write_text("hi", encoding="utf-8")
    rc = pc.main(["--repo", "o/r", "--pr", "5", "--body-file", str(f)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "401" in err and "o/r#5" in err
