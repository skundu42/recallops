from __future__ import annotations

from pathlib import Path

import pytest

from recallops import hashing
from recallops.pipeline.parsers import ParsedDoc, parse, parser_stage_spec


def test_text_v1_normalizes_crlf_and_strips_trailing_ws():
    raw = b"# Title\r\nline one   \r\nline two\t\r\n\r\nend"
    doc = parse("a.md", raw, tool="text-v1")
    assert doc.text == "# Title\nline one\nline two\n\nend"


def test_text_v1_normalizes_lone_cr():
    assert parse("a.md", b"one\rtwo\r\nthree").text == "one\ntwo\nthree"


def test_text_v1_preserves_leading_ws_and_content():
    raw = b"  indented\nplain\n"
    doc = parse("a.md", raw)
    assert doc.text == "  indented\nplain\n"


def test_parse_default_tool_is_text_v1():
    raw = b"hello  \nworld"
    assert parse("a.md", raw).text == parse("a.md", raw, tool="text-v1").text


def test_parsed_doc_fields():
    raw = b"content"
    doc = parse("dir/file.md", raw)
    assert isinstance(doc, ParsedDoc)
    assert doc.source_path == "dir/file.md"
    assert doc.raw_hash == hashing.doc_id(raw)


def test_raw_hash_depends_on_bytes_not_path():
    raw = b"same bytes"
    assert parse("x.md", raw).raw_hash == parse("y.md", raw).raw_hash
    assert parse("x.md", raw).raw_hash != parse("x.md", b"other bytes").raw_hash


def test_markdown_v2_strips_heading_markers():
    raw = b"# Top\n\n## Section two\n\nBody text here.\n\n### Deep\nmore\n"
    doc = parse("a.md", raw, tool="markdown-v2")
    assert doc.text == "Top\n\nSection two\n\nBody text here.\n\nDeep\nmore\n"


def test_markdown_v2_strips_emphasis_markers():
    raw = b"This is *important* and **very bold** and _quiet_ text.\n"
    doc = parse("a.md", raw, tool="markdown-v2")
    assert doc.text == "This is important and very bold and quiet text.\n"


def test_markdown_v2_handles_nested_emphasis():
    raw = b"a **_both_** b\n"
    assert parse("a.md", raw, tool="markdown-v2").text == "a both b\n"


def test_markdown_v2_keeps_snake_case_identifiers():
    raw = b"call the rate_limit_config helper\n"
    doc = parse("a.md", raw, tool="markdown-v2")
    assert "rate_limit_config" in doc.text


def test_markdown_v2_differs_from_text_v1_on_markdown_input():
    raw = b"# Heading\n\nplain body\n"
    assert parse("a.md", raw, tool="markdown-v2").text != parse("a.md", raw, tool="text-v1").text


def test_markdown_v2_differs_on_every_example_corpus_doc(corpus_dir: Path):
    files = sorted(corpus_dir.rglob("*.md"))
    assert len(files) >= 12
    for f in files:
        raw = f.read_bytes()
        t1 = parse(str(f), raw, tool="text-v1").text
        t2 = parse(str(f), raw, tool="markdown-v2").text
        assert t2 != t1, f"markdown-v2 should change {f}"
        assert t2.count("#") < t1.count("#")


def test_parse_deterministic(corpus_dir: Path):
    f = sorted(corpus_dir.rglob("*.md"))[0]
    raw = f.read_bytes()
    for tool in ("text-v1", "markdown-v2"):
        assert parse(str(f), raw, tool) == parse(str(f), raw, tool)


def test_parse_unknown_tool_raises():
    with pytest.raises(ValueError):
        parse("a.md", b"x", tool="pdf-v9")


def test_parser_stage_spec():
    spec = parser_stage_spec("text-v1")
    assert spec.id == "parse"
    assert spec.tool == "text-v1"
    assert spec.version == "1"
    assert spec.params == {}
    md = parser_stage_spec("markdown-v2")
    assert md.tool == "markdown-v2"
    assert md.params_hash == spec.params_hash


def test_parser_stage_spec_unknown_tool_raises():
    with pytest.raises(ValueError):
        parser_stage_spec("nope")
