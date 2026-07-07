from __future__ import annotations

import re
from pathlib import Path

import pytest

from recallops import hashing
from recallops.pipeline.chunkers import Span, chunk_doc, get_chunker
from recallops.pipeline.parsers import parse

FIXED = "recall.chunkers.fixed_token"
HEADING = "recall.chunkers.markdown_heading"
SENTENCE = "recall.chunkers.sentence"

ALL_CONFIGS = [
    (FIXED, {"max_tokens": 40, "overlap": 8}),
    (FIXED, {"max_tokens": 60, "overlap": 0}),
    (HEADING, {"max_tokens": 120, "overlap": 20}),
    (HEADING, {"max_tokens": 800, "overlap": 120}),
    (SENTENCE, {"max_sentences": 3}),
]


def tokens_of(text: str) -> list[str]:
    return text.split()


def assert_covers_non_ws(text: str, spans: list[Span]) -> None:
    covered = [False] * len(text)
    for s in spans:
        for i in range(s.start, s.end):
            covered[i] = True
    missing = [i for i, ch in enumerate(text) if not ch.isspace() and not covered[i]]
    assert missing == [], f"non-whitespace chars not covered: {missing[:10]}"


@pytest.mark.parametrize("tool,params", ALL_CONFIGS)
def test_span_text_identity_on_example_corpus(corpus_dir: Path, tool: str, params: dict):
    files = sorted(corpus_dir.rglob("*.md"))
    assert len(files) >= 12
    for f in files:
        text = parse(str(f), f.read_bytes()).text
        records = chunk_doc("doc_x", text, tool, params, "parse", "chunk")
        assert records, f"no chunks for {f}"
        for rec in records:
            assert text[rec.span_start:rec.span_end] == rec.text
        assert [r.ordinal for r in records] == list(range(len(records)))
        starts = [r.span_start for r in records]
        assert starts == sorted(starts)
        assert len(set(starts)) == len(starts)
        assert_covers_non_ws(text, [Span(r.span_start, r.span_end) for r in records])


def test_fixed_token_window_and_overlap_counts():
    text = " ".join(f"tok{i}" for i in range(10))
    chunker = get_chunker(FIXED, {"max_tokens": 4, "overlap": 1})
    spans = chunker.chunk(text)
    chunks = [tokens_of(text[s.start:s.end]) for s in spans]
    assert chunks == [
        ["tok0", "tok1", "tok2", "tok3"],
        ["tok3", "tok4", "tok5", "tok6"],
        ["tok6", "tok7", "tok8", "tok9"],
    ]
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev[-1] == nxt[0]


def test_fixed_token_no_overlap_partitions_tokens():
    text = " ".join(f"w{i}" for i in range(7))
    spans = get_chunker(FIXED, {"max_tokens": 3, "overlap": 0}).chunk(text)
    chunks = [tokens_of(text[s.start:s.end]) for s in spans]
    assert chunks == [["w0", "w1", "w2"], ["w3", "w4", "w5"], ["w6"]]


def test_fixed_token_short_text_single_chunk():
    text = "only three tokens"
    spans = get_chunker(FIXED, {"max_tokens": 50, "overlap": 10}).chunk(text)
    assert len(spans) == 1
    assert text[spans[0].start:spans[0].end] == "only three tokens"


def test_fixed_token_spans_trimmed_to_tokens():
    text = "   padded    text   "
    spans = get_chunker(FIXED, {"max_tokens": 10, "overlap": 0}).chunk(text)
    assert len(spans) == 1
    chunk = text[spans[0].start:spans[0].end]
    assert chunk == "padded    text"


def test_fixed_token_empty_and_whitespace_only():
    chunker = get_chunker(FIXED, {"max_tokens": 5, "overlap": 1})
    assert chunker.chunk("") == []
    assert chunker.chunk("   \n\n  ") == []


def test_markdown_heading_one_chunk_per_short_section_with_heading_inside():
    text = (
        "# Alpha Widget\n\n"
        "## Setup\n\nInstall the widget with the bootstrap command.\n\n"
        "## Troubleshooting\n\nIf the checksum fails, retry the bootstrap.\n"
    )
    spans = get_chunker(HEADING, {"max_tokens": 100, "overlap": 0}).chunk(text)
    chunks = [text[s.start:s.end] for s in spans]
    assert len(chunks) == 3
    assert chunks[0] == "# Alpha Widget"
    assert chunks[1].startswith("## Setup")
    assert "bootstrap command" in chunks[1]
    assert chunks[2].startswith("## Troubleshooting")
    assert "retry the bootstrap" in chunks[2]


def test_markdown_heading_preamble_before_first_heading():
    text = "intro line before any heading\n\n# First\n\nbody\n"
    spans = get_chunker(HEADING, {"max_tokens": 100, "overlap": 0}).chunk(text)
    chunks = [text[s.start:s.end] for s in spans]
    assert len(chunks) == 2
    assert chunks[0] == "intro line before any heading"
    assert chunks[1].startswith("# First")


def test_markdown_heading_subsplits_long_section_keeping_heading_in_first():
    body = " ".join(f"word{i}" for i in range(50))
    text = f"## Long Section\n\n{body}\n"
    spans = get_chunker(HEADING, {"max_tokens": 20, "overlap": 5}).chunk(text)
    chunks = [text[s.start:s.end] for s in spans]
    assert len(chunks) > 1
    assert chunks[0].startswith("## Long Section")
    for c in chunks:
        assert len(tokens_of(c)) <= 20
    assert_covers_non_ws(text, spans)


def test_markdown_heading_no_headings_falls_back_to_token_split():
    text = " ".join(f"word{i}" for i in range(30))
    spans = get_chunker(HEADING, {"max_tokens": 10, "overlap": 0}).chunk(text)
    assert len(spans) == 3
    assert_covers_non_ws(text, spans)


def test_markdown_heading_on_corpus_headings_stay_in_chunk(corpus_dir: Path):
    f = corpus_dir / "billing" / "refunds.md"
    text = parse(str(f), f.read_bytes()).text
    spans = get_chunker(HEADING, {"max_tokens": 800, "overlap": 120}).chunk(text)
    chunks = [text[s.start:s.end] for s in spans]
    headings = [m.group(0).rstrip() for m in re.finditer(r"^#{1,6} .*$", text, re.MULTILINE)]
    assert len(chunks) == len(headings)
    for chunk, heading in zip(chunks, headings):
        assert chunk.startswith(heading)


def test_sentence_chunker_groups_sentences():
    text = "One is first. Two follows! Three asks? Four here. Five ends."
    spans = get_chunker(SENTENCE, {"max_sentences": 2}).chunk(text)
    chunks = [text[s.start:s.end] for s in spans]
    assert chunks == [
        "One is first. Two follows!",
        "Three asks? Four here.",
        "Five ends.",
    ]


def test_sentence_chunker_treats_paragraph_break_as_boundary():
    text = "# Heading Without Period\n\nA real sentence follows. And another one."
    spans = get_chunker(SENTENCE, {"max_sentences": 1}).chunk(text)
    chunks = [text[s.start:s.end] for s in spans]
    assert chunks[0] == "# Heading Without Period"
    assert chunks[1] == "A real sentence follows."
    assert chunks[2] == "And another one."


def test_sentence_chunker_keeps_wrapped_lines_in_one_sentence():
    text = "Refunds are prorated to the nearest\nwhole month remaining. Next sentence."
    spans = get_chunker(SENTENCE, {"max_sentences": 1}).chunk(text)
    chunks = [text[s.start:s.end] for s in spans]
    assert chunks[0] == "Refunds are prorated to the nearest\nwhole month remaining."
    assert chunks[1] == "Next sentence."


def test_chunk_doc_records_are_content_addressed():
    text = "# T\n\nalpha beta gamma delta epsilon zeta eta theta.\n"
    records = chunk_doc("doc_abc", text, FIXED, {"max_tokens": 4, "overlap": 1}, "parse", "chunk")
    assert records
    for rec in records:
        assert rec.doc_id == "doc_abc"
        assert rec.chunk_id == hashing.chunk_id("doc_abc", rec.span_start, rec.span_end, rec.text)
        assert rec.text_hash == hashing.text_hash(rec.text)
        assert rec.parse_stage_id == "parse"
        assert rec.chunk_stage_id == "chunk"


@pytest.mark.parametrize("tool,params", ALL_CONFIGS)
def test_chunk_doc_deterministic(corpus_dir: Path, tool: str, params: dict):
    f = corpus_dir / "security" / "sso.md"
    text = parse(str(f), f.read_bytes()).text
    a = chunk_doc("doc_1", text, tool, params, "parse", "chunk")
    b = chunk_doc("doc_1", text, tool, params, "parse", "chunk")
    assert [r.to_dict() for r in a] == [r.to_dict() for r in b]


@pytest.mark.parametrize("tool,params", ALL_CONFIGS)
def test_span_text_identity_holds_on_markdown_v2_parsed_text(corpus_dir: Path, tool: str, params: dict):
    f = corpus_dir / "api" / "auth.md"
    text = parse(str(f), f.read_bytes(), tool="markdown-v2").text
    records = chunk_doc("doc_md2", text, tool, params, "parse", "chunk")
    assert records
    for rec in records:
        assert text[rec.span_start:rec.span_end] == rec.text
    assert_covers_non_ws(text, [Span(r.span_start, r.span_end) for r in records])


def test_chunk_doc_empty_text():
    assert chunk_doc("doc_1", "", FIXED, {"max_tokens": 5, "overlap": 0}, "p", "c") == []


def test_get_chunker_unknown_tool_raises():
    with pytest.raises(ValueError):
        get_chunker("recall.chunkers.semantic", {})


def test_get_chunker_invalid_params_raise():
    with pytest.raises(ValueError):
        get_chunker(FIXED, {"max_tokens": 0, "overlap": 0})
    with pytest.raises(ValueError):
        get_chunker(FIXED, {"max_tokens": 5, "overlap": -1})
    with pytest.raises(ValueError):
        get_chunker(FIXED, {"max_tokens": 5, "overlap": 5})
    with pytest.raises(ValueError):
        get_chunker(SENTENCE, {"max_sentences": 0})
