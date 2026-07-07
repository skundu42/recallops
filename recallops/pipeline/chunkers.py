"""Chunkers producing char spans into the parsed text (FR-1.3).

Invariant: for every produced record, ``text[span_start:span_end] == record.text``
exactly, spans are what make chunk-fate alignment (FR-5.3) possible. Spans are
ordered by strictly increasing start and cover the text minus whitespace gaps;
with overlap the trailing tokens of one chunk repeat at the head of the next.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from .. import hashing
from ..models import ChunkRecord

FIXED_TOKEN = "recall.chunkers.fixed_token"
MARKDOWN_HEADING = "recall.chunkers.markdown_heading"
SENTENCE = "recall.chunkers.sentence"
CHUNKER_TOOLS = (FIXED_TOKEN, MARKDOWN_HEADING, SENTENCE)

_TOKEN = re.compile(r"\S+")
_HEADING_LINE = re.compile(r"(?m)^#{1,6}(?:[ \t]|$)")
_TERMINAL = ".!?"


@dataclass(frozen=True)
class Span:
    start: int
    end: int


class Chunker(Protocol):
    def chunk(self, text: str) -> list[Span]: ...


def _token_spans(text: str, offset: int = 0) -> list[Span]:
    return [Span(m.start() + offset, m.end() + offset) for m in _TOKEN.finditer(text)]


def _window_spans(tokens: list[Span], max_tokens: int, overlap: int) -> list[Span]:
    if not tokens:
        return []
    step = max_tokens - overlap
    spans: list[Span] = []
    i = 0
    while True:
        j = min(i + max_tokens, len(tokens))
        spans.append(Span(tokens[i].start, tokens[j - 1].end))
        if j >= len(tokens):
            return spans
        i += step


@dataclass(frozen=True)
class FixedTokenChunker:
    max_tokens: int
    overlap: int

    def chunk(self, text: str) -> list[Span]:
        return _window_spans(_token_spans(text), self.max_tokens, self.overlap)


def _sections(text: str) -> list[tuple[int, int]]:
    starts = [m.start() for m in _HEADING_LINE.finditer(text)]
    if not starts:
        return [(0, len(text))]
    bounds: list[tuple[int, int]] = []
    if starts[0] > 0:
        bounds.append((0, starts[0]))
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        bounds.append((start, end))
    return bounds


@dataclass(frozen=True)
class MarkdownHeadingChunker:
    max_tokens: int
    overlap: int

    def chunk(self, text: str) -> list[Span]:
        spans: list[Span] = []
        for start, end in _sections(text):
            tokens = _token_spans(text[start:end], offset=start)
            spans.extend(_window_spans(tokens, self.max_tokens, self.overlap))
        return spans


def _sentence_spans(text: str) -> list[Span]:
    spans: list[Span] = []
    n = len(text)
    i = 0
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        end: int | None = None
        j = i
        while j < n:
            ch = text[j]
            if ch in _TERMINAL:
                k = j
                while k < n and text[k] in _TERMINAL:
                    k += 1
                if k >= n or text[k].isspace():
                    end = k
                    break
                j = k
                continue
            if ch == "\n" and j + 1 < n and text[j + 1] == "\n":
                end = j
                break
            j += 1
        if end is None:
            end = n
        while end > i and text[end - 1].isspace():
            end -= 1
        spans.append(Span(i, end))
        i = end
    return spans


@dataclass(frozen=True)
class SentenceChunker:
    max_sentences: int

    def chunk(self, text: str) -> list[Span]:
        sentences = _sentence_spans(text)
        return [
            Span(group[0].start, group[-1].end)
            for group in (
                sentences[i:i + self.max_sentences]
                for i in range(0, len(sentences), self.max_sentences)
            )
        ]


def _window_params(tool: str, params: dict) -> tuple[int, int]:
    if "max_tokens" not in params:
        raise ValueError(f"{tool} requires params['max_tokens']")
    max_tokens = int(params["max_tokens"])
    overlap = int(params.get("overlap", 0))
    if max_tokens < 1:
        raise ValueError(f"{tool}: max_tokens must be >= 1, got {max_tokens}")
    if not 0 <= overlap < max_tokens:
        raise ValueError(f"{tool}: overlap must satisfy 0 <= overlap < max_tokens, got {overlap}")
    return max_tokens, overlap


def get_chunker(tool: str, params: dict) -> Chunker:
    if tool == FIXED_TOKEN:
        return FixedTokenChunker(*_window_params(tool, params))
    if tool == MARKDOWN_HEADING:
        return MarkdownHeadingChunker(*_window_params(tool, params))
    if tool == SENTENCE:
        if "max_sentences" not in params:
            raise ValueError(f"{tool} requires params['max_sentences']")
        max_sentences = int(params["max_sentences"])
        if max_sentences < 1:
            raise ValueError(f"{tool}: max_sentences must be >= 1, got {max_sentences}")
        return SentenceChunker(max_sentences)
    raise ValueError(f"unknown chunker tool {tool!r}; expected one of {CHUNKER_TOOLS}")


def chunk_doc(doc_id: str, text: str, tool: str, params: dict,
              parse_stage_id: str, chunk_stage_id: str) -> list[ChunkRecord]:
    chunker = get_chunker(tool, params)
    records: list[ChunkRecord] = []
    for ordinal, span in enumerate(chunker.chunk(text)):
        chunk_text = text[span.start:span.end]
        records.append(ChunkRecord(
            chunk_id=hashing.chunk_id(doc_id, span.start, span.end, chunk_text),
            doc_id=doc_id,
            span_start=span.start,
            span_end=span.end,
            ordinal=ordinal,
            text=chunk_text,
            text_hash=hashing.text_hash(chunk_text),
            parse_stage_id=parse_stage_id,
            chunk_stage_id=chunk_stage_id,
        ))
    return records
