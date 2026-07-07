"""Document parsers (FR-1.3 lineage inputs).

``text-v1`` is the default byte→text normalizer. ``markdown-v2`` additionally
strips heading markers and emphasis, producing *different* parsed text for the
same bytes, it exists to exercise parser-change degradation (FR-5.4), where
chunk spans become incomparable across snapshots.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .. import hashing
from ..models import StageSpec

PARSER_TOOLS = ("text-v1", "markdown-v2")

_HEADING_PREFIX = re.compile(r"^#{1,6}[ \t]*")
_EMPHASIS = re.compile(r"(?<!\w)(\*{1,3}|_{1,3})(?=\S)(.+?)(?<=\S)\1(?!\w)")


@dataclass(frozen=True)
class ParsedDoc:
    source_path: str
    raw_hash: str
    text: str


def _normalize(raw: bytes) -> str:
    text = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n"))


def _strip_markdown(text: str) -> str:
    lines = [
        _HEADING_PREFIX.sub("", line) if line.startswith("#") else line
        for line in text.split("\n")
    ]
    text = "\n".join(lines)
    while True:
        stripped = _EMPHASIS.sub(r"\2", text)
        if stripped == text:
            break
        text = stripped
    return "\n".join(line.rstrip() for line in text.split("\n"))


def _check_tool(tool: str) -> None:
    if tool not in PARSER_TOOLS:
        raise ValueError(f"unknown parser tool {tool!r}; expected one of {PARSER_TOOLS}")


def parse(source_path: str, raw: bytes, tool: str = "text-v1") -> ParsedDoc:
    _check_tool(tool)
    text = _normalize(raw)
    if tool == "markdown-v2":
        text = _strip_markdown(text)
    return ParsedDoc(source_path=source_path, raw_hash=hashing.doc_id(raw), text=text)


def parser_stage_spec(tool: str) -> StageSpec:
    _check_tool(tool)
    return StageSpec(id="parse", tool=tool, version="1")
