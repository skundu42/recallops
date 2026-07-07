"""Rerankers: query + (chunk_id, text) candidates -> ranked (chunk_id, score).

``recall.rerankers.overlap`` scores unique-token overlap
``|query_tokens ∩ chunk_tokens| / |query_tokens|``; ties keep the prior
candidate order (stable sort), an empty-token query scores everything 0.0.
"""
from __future__ import annotations

from collections.abc import Callable

from .bm25 import tokenize

Reranker = Callable[[str, list[tuple[str, str]]], list[tuple[str, float]]]


def _overlap(query: str, candidates: list[tuple[str, str]]) -> list[tuple[str, float]]:
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return [(cid, 0.0) for cid, _ in candidates]
    scored = [
        (cid, len(query_tokens & set(tokenize(text))) / len(query_tokens))
        for cid, text in candidates
    ]
    return sorted(scored, key=lambda kv: -kv[1])


def get_reranker(tool: str, params: dict) -> Reranker:
    if tool == "recall.rerankers.overlap":
        return _overlap
    raise ValueError(f"unknown reranker tool: {tool!r}")
