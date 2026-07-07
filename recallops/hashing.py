"""Content addressing for all RecallOps artifacts (PRD FR-1.1).

Every identifier in the system derives from content, never from time or
randomness, this is what makes manifests reproducible and arms memoizable.
"""
from __future__ import annotations

import hashlib
import json

_SEP = b"\x1f"


def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def h(*parts: str | bytes) -> str:
    hasher = hashlib.sha256()
    for i, part in enumerate(parts):
        if i:
            hasher.update(_SEP)
        hasher.update(part.encode("utf-8") if isinstance(part, str) else part)
    return hasher.hexdigest()[:16]


def doc_id(data: bytes) -> str:
    return "doc_" + h(data)


def text_hash(text: str) -> str:
    return "tx_" + h(text)


def chunk_id(doc: str, start: int, end: int, text: str) -> str:
    return "ch_" + h(doc, str(start), str(end), text)


def params_hash(params: dict) -> str:
    return "ph_" + h(canonical_json(params))


def embedding_key(text_h: str, provider: str, model: str, dims: int, ph: str) -> str:
    return "emb_" + h(text_h, provider, model, str(dims), ph)


def merkle_root(pairs: list[tuple[str, str]]) -> str:
    return "mr_" + h(canonical_json(sorted(pairs)))


def snapshot_hash(core: dict) -> str:
    return "snap_" + h(canonical_json(core))
