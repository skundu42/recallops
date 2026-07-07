"""Golden dataset bootstrap (PRD FR-3).

The offline heuristic generator samples snapshot chunks round-robin across
documents, picks each chunk's most corpus-distinctive sentence, and words a
question from deterministic templates, no LLM or network required. Import
and mining paths map external formats onto the §10.3 case schema.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from .bm25 import tokenize
from .models import ChunkRecord, GoldenCase, GoldenDataset, SnapshotManifest
from .store import ProjectStore

EXACT_TERM = "exact-term"
PARAPHRASE = "paraphrase"

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_MIN_SENTENCE_TOKENS = 6
_AUX_VERBS = frozenset({
    "is", "are", "was", "were", "has", "have", "had", "does", "do",
    "can", "could", "will", "would", "must", "should", "may", "might",
})
_CLAUSE_CAP = 12
_SUBJECT_CAP = 6


def _chunkset_key(manifest: SnapshotManifest) -> str:
    return PurePosixPath(manifest.artifacts["chunks_uri"]).stem


def _top_dir(source_path: str) -> str:
    head, _, tail = source_path.partition("/")
    return head if tail else PurePosixPath(source_path).stem


def _normalize_question(question: str) -> str:
    return " ".join(question.lower().split())


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def _doc_frequencies(records: list[ChunkRecord]) -> dict[str, int]:
    df: Counter[str] = Counter()
    for record in records:
        df.update(set(tokenize(record.text)))
    return df


def _rarity(tokens: list[str], df: dict[str, int]) -> float:
    return sum(1.0 / df.get(t, 1) for t in tokens)


def _distinctive_sentence(text: str, df: dict[str, int]) -> list[str] | None:
    tokenized = [t for t in (tokenize(s) for s in _sentences(text)) if t]
    if not tokenized:
        return None
    pool = [t for t in tokenized if len(t) >= _MIN_SENTENCE_TOKENS] or tokenized
    return max(pool, key=lambda t: _rarity(t, df))


def _rarest_phrase(tokens: list[str], df: dict[str, int]) -> str:
    best, best_score = tokens[:4], -1.0
    for size in (2, 3, 4):
        for i in range(len(tokens) - size + 1):
            gram = tokens[i:i + size]
            score = _rarity(gram, df) / size
            if score > best_score:
                best, best_score = gram, score
    return " ".join(best)


def _exact_term_question(tokens: list[str], df: dict[str, int]) -> str:
    return f"What does the documentation say about {_rarest_phrase(tokens, df)}?"


def _clause_after(tokens: list[str], marker: str) -> list[str] | None:
    if marker not in tokens:
        return None
    idx = tokens.index(marker)
    clause = tokens[idx + 1: idx + 1 + _CLAUSE_CAP]
    return clause or None


def _paraphrase_question(tokens: list[str]) -> str:
    clause = _clause_after(tokens, "when")
    if clause:
        return f"What happens when {' '.join(clause)}?"
    clause = _clause_after(tokens, "if")
    if clause:
        return f"What happens if {' '.join(clause)}?"
    subject = tokens[:4]
    for i, token in enumerate(tokens):
        if token in _AUX_VERBS and i > 0:
            subject = tokens[:i][:_SUBJECT_CAP]
            break
    return f"How is {' '.join(subject)} handled?"


def _llm_prompt(kind: str, source_path: str, text: str) -> str:
    return (
        f"Write one {kind} question answered by this passage from {source_path}. "
        f"Return only the question.\n\n{text}"
    )


def _sample_order(records: list[ChunkRecord], source_by_doc: dict[str, str],
                  seed: int) -> list[ChunkRecord]:
    rng = random.Random(seed)
    by_doc: dict[str, list[ChunkRecord]] = {}
    for record in records:
        by_doc.setdefault(record.doc_id, []).append(record)
    doc_ids = sorted(by_doc, key=lambda d: (source_by_doc.get(d, ""), d))
    rng.shuffle(doc_ids)
    queues: list[list[ChunkRecord]] = []
    for doc in doc_ids:
        group = sorted(by_doc[doc], key=lambda r: r.ordinal)
        rng.shuffle(group)
        queues.append(group)
    order: list[ChunkRecord] = []
    depth = 0
    while True:
        row = [q[depth] for q in queues if depth < len(q)]
        if not row:
            return order
        order.extend(row)
        depth += 1


def generate(store: ProjectStore, manifest: SnapshotManifest, n: int = 100, seed: int = 0,
             llm: Callable[[str], str] | None = None, name: str = "golden") -> GoldenDataset:
    records = store.get_chunks(_chunkset_key(manifest))
    docs = store.docs_for_merkle(manifest.corpus.merkle_root)
    source_by_doc = {d["doc_id"]: d["source_path"] for d in docs}
    df = _doc_frequencies(records)

    cases: list[GoldenCase] = []
    seen: set[str] = set()
    for record in _sample_order(records, source_by_doc, seed):
        if len(cases) >= n:
            break
        source = source_by_doc.get(record.doc_id)
        tokens = _distinctive_sentence(record.text, df)
        if source is None or tokens is None:
            continue
        kind = EXACT_TERM if len(cases) % 2 == 0 else PARAPHRASE
        if llm is not None:
            question = llm(_llm_prompt(kind, source, record.text)).strip()
        elif kind == EXACT_TERM:
            question = _exact_term_question(tokens, df)
        else:
            question = _paraphrase_question(tokens)
        norm = _normalize_question(question)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        cases.append(GoldenCase(
            id=f"q_{len(cases):03d}",
            question=question,
            expected_sources=[source],
            tags=[_top_dir(source), kind],
            origin="synthetic",
        ))
    return GoldenDataset(f"{name}-v1", cases)


def _imported_case(rec: dict, index: int) -> GoldenCase:
    if "question" in rec:
        question = rec["question"]
        expected = rec.get("reference_contexts") or rec.get("ground_truth") or []
    elif "input" in rec:
        question = rec["input"]
        expected = rec.get("context") or []
    else:
        raise ValueError(f"unrecognized golden case record with keys {sorted(rec)}")
    if isinstance(expected, str):
        expected = [expected]
    return GoldenCase(
        id=rec.get("id") or f"q_{index:03d}",
        question=question,
        expected_sources=list(expected),
        tags=list(rec.get("tags", [])),
    )


def import_file(path: Path, dataset_id: str) -> GoldenDataset:
    text = Path(path).read_text(encoding="utf-8").strip()
    try:
        doc = json.loads(text)
    except json.JSONDecodeError:
        doc = None
    if isinstance(doc, dict) and "cases" in doc:
        return GoldenDataset(dataset_id, [GoldenCase.from_dict(c) for c in doc["cases"]])
    lines = [line for line in text.splitlines() if line.strip()]
    return GoldenDataset(dataset_id, [_imported_case(json.loads(line), i)
                                      for i, line in enumerate(lines)])


def mine_jsonl(path: Path, dataset_id: str) -> GoldenDataset:
    cases: list[GoldenCase] = []
    lines = [line for line in Path(path).read_text(encoding="utf-8").splitlines()
             if line.strip()]
    for i, line in enumerate(lines):
        rec = json.loads(line)
        cases.append(GoldenCase(
            id=f"q_{i:03d}",
            question=rec["query"],
            expected_sources=list(rec["expected_sources"]),
            tags=[],
            origin="production",
            source_trace=rec.get("trace_id"),
        ))
    return GoldenDataset(dataset_id, cases)


def curate(ds: GoldenDataset, decisions: dict[str, str]) -> GoldenDataset:
    for case_id, decision in decisions.items():
        if decision not in ("accept", "reject"):
            raise ValueError(f"invalid decision {decision!r} for case {case_id!r}")
    kept = [c for c in ds.cases if decisions.get(c.id) != "reject"]
    return GoldenDataset(ds.dataset_id, kept)


def stratification_report(ds: GoldenDataset) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for case in ds.cases:
        counts.update(case.tags)
    return dict(sorted(counts.items()))
