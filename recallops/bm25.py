"""In-package BM25 over chunk texts (FR-6.1/FR-6.2 sparse leg).

Lucene-style idf ``ln((N - df + 0.5) / (df + 0.5) + 1)``, always positive, so
a term occurring in every document still contributes. Query tokens contribute
per occurrence (a repeated query term scores twice).
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class BM25Index:
    corpus: dict[str, str]
    k1: float = 1.5
    b: float = 0.75
    _tf: dict[str, Counter] = field(init=False, repr=False)
    _dl: dict[str, int] = field(init=False, repr=False)
    _df: dict[str, int] = field(init=False, repr=False)
    _order: dict[str, int] = field(init=False, repr=False)
    _n: int = field(init=False, repr=False)
    _avgdl: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._tf = {}
        self._dl = {}
        self._df = {}
        self._order = {cid: i for i, cid in enumerate(self.corpus)}
        for cid, text in self.corpus.items():
            tokens = tokenize(text)
            counts = Counter(tokens)
            self._tf[cid] = counts
            self._dl[cid] = len(tokens)
            for term in counts:
                self._df[term] = self._df.get(term, 0) + 1
        self._n = len(self.corpus)
        total = sum(self._dl.values())
        self._avgdl = total / self._n if total else 1.0

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log((self._n - df + 0.5) / (df + 0.5) + 1.0)

    def scores(self, query: str) -> dict[str, float]:
        query_tokens = tokenize(query)
        out: dict[str, float] = {}
        for cid in self.corpus:
            tf = self._tf[cid]
            norm = self.k1 * (1.0 - self.b + self.b * self._dl[cid] / self._avgdl)
            score = 0.0
            for term in query_tokens:
                f = tf.get(term, 0)
                if f:
                    score += self._idf(term) * f * (self.k1 + 1.0) / (f + norm)
            if score > 0.0:
                out[cid] = score
        return out

    def _ranked(self, query: str) -> list[tuple[str, float]]:
        return sorted(self.scores(query).items(), key=lambda kv: (-kv[1], self._order[kv[0]]))

    def top(self, query: str, n: int) -> list[tuple[str, float]]:
        return self._ranked(query)[:n]

    def rank_of(self, query: str, chunk_id: str) -> int | None:
        for i, (cid, _) in enumerate(self._ranked(query)):
            if cid == chunk_id:
                return i + 1
        return None
