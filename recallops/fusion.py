"""Dense/sparse candidate fusion (retrieve-stage ``hybrid`` block).

Both methods rank the candidate UNION of the two input lists.

``weighted``: each list is min-max normalized over its own scores; a candidate
absent from a list contributes normalized 0 for that list (NOT the list's min
score, an absent candidate must never outrank a present low scorer under the
opposite weight). A list with zero score range (single candidate or all-equal)
normalizes every present candidate to 1.0 so presence still beats absence.
``score = (1 - w) * dense_norm + w * sparse_norm`` with ``w = params["bm25_weight"]``.

``rrf``: reciprocal rank fusion, ``sum 1 / (k0 + rank)`` over the lists a
candidate appears in (1-based ranks), ``k0 = params.get("k0", 60)``.

Ties break deterministically by dense rank, then sparse rank, then candidate id.
"""
from __future__ import annotations


def _first_ranks(pairs: list[tuple[str, float]]) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for i, (cid, _) in enumerate(pairs):
        ranks.setdefault(cid, i + 1)
    return ranks


def _minmax(pairs: list[tuple[str, float]]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for cid, s in pairs:
        scores.setdefault(cid, float(s))
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        return dict.fromkeys(scores, 1.0)
    return {cid: (s - lo) / (hi - lo) for cid, s in scores.items()}


def fuse(dense: list[tuple[str, float]], sparse: list[tuple[str, float]],
         method: str, params: dict) -> list[tuple[str, float]]:
    dense_rank = _first_ranks(dense)
    sparse_rank = _first_ranks(sparse)
    union = list(dense_rank)
    union.extend(cid for cid in sparse_rank if cid not in dense_rank)

    if method == "weighted":
        w = float(params["bm25_weight"])
        dense_norm = _minmax(dense)
        sparse_norm = _minmax(sparse)
        scored = {
            cid: (1.0 - w) * dense_norm.get(cid, 0.0) + w * sparse_norm.get(cid, 0.0)
            for cid in union
        }
    elif method == "rrf":
        k0 = float(params.get("k0", 60))
        scored = {
            cid: (1.0 / (k0 + dense_rank[cid]) if cid in dense_rank else 0.0)
            + (1.0 / (k0 + sparse_rank[cid]) if cid in sparse_rank else 0.0)
            for cid in union
        }
    else:
        raise ValueError(f"unknown fusion method: {method!r}")

    absent = len(dense) + len(sparse) + 1
    ordered = sorted(
        union,
        key=lambda cid: (
            -scored[cid],
            dense_rank.get(cid, absent),
            sparse_rank.get(cid, absent),
            cid,
        ),
    )
    return [(cid, scored[cid]) for cid in ordered]
