"""Reciprocal Rank Fusion (RRF): several ranked lists in, one ranked list out.

Why RRF and not a weighted sum of scores: the two branches of hybrid search score on unrelated scales (a
cosine distance in [0, 2] and a `ts_rank_cd` that grows with term frequency and has no upper bound).
Normalising them is fragile (a min-max over the top-50 shifts with every query), so RRF ignores the scores
and uses only the **rank** each branch assigns:

    score(d) = sum over branches b of 1 / (k + rank_b(d))        rank is 1-based

A chunk found by both branches accumulates two terms, so agreement between them is rewarded without any
weight to tune. `k` damps the first positions: with k = 60 (the value of the original paper, Cormack,
Clarke & Buettcher 2009) rank 1 is worth 1/61 and rank 10 is worth 1/70, so one branch's top hit cannot
steamroll the other's broad agreement; with k = 0 the first position would weigh ten times the tenth.

Pure function, no I/O: the branches pass the ordered chunk ids and get back fused candidates with the
score and the per-branch rank, so the caller can show *why* a chunk ended where it did.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

# The constant of the original paper. Exposed through settings (RRF_K) so it can be tuned without code.
DEFAULT_RRF_K = 60


@dataclass(frozen=True)
class FusedCandidate:
    """One chunk after fusion: its RRF score and the 1-based rank it had in each branch that returned it."""

    key: int
    score: float
    ranks: dict[str, int] = field(default_factory=dict)


def reciprocal_rank_fusion(rankings: Mapping[str, Sequence[int]], *, k: int = DEFAULT_RRF_K) -> list[FusedCandidate]:
    """Fuse `{branch_name: [key, key, ...]}` (best first) into one list ordered by RRF score, best first.

    Ties are broken deterministically: higher score, then the best rank any branch gave the key, then the
    key itself. A branch that lists the same key twice only votes once (first occurrence).
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    scores: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}
    for branch, ordered in rankings.items():
        seen: set[int] = set()
        for position, key in enumerate(ordered, start=1):
            if key in seen:
                continue
            seen.add(key)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + position)
            ranks.setdefault(key, {})[branch] = position

    def sort_key(key: int) -> tuple[float, int, int]:
        return (-scores[key], min(ranks[key].values()), key)

    return [FusedCandidate(key=key, score=scores[key], ranks=dict(ranks[key])) for key in sorted(scores, key=sort_key)]
