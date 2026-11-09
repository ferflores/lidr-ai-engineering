"""Metrics of the retrieval eval (session 10): precision at k over a hand-annotated golden set, and latency.

Relevance is annotated at **budget** level (the golden set says which historical budgets are useful to
estimate the query), while the system returns **chunks** (one component each). A chunk is relevant when its
`budget_id` is in the query's relevant set. Two readings of the same top-k:

- `precision_at_k`: fraction of the k chunks that belong to a relevant budget. The exercise's metric.
- `relevant_budgets_recall_at_k`: fraction of the relevant budgets that appear at least once in the top-k.
  Complements precision: five chunks of one relevant budget is precision 1.0, and still a poor context for
  the generator if the query had two relevant budgets.

Latency is summarised with order statistics (median, p95), not the mean: a single cold request would move
the mean and say nothing about what a user sees most of the time.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass


def precision_at_k(retrieved_budget_ids: Sequence[str], relevant_budget_ids: Collection[str], k: int) -> float:
    """Share of the first k retrieved chunks whose budget is relevant. Fewer than k results count as misses."""
    if k < 1:
        raise ValueError("k must be >= 1")
    relevant = set(relevant_budget_ids)
    hits = sum(1 for budget_id in retrieved_budget_ids[:k] if budget_id in relevant)
    return hits / k


def relevant_budgets_recall_at_k(retrieved_budget_ids: Sequence[str], relevant_budget_ids: Collection[str], k: int) -> float:
    """Share of the relevant budgets that show up at least once in the first k chunks."""
    if k < 1:
        raise ValueError("k must be >= 1")
    relevant = set(relevant_budget_ids)
    if not relevant:
        raise ValueError("a golden query needs at least one relevant budget")
    found = {budget_id for budget_id in retrieved_budget_ids[:k] if budget_id in relevant}
    return len(found) / len(relevant)


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile (pct in [0, 100]); exact for the small samples of this eval."""
    if not values:
        raise ValueError("percentile of an empty sample")
    if not 0 <= pct <= 100:
        raise ValueError("pct must be in [0, 100]")
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return float(ordered[rank - 1])


@dataclass(frozen=True)
class LatencySummary:
    samples: int
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float

    @classmethod
    def from_samples(cls, values: Sequence[float]) -> "LatencySummary":
        return cls(
            samples=len(values),
            median_ms=percentile(values, 50),
            p95_ms=percentile(values, 95),
            min_ms=float(min(values)),
            max_ms=float(max(values)),
        )
