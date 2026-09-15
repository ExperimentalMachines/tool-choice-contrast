"""Statistics. Stdlib only -- the upstream repository has no statistics utilities at all.

Every paired quantity is computed over stems, never items. A stem contributes two or three arms,
so items are not independent: an item-level bootstrap here would produce intervals roughly 40% too
narrow, which is worse than reporting none.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Interval:
    point: float | None
    low: float | None
    high: float | None
    method: str

    def as_dict(self) -> dict[str, object]:
        return {"point": self.point, "ci_low": self.low, "ci_high": self.high, "method": self.method}


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> Interval:
    """Wilson score interval. Returns a null point for an empty denominator, never 0.0."""
    if total <= 0:
        return Interval(None, None, None, "wilson[UNMEASURABLE]")
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return Interval(p, max(0.0, centre - half), min(1.0, centre + half), "wilson")


def mcnemar_exact(b: int, c: int) -> dict[str, object]:
    """Two-sided exact binomial test on the discordant pairs.

    `b` and `c` are always reported alongside the p-value. Their difference cancels a model that
    breaks twenty stems and fixes twenty, and only the raw pair shows that.
    """
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "discordant": 0, "p_value": None, "method": "exact_binomial[NO_DISCORDANT]"}
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5**n)
    return {"b": b, "c": c, "discordant": n, "p_value": min(1.0, 2 * tail), "method": "exact_binomial"}


def cluster_bootstrap(
    clusters: Sequence[object],
    statistic: Callable[[Sequence[object]], float | None],
    *,
    resamples: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Interval:
    """Percentile bootstrap resampling whole stems, carrying all their arms together."""
    point = statistic(clusters)
    if point is None or not clusters:
        return Interval(point, None, None, "cluster_bootstrap[UNMEASURABLE]")
    rng = random.Random(seed)
    n = len(clusters)
    draws: list[float] = []
    for _ in range(resamples):
        sample = [clusters[rng.randrange(n)] for _ in range(n)]
        value = statistic(sample)
        if value is not None:
            draws.append(value)
    if not draws:
        return Interval(point, None, None, "cluster_bootstrap[UNMEASURABLE]")
    draws.sort()
    lo = draws[max(0, int((alpha / 2) * len(draws)) - 1)]
    hi = draws[min(len(draws) - 1, int((1 - alpha / 2) * len(draws)))]
    return Interval(point, lo, hi, f"cluster_bootstrap[B={len(draws)}]")
