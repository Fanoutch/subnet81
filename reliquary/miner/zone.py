"""Local σ filter — mirror of validator's is_in_zone, used pre-submit.

See docs/superpowers/specs/2026-05-03-optimized-miner-design.md section 6.
"""
from __future__ import annotations

from math import sqrt
from typing import Sequence

ZONE_THRESHOLD_STEADY = 0.43
ZONE_THRESHOLD_BOOTSTRAP = 0.33


def population_std(rewards: Sequence[float]) -> float:
    """Population standard deviation (n in denominator, not n-1)."""
    n = len(rewards)
    if n == 0:
        return 0.0
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    return sqrt(var)


def is_in_zone(rewards: Sequence[float], bootstrap: bool = False) -> bool:
    """True iff σ >= the active threshold (0.33 bootstrap, 0.43 steady)."""
    threshold = ZONE_THRESHOLD_BOOTSTRAP if bootstrap else ZONE_THRESHOLD_STEADY
    return population_std(rewards) >= threshold
