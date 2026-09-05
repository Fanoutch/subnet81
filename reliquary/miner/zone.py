"""Local σ filter — mirror of validator's is_in_zone, used pre-submit.

See docs/superpowers/specs/2026-05-03-optimized-miner-design.md section 6.
"""
from __future__ import annotations

from math import sqrt
from typing import Sequence

ZONE_THRESHOLD_STEADY = 0.43     # v3 — gardées pour les imports existants
ZONE_THRESHOLD_BOOTSTRAP = 0.33


def active_thresholds() -> tuple[float, float]:
    """(steady, bootstrap) du protocole actif.

    v4 = critère dynamic-sampling DAPO (upstream 8c38992) :
    σ(k=1|15, M=16) = 0.2421 → 0.24 admet tout k∈[1,15], 0.22 bootstrap ;
    les groupes unanimes (σ=0) restent hors zone. Import paresseux pour la
    testabilité (pattern upstream).
    """
    from reliquary.constants import PROTOCOL_VERSION

    if PROTOCOL_VERSION >= 4:
        return 0.24, 0.22
    return ZONE_THRESHOLD_STEADY, ZONE_THRESHOLD_BOOTSTRAP


def population_std(rewards: Sequence[float]) -> float:
    """Population standard deviation (n in denominator, not n-1)."""
    n = len(rewards)
    if n == 0:
        return 0.0
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    return sqrt(var)


def is_in_zone(rewards: Sequence[float], bootstrap: bool = False) -> bool:
    """True iff σ >= the active threshold (v3 : 0.43/0.33 ; v4 : 0.24/0.22)."""
    steady, boot = active_thresholds()
    threshold = boot if bootstrap else steady
    return population_std(rewards) >= threshold
