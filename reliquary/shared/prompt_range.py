"""Deterministic per-window prompt range.

Validator and miner each derive the same contiguous ``[lo, hi)`` slice of an
environment's prompt index space from the shared per-window ``randomness``
seed, so a static/shared bank of pre-curated prompts only lands in-range a
small fraction of windows. Pure and dependency-free: both sides import this
single source of truth — any divergence would reject honest miners.
"""

from __future__ import annotations

import hashlib


def window_prompt_range(
    randomness: str,
    env_name: str,
    universe_n: int,
    size: int,
) -> tuple[int, int]:
    """Return the ``[lo, hi)`` prompt-index slice eligible this window.

    ``randomness`` is the per-window seed both sides already agree on (block
    hash + drand round). ``env_name`` domain-separates math vs code so their
    slices are independent. ``universe_n`` is the prompt index space size
    (``len(env)``); both sides MUST pass the same value, which holds whenever
    they load identical shards (already required by token binding). When
    ``universe_n <= size`` the whole space is eligible (no restriction).
    """
    if universe_n <= size:
        return (0, universe_n)
    seed = hashlib.sha256(
        b"prompt-range/v1|" + env_name.encode() + b"|" + randomness.encode()
    ).digest()
    lo = int.from_bytes(seed[:8], "big") % (universe_n - size)
    return (lo, lo + size)
