"""Tie-break in pick_bake_env must not starve the second environment.

Measured 2026-07-21: 130 math bakes against 5 code bakes (26:1). Cause — under
the sigma filter NOTHING ever reaches the pool, so pool_counts stays {0, 0} and
every deficit is equal. The strict ``>`` comparison then always returns the
FIRST env in target_slots order, so openmathinstruct won every single tie and
opencodeinstruct was effectively never baked.

That matters: the code env is far less contested (window 24308: 44 distinct
math prompts already taken vs 2 for code) and unmined emissions are burned.

Rotating on ties keeps the deficit logic intact (a real deficit still wins)
while splitting the ties evenly.
"""

from __future__ import annotations

import pytest

from reliquary.miner.mix_controller import pick_bake_env

ENVS = {"openmathinstruct": 4, "opencodeinstruct": 4}


def test_a_real_deficit_still_decides():
    """Rotation must not override the actual balancing signal."""
    # code already has 4 in the pool, math none -> math must win outright
    assert pick_bake_env(ENVS, {"openmathinstruct": 0, "opencodeinstruct": 4}) \
        == "openmathinstruct"
    assert pick_bake_env(ENVS, {"openmathinstruct": 4, "opencodeinstruct": 0}) \
        == "opencodeinstruct"


def test_ties_alternate_instead_of_always_picking_the_first_env():
    """The starvation bug: equal deficits used to return math every time."""
    empty = {"openmathinstruct": 0, "opencodeinstruct": 0}
    picks = [pick_bake_env(ENVS, empty) for _ in range(8)]
    assert set(picks) == set(ENVS), f"one env never picked: {picks}"


def test_ties_are_split_evenly_over_many_calls():
    """Roughly 50/50 — not 26:1 as measured in production."""
    empty = {"openmathinstruct": 0, "opencodeinstruct": 0}
    picks = [pick_bake_env(ENVS, empty) for _ in range(100)]
    math = picks.count("openmathinstruct")
    assert 40 <= math <= 60, f"skewed split: {math}/100 math"


def test_single_env_is_unchanged():
    """Phase-1 behaviour: with one env it must always return that env."""
    solo = {"openmathinstruct": 8}
    assert {pick_bake_env(solo, {}) for _ in range(5)} == {"openmathinstruct"}
