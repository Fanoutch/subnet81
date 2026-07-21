"""Diagnostic override for the local sigma-zone threshold.

The miner has never exercised its submission path in production: every group so
far (238 measured on 2026-07-21) was dropped locally by `_skip_for_out_of_zone`
before any network call, so precommit/reveal/verdict are untested end-to-end.

This override exists to deliberately let a group through so the REAL production
submit path runs against the live validator. The verdict will be OUT_OF_ZONE
(rejected before the GRAIL proof path, so it costs no expensive-proof budget),
but the handshake, the byte-identity of the revealed payload, and the timing
become observable.

DEFAULT MUST REMAIN 0.43 — the validator's steady SIGMA_MIN. A permanently
loosened threshold would submit work the validator always rejects, burning the
8-submissions-per-window quota for nothing.
"""

from __future__ import annotations

import importlib

import pytest


def _skip(monkeypatch, rewards, override=None):
    """Call the production filter with an optional env override."""
    if override is None:
        monkeypatch.delenv("RELIQUARY_ZONE_SIGMA_MIN", raising=False)
    else:
        monkeypatch.setenv("RELIQUARY_ZONE_SIGMA_MIN", override)
    engine = importlib.import_module("reliquary.miner.engine")
    return engine._skip_for_out_of_zone(rewards)


# 8 binary rollouts: k successes -> sigma = sqrt(p(1-p)), p = k/8
K1 = [1.0] + [0.0] * 7      # sigma 0.331 — just below the payable band
K2 = [1.0] * 2 + [0.0] * 6  # sigma 0.433 — lowest payable
K8 = [1.0] * 8              # sigma 0.000 — unanimous


def test_default_threshold_is_unchanged_at_the_validator_steady_value(monkeypatch):
    """No env set => exactly today's behaviour. This is the safety property."""
    assert _skip(monkeypatch, K1) is True    # 0.331 < 0.43 -> dropped
    assert _skip(monkeypatch, K2) is False   # 0.433 >= 0.43 -> kept


def test_override_lets_a_below_band_group_through(monkeypatch):
    """The whole point: exercise the submit path with a group we'd normally drop."""
    assert _skip(monkeypatch, K1, override="0.0") is False


def test_override_admits_even_a_unanimous_group(monkeypatch):
    """sigma=0 must pass at threshold 0, otherwise the diagnostic can still stall."""
    assert _skip(monkeypatch, K8, override="0.0") is False


def test_override_is_honoured_at_call_time_not_import_time(monkeypatch):
    """Flipping the env between calls must take effect without a reimport —
    otherwise a running miner could not be diagnosed without a restart."""
    assert _skip(monkeypatch, K1, override="0.0") is False
    assert _skip(monkeypatch, K1, override="0.43") is True


def test_malformed_override_falls_back_to_the_safe_default(monkeypatch):
    """A typo in the env must not silently disable the filter and flood the
    validator with rejects."""
    assert _skip(monkeypatch, K1, override="not-a-number") is True
