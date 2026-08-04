"""Composition-level termination gate — the bad_termination fix.

Live evidence (submit_diag, 2026-08-03): groups reached submission carrying
rollouts truncated at OUR cap (cl=2600, no EOS). The validator's
_classify_termination rejects any no-EOS rollout that stopped BELOW the
protocol cap (16384) as bad_termination — its "truncated" tolerance only
applies AT the protocol cap. The bake-site gates were bypassed by newer
pipeline paths, so the filter moves to the ONE choke point every path uses:
_try_select. Rule: in a non-BFT env (code), a rollout without EOS is simply
NOT composable. Math (BFT) keeps its force-answer flow untouched.
"""
from __future__ import annotations

from reliquary.miner.engine import terminating_rollouts


def _r(in_eos: bool, tag: str) -> dict:
    return {"in_eos": in_eos, "tag": tag}


def test_code_env_drops_non_eos_rollouts():
    rollouts = [_r(True, "a"), _r(False, "b"), _r(True, "c"), _r(False, "d")]
    kept = terminating_rollouts(rollouts, "opencodeinstruct")
    assert [r["tag"] for r in kept] == ["a", "c"]


def test_code_env_all_eos_passthrough():
    rollouts = [_r(True, "a"), _r(True, "b")]
    assert terminating_rollouts(rollouts, "opencodeinstruct") == rollouts


def test_math_bft_env_untouched():
    # BFT math rollouts may legitimately lack a natural EOS pre-force-answer;
    # their termination legality is handled by the BFT flow, not this gate.
    rollouts = [_r(False, "a"), _r(True, "b")]
    assert terminating_rollouts(rollouts, "openmathinstruct") == rollouts


def test_missing_in_eos_key_counts_as_non_terminated():
    rollouts = [{"tag": "no-key"}, _r(True, "ok")]
    kept = terminating_rollouts(rollouts, "opencodeinstruct")
    assert [r["tag"] for r in kept] == ["ok"]
