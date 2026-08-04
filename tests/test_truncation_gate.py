"""Study §5 — local truncation gate for the short generation cap.

With RELIQUARY_MAX_NEW_TOKENS lowered (study: 2600; winning code answers are
600-1000 tokens), partial truncation becomes common. The validator marks a
rollout truncated itself (cap without EOS) and REJECTS any submission with more
than MAX_TRUNCATED_PER_SUBMISSION_BY_ENV truncated rollouts (code: 3). The study
runs stricter: MAX_TRUNCATED_CODE=0 — submit only 100%-EOS groups; a group that
rambles is dropped locally (that's the §5 win: ~50s wasted instead of ~5min).

Gate semantics (code env only — math keeps the legacy BFT behaviour where
hitting the thinking budget is normal and force-answered):
  drop iff n_truncated > allowed, allowed = RELIQUARY_MAX_TRUNCATED_CODE (def 0).
"""
from __future__ import annotations

from reliquary.miner.engine import max_truncated_allowed, too_many_truncated


def test_default_allowance_is_zero_all_eos_required():
    assert max_truncated_allowed({}) == 0


def test_env_override_up_to_validator_limit():
    assert max_truncated_allowed({"RELIQUARY_MAX_TRUNCATED_CODE": "3"}) == 3
    # malformed / negative → safe default 0
    assert max_truncated_allowed({"RELIQUARY_MAX_TRUNCATED_CODE": "x"}) == 0
    assert max_truncated_allowed({"RELIQUARY_MAX_TRUNCATED_CODE": "-1"}) == 0


def test_gate_code_env_strict_by_default():
    # code (non-BFT): 8 rollouts, 7 terminated → 1 truncated > 0 → drop
    assert too_many_truncated(8, 7, "opencodeinstruct", {}) is True
    # all terminated → keep
    assert too_many_truncated(8, 8, "opencodeinstruct", {}) is False


def test_gate_code_env_respects_allowance():
    env = {"RELIQUARY_MAX_TRUNCATED_CODE": "3"}
    assert too_many_truncated(8, 5, "opencodeinstruct", env) is False  # 3 ≤ 3
    assert too_many_truncated(8, 4, "opencodeinstruct", env) is True   # 4 > 3


def test_gate_math_keeps_legacy_zero_terminated_rule():
    # math (BFT): only the historical "no rollout terminated" drop applies —
    # partial non-termination is normal pre-force-answer.
    assert too_many_truncated(8, 1, "openmathinstruct", {}) is False
    assert too_many_truncated(8, 0, "openmathinstruct", {}) is True
