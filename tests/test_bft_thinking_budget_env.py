"""BFT_THINKING_BUDGET is env-configurable and defaults to the live 4B/v3 value.

BFT_THINKING_BUDGET is a WIRE CONSTANT — the miner and validator must agree to
the token (the forced-seed span is pinned at prompt_len + budget). Live 4B/v3 =
15616 (was 2048 on the dead 2B/v2). BFT is MATH-only, so this is irrelevant to
our code mining but kept live-correct. Env override (RELIQUARY_BFT_THINKING_BUDGET)
allows a rollback to 2048 without a code change; malformed / non-positive falls
back to the live default.
"""
from __future__ import annotations

from reliquary import constants


def test_defaults_to_live_4b_v3_value_when_unset():
    assert constants.bft_thinking_budget({}) == 15616


def test_reads_env_override_for_rollback():
    # any positive int flows through unchanged (e.g. rollback to 2B/v2).
    assert constants.bft_thinking_budget(
        {"RELIQUARY_BFT_THINKING_BUDGET": "2048"}
    ) == 2048


def test_malformed_value_falls_back_to_default():
    # A bad value must NOT silently zero the budget (would break every rollout).
    assert constants.bft_thinking_budget(
        {"RELIQUARY_BFT_THINKING_BUDGET": "not-a-number"}
    ) == 15616


def test_non_positive_value_falls_back_to_default():
    assert constants.bft_thinking_budget(
        {"RELIQUARY_BFT_THINKING_BUDGET": "0"}
    ) == 15616


def test_shipped_budget_and_cap_are_v3_and_assert_holds():
    assert constants.BFT_THINKING_BUDGET == 15616
    assert constants.MAX_NEW_TOKENS_PROTOCOL_CAP == 16384
    assert (
        constants.MAX_NEW_TOKENS_PROTOCOL_CAP
        >= constants.BFT_THINKING_BUDGET + constants.BFT_ANSWER_BUDGET
    )
