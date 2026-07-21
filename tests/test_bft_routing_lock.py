"""Lock the BFT math-only carve-out. The validator's forced-rollout carve-out
is scoped to openmathinstruct (#103/ca3ac67); forcing a CODE rollout would be
rejected. These pin the routing predicates the engine uses
(_generate_m_rollouts / _pre_bake_batch call bft_applicable(env_name)) so a
future edit cannot silently start force-terminating code."""
from reliquary.constants import BFT_THINKING_BUDGET
from reliquary.miner.bft import bft_applicable, phase1_max_new_tokens


def test_bft_applies_to_math_and_single_env_only():
    assert bft_applicable("openmathinstruct") is True
    assert bft_applicable(None) is True            # single-env deploy = math
    assert bft_applicable("opencodeinstruct") is False
    assert bft_applicable("somethingelse") is False


def test_phase1_budget_is_exactly_thinking_budget_for_bft():
    # math/single-env → EXACTLY the thinking budget (force span pins to
    # prompt_len + BFT_THINKING_BUDGET; a smaller cap = 100% TOKEN_TAMPERED)
    assert phase1_max_new_tokens(99999, "openmathinstruct") == BFT_THINKING_BUDGET
    assert phase1_max_new_tokens(99999, None) == BFT_THINKING_BUDGET
    # code → the miner's configured cap, single-pass (no forcing)
    assert phase1_max_new_tokens(4096, "opencodeinstruct") == 4096
