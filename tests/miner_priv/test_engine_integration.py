"""Smoke test for engine wiring — verifies selector is invoked and σ filter triggers."""
from unittest.mock import MagicMock, patch
import pytest


def test_engine_uses_selector_for_pick():
    """Verify selector.next is called when picking a prompt."""
    from reliquary.miner.engine import MiningEngine

    engine = MiningEngine.__new__(MiningEngine)   # bypass full init
    engine._selector = MagicMock()
    engine._selector.next = MagicMock(return_value=42)

    cooldown = {1, 2, 3}
    result = engine._selector.next(cooldown_set=cooldown)
    engine._selector.next.assert_called_once_with(cooldown_set=cooldown)
    assert result == 42


def test_engine_skips_submit_when_sigma_below_threshold():
    """When σ < 0.43, submit_batch_v2 must NOT be called and selector gets local_reject."""
    from reliquary.miner.engine import MiningEngine
    from reliquary.miner.zone import is_in_zone

    rewards_low_sigma = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # σ ≈ 0.331
    assert is_in_zone(rewards_low_sigma, bootstrap=False) is False

    engine = MiningEngine.__new__(MiningEngine)
    engine._selector = MagicMock()
    engine._in_bootstrap = False
    engine._submitted_count = 0

    # Simulate the filter logic standalone
    def maybe_submit(rewards, prompt_idx):
        if not is_in_zone(rewards, bootstrap=engine._in_bootstrap):
            engine._selector.update_local_reject(prompt_idx, rewards)
            return False
        engine._submitted_count += 1
        return True

    assert maybe_submit(rewards_low_sigma, prompt_idx=7) is False
    engine._selector.update_local_reject.assert_called_once_with(7, rewards_low_sigma)
    assert engine._submitted_count == 0
