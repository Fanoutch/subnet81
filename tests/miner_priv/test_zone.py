import pytest
from reliquary.miner.zone import population_std, is_in_zone


def test_population_std_all_zeros():
    assert population_std([0.0] * 8) == pytest.approx(0.0)


def test_population_std_all_ones():
    assert population_std([1.0] * 8) == pytest.approx(0.0)


def test_population_std_half():
    """4 ones, 4 zeros — population σ = 0.5."""
    assert population_std([1, 1, 1, 1, 0, 0, 0, 0]) == pytest.approx(0.5)


def test_population_std_two_correct():
    """2/8 → σ = sqrt(12)/8 ≈ 0.4330."""
    assert population_std([1, 1, 0, 0, 0, 0, 0, 0]) == pytest.approx(0.4330127, abs=1e-6)


def test_in_zone_steady_passes_at_two():
    """σ ≈ 0.433 >= 0.43 (steady threshold)."""
    rewards = [1, 1, 0, 0, 0, 0, 0, 0]
    assert is_in_zone(rewards, bootstrap=False) is True


def test_in_zone_steady_fails_at_one():
    """σ ≈ 0.331 < 0.43."""
    rewards = [1, 0, 0, 0, 0, 0, 0, 0]
    assert is_in_zone(rewards, bootstrap=False) is False


def test_in_zone_bootstrap_passes_at_one():
    """During bootstrap, threshold is 0.33; σ ≈ 0.331 just passes."""
    rewards = [1, 0, 0, 0, 0, 0, 0, 0]
    assert is_in_zone(rewards, bootstrap=True) is True


def test_in_zone_steady_passes_at_six():
    """6/8 same as 2/8 by symmetry."""
    rewards = [1, 1, 1, 1, 1, 1, 0, 0]
    assert is_in_zone(rewards, bootstrap=False) is True
