import random
import pytest
from reliquary.miner.selector import BetaPosterior


def test_default_is_uniform():
    p = BetaPosterior()
    assert p.alpha == 1.0
    assert p.beta == 1.0


def test_update_increments():
    p = BetaPosterior()
    p.update(k_success=4, n_total=8)
    assert p.alpha == 5.0   # 1 + 4
    assert p.beta == 5.0    # 1 + (8 - 4)


def test_update_zero_successes():
    p = BetaPosterior(alpha=2.0, beta=3.0)
    p.update(k_success=0, n_total=8)
    assert p.alpha == 2.0
    assert p.beta == 11.0


def test_sample_returns_in_unit_interval():
    p = BetaPosterior(alpha=2.0, beta=5.0)
    rng = random.Random(42)
    for _ in range(100):
        x = p.sample(rng)
        assert 0.0 < x < 1.0


def test_sample_mean_close_to_alpha_over_total():
    """Beta(α, β) has mean α/(α+β); empirical sample mean should match."""
    p = BetaPosterior(alpha=20.0, beta=30.0)
    rng = random.Random(42)
    samples = [p.sample(rng) for _ in range(5000)]
    expected = 20.0 / 50.0
    empirical = sum(samples) / len(samples)
    assert empirical == pytest.approx(expected, abs=0.02)


def test_decay():
    """Decay shrinks evidence (alpha-1, beta-1) by the factor."""
    p = BetaPosterior(alpha=11.0, beta=21.0)  # evidence: 10 successes, 20 fails
    p.decay(0.5)
    # New alpha = 1 + 0.5*10 = 6, new beta = 1 + 0.5*20 = 11
    assert p.alpha == pytest.approx(6.0, abs=1e-9)
    assert p.beta == pytest.approx(11.0, abs=1e-9)


def test_decay_one_is_noop():
    p = BetaPosterior(alpha=11.0, beta=21.0)
    p.decay(1.0)
    assert p.alpha == 11.0
    assert p.beta == 21.0


def test_decay_zero_resets_to_prior():
    p = BetaPosterior(alpha=11.0, beta=21.0)
    p.decay(0.0)
    assert p.alpha == 1.0
    assert p.beta == 1.0
