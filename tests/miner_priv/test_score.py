import pytest
from reliquary.miner.selector import score_in_zone


def test_score_at_p_half_max():
    """For p=0.5 with n=8, P(2 <= X <= 6) = 238/256."""
    assert score_in_zone(0.5, n=8) == pytest.approx(238 / 256, abs=1e-6)


def test_score_at_extremes():
    """p=0 and p=1 give zero in-zone probability."""
    assert score_in_zone(0.0, n=8) == pytest.approx(0.0, abs=1e-9)
    assert score_in_zone(1.0, n=8) == pytest.approx(0.0, abs=1e-9)


def test_score_symmetry():
    """score is symmetric around p=0.5."""
    for p in (0.1, 0.2, 0.3, 0.4):
        assert score_in_zone(p, n=8) == pytest.approx(score_in_zone(1 - p, n=8), abs=1e-9)


def test_score_known_values():
    """Tabulated values from spec section 4.1."""
    assert score_in_zone(0.5, n=8) == pytest.approx(0.93, abs=0.01)
    assert score_in_zone(0.3, n=8) == pytest.approx(0.74, abs=0.01)
    assert score_in_zone(0.2, n=8) == pytest.approx(0.49, abs=0.02)
    assert score_in_zone(0.1, n=8) == pytest.approx(0.19, abs=0.02)


def test_score_bounded():
    """0 <= score <= 1 for all p in [0,1]."""
    for p in [0.0, 0.05, 0.5, 0.95, 1.0]:
        s = score_in_zone(p, n=8)
        assert 0.0 <= s <= 1.0
