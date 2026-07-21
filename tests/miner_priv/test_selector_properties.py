import random
from unittest.mock import MagicMock

from hypothesis import given, settings, strategies as st

from reliquary.miner.selector import BetaPosterior, Selector, score_in_zone


def _fake_buckets(n: int = 50):
    b = MagicMock()
    b.__len__ = MagicMock(return_value=n)
    b.bucket_of = MagicMock(side_effect=lambda i: ("X", "Y"))
    return b


@given(p=st.floats(min_value=0.0, max_value=1.0))
def test_score_bounded_unit_interval(p):
    s = score_in_zone(p, n=8)
    assert 0.0 <= s <= 1.0


@given(
    cooldown_count=st.integers(min_value=0, max_value=49),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_next_returns_valid_or_raises(cooldown_count, seed):
    sel = Selector(buckets=_fake_buckets(50), rng=random.Random(seed))
    cooldown = set(random.Random(seed + 1).sample(range(50), cooldown_count))
    if cooldown_count >= 50:
        return  # tested explicitly elsewhere
    idx = sel.next(cooldown_set=cooldown)
    assert 0 <= idx < 50
    assert idx not in cooldown


@given(
    updates=st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=49),
            st.integers(min_value=0, max_value=8),
        ),
        min_size=0,
        max_size=20,
    )
)
@settings(max_examples=50)
def test_update_commutative_on_independent_prompts(updates):
    """Order of updates on distinct prompts shouldn't change final posteriors."""
    sel_a = Selector(buckets=_fake_buckets(50), rng=random.Random(0))
    sel_b = Selector(buckets=_fake_buckets(50), rng=random.Random(0))

    # Filter to distinct prompt_idx so commutativity holds
    seen = set()
    distinct = []
    for idx, k in updates:
        if idx not in seen:
            seen.add(idx)
            distinct.append((idx, k))

    for idx, k in distinct:
        rewards = [1.0] * k + [0.0] * (8 - k)
        sel_a.update_accepted(idx, rewards)

    for idx, k in reversed(distinct):
        rewards = [1.0] * k + [0.0] * (8 - k)
        sel_b.update_accepted(idx, rewards)

    for idx in seen:
        if idx in sel_a._prompt_post:
            assert sel_a._prompt_post[idx].alpha == sel_b._prompt_post[idx].alpha
            assert sel_a._prompt_post[idx].beta == sel_b._prompt_post[idx].beta


@given(
    decay=st.floats(min_value=0.0, max_value=1.0),
    a=st.floats(min_value=1.0, max_value=1000.0),
    b=st.floats(min_value=1.0, max_value=1000.0),
)
def test_decay_is_idempotent_at_factor_one(decay, a, b):
    """decay(1.0) is a noop. Other factors monotonically shrink evidence."""
    p = BetaPosterior(alpha=a, beta=b)
    p.decay(1.0)
    assert p.alpha == a
    assert p.beta == b


@given(
    a=st.floats(min_value=1.0, max_value=1000.0),
    b=st.floats(min_value=1.0, max_value=1000.0),
)
def test_decay_zero_resets_to_prior(a, b):
    p = BetaPosterior(alpha=a, beta=b)
    p.decay(0.0)
    assert p.alpha == 1.0
    assert p.beta == 1.0
