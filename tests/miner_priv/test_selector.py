import random
from collections import Counter
from unittest.mock import MagicMock

import pytest

from reliquary.miner.selector import Selector, BetaPosterior


def _fake_buckets():
    b = MagicMock()
    b.__len__ = MagicMock(return_value=10)
    b.bucket_of = MagicMock(side_effect=lambda i: ("Algebra", "Level 3"))
    return b


def test_next_returns_valid_idx_at_cold_start():
    sel = Selector(buckets=_fake_buckets(), rng=random.Random(42))
    idx = sel.next(cooldown_set=set())
    assert 0 <= idx < 10


def test_next_skips_cooldown():
    sel = Selector(buckets=_fake_buckets(), rng=random.Random(42))
    cooldown = {0, 1, 2, 3, 4, 5, 6, 7, 8}  # only 9 is eligible
    idx = sel.next(cooldown_set=cooldown)
    assert idx == 9


def test_next_raises_when_fully_cooldowned():
    sel = Selector(buckets=_fake_buckets(), rng=random.Random(42))
    cooldown = set(range(10))
    with pytest.raises(RuntimeError, match="fully in cooldown"):
        sel.next(cooldown_set=cooldown)


def test_next_prefers_high_score_prompts():
    """When some prompts have well-learned posteriors near p=0.5 (high score),
    they should be chosen more often than prompts at p=0.95 (low score)."""
    rng = random.Random(123)
    sel = Selector(buckets=_fake_buckets(), rng=rng)

    # Strongly inform: idx 0 has p~0.5 (high in-zone score),
    # idx 1 has p~0.95 (low in-zone score)
    sel._prompt_post[0] = BetaPosterior(alpha=50.0, beta=50.0)
    sel._prompt_post[1] = BetaPosterior(alpha=95.0, beta=5.0)

    counts = Counter()
    for _ in range(2000):
        # Restrict candidates to {0, 1} by cooldown'ing the rest
        cooldown = set(range(2, 10))
        idx = sel.next(cooldown_set=cooldown)
        counts[idx] += 1

    # 0 should be chosen far more often than 1
    assert counts[0] > 3 * counts[1]


def test_update_accepted_increments_alpha_and_beta():
    sel = Selector(buckets=_fake_buckets(), rng=random.Random(42))
    sel.update_accepted(prompt_idx=5, rewards=[1, 1, 1, 1, 0, 0, 0, 0])

    post = sel._prompt_post[5]
    assert post.alpha == 5.0   # 1 + 4 successes
    assert post.beta == 5.0    # 1 + 4 fails

    bucket_post = sel._bucket_post[("Algebra", "Level 3")]
    assert bucket_post.alpha == 5.0
    assert bucket_post.beta == 5.0


def test_update_local_reject_uses_rewards_not_sigma():
    """LOCAL_REJECT must update from raw rewards (σ alone is ambiguous)."""
    sel = Selector(buckets=_fake_buckets(), rng=random.Random(42))
    sel.update_local_reject(prompt_idx=5, rewards=[1, 0, 0, 0, 0, 0, 0, 0])
    post = sel._prompt_post[5]
    assert post.alpha == 2.0   # 1 + 1
    assert post.beta == 8.0    # 1 + 7


def test_update_superseded_increments_competitor_only():
    sel = Selector(buckets=_fake_buckets(), rng=random.Random(42))
    sel.update_superseded(prompt_idx=5)
    sel.update_superseded(prompt_idx=5)

    assert sel._competitor_seen[5] == 2
    # Posterior NOT touched
    assert 5 not in sel._prompt_post


def test_update_other_rejection_no_change():
    """WRONG_CHECKPOINT, PROMPT_IN_COOLDOWN, etc. → no posterior update."""
    sel = Selector(buckets=_fake_buckets(), rng=random.Random(42))
    sel.update_neutral(prompt_idx=5)
    assert 5 not in sel._prompt_post
    assert 5 not in sel._competitor_seen


def test_on_checkpoint_change_decays_prompt_posteriors():
    sel = Selector(buckets=_fake_buckets(), rng=random.Random(42))
    sel._prompt_post[5] = BetaPosterior(alpha=11.0, beta=21.0)
    sel.on_checkpoint_change(decay=0.5)
    post = sel._prompt_post[5]
    assert post.alpha == pytest.approx(6.0)   # 1 + 0.5*10
    assert post.beta == pytest.approx(11.0)   # 1 + 0.5*20


def test_on_checkpoint_change_decays_bucket_posteriors():
    sel = Selector(buckets=_fake_buckets(), rng=random.Random(42))
    sel._bucket_post[("Algebra", "Level 3")] = BetaPosterior(alpha=11.0, beta=21.0)
    sel.on_checkpoint_change(decay=0.5)
    post = sel._bucket_post[("Algebra", "Level 3")]
    assert post.alpha == pytest.approx(6.0)
    assert post.beta == pytest.approx(11.0)


def test_on_checkpoint_change_halves_competitor_seen():
    sel = Selector(buckets=_fake_buckets(), rng=random.Random(42))
    sel._competitor_seen[5] = 4
    sel._competitor_seen[6] = 1
    sel.on_checkpoint_change(decay=0.5)
    assert sel._competitor_seen[5] == 2
    assert sel._competitor_seen[6] == 0


def test_on_checkpoint_change_decay_zero_resets():
    sel = Selector(buckets=_fake_buckets(), rng=random.Random(42))
    sel._prompt_post[5] = BetaPosterior(alpha=10.0, beta=20.0)
    sel._bucket_post[("Algebra", "Level 3")] = BetaPosterior(alpha=10.0, beta=20.0)
    sel.on_checkpoint_change(decay=0.0)
    assert sel._prompt_post[5].alpha == 1.0
    assert sel._prompt_post[5].beta == 1.0
    assert sel._bucket_post[("Algebra", "Level 3")].alpha == 1.0
    assert sel._bucket_post[("Algebra", "Level 3")].beta == 1.0
