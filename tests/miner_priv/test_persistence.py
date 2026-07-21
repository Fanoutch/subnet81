import random
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from reliquary.miner.selector import BetaPosterior, Selector


def _fake_buckets():
    b = MagicMock()
    b.__len__ = MagicMock(return_value=10)
    b.bucket_of = MagicMock(side_effect=lambda i: ("Algebra", "Level 3"))
    return b


def test_save_and_load_roundtrip(tmp_path: Path):
    sel = Selector(buckets=_fake_buckets(), rng=random.Random(42))
    sel._prompt_post[5] = BetaPosterior(alpha=11.0, beta=21.0)
    sel._bucket_post[("Algebra", "Level 3")] = BetaPosterior(alpha=4.0, beta=4.0)
    sel._competitor_seen[5] = 2

    path = tmp_path / "selector.pkl"
    sel.save(path, checkpoint_n=42)

    sel2 = Selector(buckets=_fake_buckets(), rng=random.Random(42))
    loaded_n = sel2.load(path, current_checkpoint_n=42)

    assert loaded_n == 42
    assert sel2._prompt_post[5].alpha == 11.0
    assert sel2._prompt_post[5].beta == 21.0
    assert sel2._bucket_post[("Algebra", "Level 3")].alpha == 4.0
    assert sel2._competitor_seen[5] == 2


def test_load_with_advanced_checkpoint_applies_decay(tmp_path: Path):
    sel = Selector(buckets=_fake_buckets(), rng=random.Random(42))
    sel._prompt_post[5] = BetaPosterior(alpha=11.0, beta=21.0)
    path = tmp_path / "selector.pkl"
    sel.save(path, checkpoint_n=10)

    sel2 = Selector(buckets=_fake_buckets(), rng=random.Random(42))
    loaded_n = sel2.load(path, current_checkpoint_n=11, decay_per_step=0.5)

    # Single-step advance with decay 0.5 → BetaPosterior decay applied once
    assert loaded_n == 11
    post = sel2._prompt_post[5]
    assert post.alpha == pytest.approx(6.0)   # 1 + 0.5*10
    assert post.beta == pytest.approx(11.0)   # 1 + 0.5*20


def test_load_missing_file_returns_none(tmp_path: Path):
    sel = Selector(buckets=_fake_buckets(), rng=random.Random(42))
    result = sel.load(tmp_path / "nonexistent.pkl", current_checkpoint_n=10)
    assert result is None
    # No state loaded
    assert sel._prompt_post == {}
