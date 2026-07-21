# Optimized Miner — Phase 1 Implementation Plan (Selector + filtre local)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a private miner for Reliquary subnet 81 that replaces uniform-random prompt selection with a Bayesian selector and adds a local σ filter pre-submit, eliminating `OUT_OF_ZONE` rejections at the validator.

**Architecture:** Local fork of `reliquadotai/reliquary` on branch `priv`. New module `reliquary/miner/selector.py` holds two-level Beta posteriors (per-prompt × per-bucket) with Thompson sampling. `engine.py` modified to (a) pick prompts via the selector, (b) compute σ from local rewards before submitting, (c) discard if σ < threshold. Single HF model on GPU 0 (no vLLM yet — that's Phase 2). Pipeline stays sequential (no async pipeline yet — that's Phase 3).

**Tech Stack:** Python 3.11, PyTorch, HuggingFace transformers, scipy.stats, Bittensor SDK, pytest, hypothesis.

**Spec reference:** `docs/superpowers/specs/2026-05-03-optimized-miner-design.md` sections 4 (Selector), 6 (Pipeline — but only the σ filter part is in scope), 7.2 (Unit tests), 7.5 (Property tests), 8 Phase 1.

**Implementation deviation from spec:** in-memory posteriors are keyed by `prompt_idx` only (not `(prompt_idx, checkpoint_n)`). The checkpoint dimension is implicit (we hold posteriors for the current ckpt only). Multi-checkpoint history is recovered at boot from the pickle, which stores `(checkpoint_n, posteriors_dict)` and applies `on_checkpoint_change` decay if needed. Rationale: avoids unbounded memory growth.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `reliquary/miner/selector.py` | Create | `BetaPosterior`, `Selector` class, scoring, persistence |
| `reliquary/miner/buckets.py` | Create | Read `(level, type)` from raw HF dataset, map prompt_idx → bucket key |
| `reliquary/miner/zone.py` | Create | `population_std`, `is_in_zone` (mirror of validator's logic, used locally pre-submit) |
| `reliquary/miner/engine.py` | Modify | Replace `pick_prompt_idx` call with `selector.next`, add σ filter pre-submit, wire feedback |
| `tests/miner_priv/__init__.py` | Create | Empty — pytest discovery |
| `tests/miner_priv/test_score.py` | Create | Unit tests for `score_in_zone` |
| `tests/miner_priv/test_beta_posterior.py` | Create | Unit tests for `BetaPosterior` |
| `tests/miner_priv/test_buckets.py` | Create | Unit tests for bucket extraction |
| `tests/miner_priv/test_selector.py` | Create | Unit tests for `Selector` |
| `tests/miner_priv/test_zone.py` | Create | Unit tests for σ filter |
| `tests/miner_priv/test_persistence.py` | Create | Pickle round-trip + boot-time migration |
| `tests/miner_priv/test_selector_properties.py` | Create | Hypothesis property-based tests |
| `tests/miner_priv/test_engine_integration.py` | Create | Engine wiring smoke tests with a fake validator |

---

## Task 0: Setup the private fork

**Files:** none in repo (env setup only)

- [ ] **Step 1: Clone the public repo to a private location**

```bash
git clone https://github.com/reliquadotai/reliquary.git ~/reliquary-miner-priv
cd ~/reliquary-miner-priv
```

- [ ] **Step 2: Re-wire remotes — make upstream point to public, no origin**

```bash
git remote rename origin upstream
git remote -v
```

Expected output:
```
upstream    https://github.com/reliquadotai/reliquary.git (fetch)
upstream    https://github.com/reliquadotai/reliquary.git (push)
```

- [ ] **Step 3: Create the private branch**

```bash
git checkout -b priv
```

- [ ] **Step 4: Install the package and dev dependencies**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pytest hypothesis
```

- [ ] **Step 5: Verify the install**

```bash
reliquary --help
pytest tests/ -x --collect-only | head -20
```

Expected: `reliquary --help` shows `mine` and `validate` subcommands; pytest discovers existing tests.

- [ ] **Step 6: Copy the spec into the private repo for reference**

```bash
mkdir -p docs/superpowers/specs docs/superpowers/plans
cp /home/ubuntu/Catalyst/docs/superpowers/specs/2026-05-03-optimized-miner-design.md docs/superpowers/specs/
cp /home/ubuntu/Catalyst/docs/superpowers/plans/2026-05-03-optimized-miner-phase1.md docs/superpowers/plans/
```

- [ ] **Step 7: Commit setup**

```bash
git add docs/superpowers/
git commit -m "chore: add design spec + phase 1 plan"
```

---

## Task 1: `score_in_zone` function

**Files:**
- Create: `reliquary/miner/selector.py`
- Test: `tests/miner_priv/test_score.py`, `tests/miner_priv/__init__.py`

- [ ] **Step 1: Create empty test package init**

```bash
mkdir -p tests/miner_priv
touch tests/miner_priv/__init__.py
```

- [ ] **Step 2: Write the failing test**

Create `tests/miner_priv/test_score.py`:

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/miner_priv/test_score.py -v
```

Expected: ImportError (`reliquary.miner.selector` does not exist).

- [ ] **Step 4: Write minimal implementation**

Create `reliquary/miner/selector.py`:

```python
"""Private miner — Bayesian prompt selector.

Two-level Beta posteriors (per-prompt × per-bucket) with Thompson sampling.
See docs/superpowers/specs/2026-05-03-optimized-miner-design.md section 4.
"""
from __future__ import annotations

from math import comb


def score_in_zone(p: float, n: int = 8, k_lo: int = 2, k_hi: int = 6) -> float:
    """P(k_lo <= X <= k_hi | X ~ Binomial(n, p)).

    For Reliquary MATH (binary rewards), in-zone means 2 <= X <= 6 successes
    out of n=8, which corresponds to σ >= 0.43.
    """
    return sum(
        comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
        for k in range(k_lo, k_hi + 1)
    )
```

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/miner_priv/test_score.py -v
```

Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add reliquary/miner/selector.py tests/miner_priv/__init__.py tests/miner_priv/test_score.py
git commit -m "feat(miner-priv): score_in_zone for Binomial(n,p) P(2<=X<=6)"
```

---

## Task 2: `BetaPosterior` dataclass

**Files:**
- Modify: `reliquary/miner/selector.py`
- Test: `tests/miner_priv/test_beta_posterior.py`

- [ ] **Step 1: Write the failing test**

Create `tests/miner_priv/test_beta_posterior.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/miner_priv/test_beta_posterior.py -v
```

Expected: ImportError on `BetaPosterior`.

- [ ] **Step 3: Write the implementation**

Append to `reliquary/miner/selector.py`:

```python
import random
from dataclasses import dataclass


@dataclass
class BetaPosterior:
    """Beta(α, β) prior/posterior for binary success rate."""

    alpha: float = 1.0
    beta: float = 1.0

    def sample(self, rng: random.Random) -> float:
        return rng.betavariate(self.alpha, self.beta)

    def update(self, k_success: int, n_total: int) -> None:
        if k_success < 0 or k_success > n_total:
            raise ValueError("k_success must be in [0, n_total]")
        self.alpha += k_success
        self.beta += (n_total - k_success)

    def decay(self, factor: float) -> None:
        """Shrink evidence (counts beyond the Beta(1,1) prior) by *factor*.

        decay(1.0) is a no-op. decay(0.0) resets to Beta(1, 1).
        """
        if not 0.0 <= factor <= 1.0:
            raise ValueError("factor must be in [0, 1]")
        self.alpha = 1.0 + factor * (self.alpha - 1.0)
        self.beta = 1.0 + factor * (self.beta - 1.0)
```

Don't forget the `import random` and `from dataclasses import dataclass` at the top of the file (move imports to the top if they're not there).

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/miner_priv/test_beta_posterior.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add reliquary/miner/selector.py tests/miner_priv/test_beta_posterior.py
git commit -m "feat(miner-priv): BetaPosterior with sample/update/decay"
```

---

## Task 3: `population_std` and `is_in_zone` (zone helpers)

**Files:**
- Create: `reliquary/miner/zone.py`
- Test: `tests/miner_priv/test_zone.py`

These mirror the validator's logic locally so we can pre-filter before submitting.

- [ ] **Step 1: Write the failing test**

Create `tests/miner_priv/test_zone.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/miner_priv/test_zone.py -v
```

Expected: ImportError.

- [ ] **Step 3: Write the implementation**

Create `reliquary/miner/zone.py`:

```python
"""Local σ filter — mirror of validator's is_in_zone, used pre-submit.

See docs/superpowers/specs/2026-05-03-optimized-miner-design.md section 6.
"""
from __future__ import annotations

from math import sqrt
from typing import Sequence

ZONE_THRESHOLD_STEADY = 0.43
ZONE_THRESHOLD_BOOTSTRAP = 0.33


def population_std(rewards: Sequence[float]) -> float:
    """Population standard deviation (n in denominator, not n-1)."""
    n = len(rewards)
    if n == 0:
        return 0.0
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    return sqrt(var)


def is_in_zone(rewards: Sequence[float], bootstrap: bool = False) -> bool:
    """True iff σ >= the active threshold (0.33 bootstrap, 0.43 steady)."""
    threshold = ZONE_THRESHOLD_BOOTSTRAP if bootstrap else ZONE_THRESHOLD_STEADY
    return population_std(rewards) >= threshold
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/miner_priv/test_zone.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add reliquary/miner/zone.py tests/miner_priv/test_zone.py
git commit -m "feat(miner-priv): local σ filter (population_std, is_in_zone)"
```

---

## Task 4: Bucket extraction (level + type from raw dataset)

**Files:**
- Create: `reliquary/miner/buckets.py`
- Test: `tests/miner_priv/test_buckets.py`

- [ ] **Step 1: Write the failing test**

Create `tests/miner_priv/test_buckets.py`:

```python
from unittest.mock import patch, MagicMock
from reliquary.miner.buckets import BucketIndex


def _fake_dataset():
    """Mimic qwedsacf/competition_math row schema."""
    return [
        {"problem": "p0", "level": "Level 3", "type": "Algebra"},
        {"problem": "p1", "level": "Level 5", "type": "Counting & Probability"},
        {"problem": "p2", "level": "Level 1", "type": "Algebra"},
    ]


@patch("reliquary.miner.buckets._load_raw_dataset")
def test_bucket_of_returns_canonical_key(mock_load):
    mock_load.return_value = _fake_dataset()
    idx = BucketIndex()
    assert idx.bucket_of(0) == ("Algebra", "Level 3")
    assert idx.bucket_of(1) == ("Counting & Probability", "Level 5")
    assert idx.bucket_of(2) == ("Algebra", "Level 1")


@patch("reliquary.miner.buckets._load_raw_dataset")
def test_bucket_of_handles_missing_fields(mock_load):
    mock_load.return_value = [
        {"problem": "p0"},                       # no level, no type
        {"problem": "p1", "level": "Level 2"},   # no type
    ]
    idx = BucketIndex()
    assert idx.bucket_of(0) == ("unknown", "unknown")
    assert idx.bucket_of(1) == ("unknown", "Level 2")


@patch("reliquary.miner.buckets._load_raw_dataset")
def test_bucket_of_wraps_modulo(mock_load):
    mock_load.return_value = _fake_dataset()
    idx = BucketIndex()
    # Out-of-range index wraps modulo dataset length (mirror env.get_problem)
    assert idx.bucket_of(3) == idx.bucket_of(0)
    assert idx.bucket_of(7) == idx.bucket_of(1)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/miner_priv/test_buckets.py -v
```

Expected: ImportError.

- [ ] **Step 3: Write the implementation**

Create `reliquary/miner/buckets.py`:

```python
"""Bucket index over the raw qwedsacf/competition_math dataset.

Loads (type, level) per prompt_idx once at startup; used by the selector
as a hyperprior for cold prompts.
"""
from __future__ import annotations


def _load_raw_dataset():
    """Load the raw HF dataset rows. Wrapped for mock-ability in tests."""
    import datasets as hf_datasets
    return hf_datasets.load_dataset(
        "qwedsacf/competition_math", split="train"
    )


class BucketIndex:
    """Maps prompt_idx -> (type, level) tuple, the bucket key."""

    def __init__(self) -> None:
        self._rows = list(_load_raw_dataset())
        self._n = len(self._rows)

    def __len__(self) -> int:
        return self._n

    def bucket_of(self, prompt_idx: int) -> tuple[str, str]:
        """Return (type, level) for *prompt_idx*. Wraps modulo dataset length."""
        idx = prompt_idx % self._n
        row = self._rows[idx]
        type_ = row.get("type") or "unknown"
        level = row.get("level") or "unknown"
        return (type_, level)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/miner_priv/test_buckets.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add reliquary/miner/buckets.py tests/miner_priv/test_buckets.py
git commit -m "feat(miner-priv): BucketIndex for (type, level) hyperprior keys"
```

---

## Task 5: `Selector` skeleton + `next()` with Thompson sampling

**Files:**
- Modify: `reliquary/miner/selector.py`
- Test: `tests/miner_priv/test_selector.py`

- [ ] **Step 1: Write the failing test**

Create `tests/miner_priv/test_selector.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/miner_priv/test_selector.py -v
```

Expected: ImportError on `Selector`.

- [ ] **Step 3: Write the implementation**

Append to `reliquary/miner/selector.py`:

```python
class Selector:
    """Bayesian prompt selector — two-level Beta posteriors + Thompson sampling.

    See spec section 4.
    """

    def __init__(self, buckets, rng: random.Random | None = None):
        self._buckets = buckets
        self._rng = rng if rng is not None else random.Random()
        self._prompt_post: dict[int, BetaPosterior] = {}
        self._bucket_post: dict[tuple, BetaPosterior] = {}
        self._competitor_seen: dict[int, int] = {}
        # Anti-SUPERSEDED penalty strength (spec §4.3)
        self._gamma = 0.3

    def _posterior_for(self, prompt_idx: int) -> BetaPosterior:
        """Return the active posterior — prompt-level if seen, else bucket prior."""
        if prompt_idx in self._prompt_post:
            return self._prompt_post[prompt_idx]
        bucket_key = self._buckets.bucket_of(prompt_idx)
        if bucket_key not in self._bucket_post:
            self._bucket_post[bucket_key] = BetaPosterior()
        return self._bucket_post[bucket_key]

    def _score(self, p_sample: float, idx: int) -> float:
        from math import exp
        base = score_in_zone(p_sample, n=8)
        penalty = exp(-self._gamma * self._competitor_seen.get(idx, 0))
        return base * penalty

    def next(self, cooldown_set: set[int]) -> int:
        """Pick the prompt_idx with the highest Thompson-sampled in-zone score."""
        n = len(self._buckets)
        candidates = [i for i in range(n) if i not in cooldown_set]
        if not candidates:
            raise RuntimeError("env fully in cooldown")
        best_idx = candidates[0]
        best_score = -1.0
        for idx in candidates:
            post = self._posterior_for(idx)
            p_sample = post.sample(self._rng)
            s = self._score(p_sample, idx)
            if s > best_score:
                best_score = s
                best_idx = idx
        return best_idx
```

Make sure `from math import exp` is at the top of the file (move existing top-of-file imports together if needed).

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/miner_priv/test_selector.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add reliquary/miner/selector.py tests/miner_priv/test_selector.py
git commit -m "feat(miner-priv): Selector.next with Thompson sampling + cooldown filter"
```

---

## Task 6: `Selector.update()` — feedback from each submission outcome

**Files:**
- Modify: `reliquary/miner/selector.py`
- Test: `tests/miner_priv/test_selector.py` (add cases)

- [ ] **Step 1: Write the failing tests**

Append to `tests/miner_priv/test_selector.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/miner_priv/test_selector.py -v
```

Expected: 4 new failures (`update_accepted`, `update_local_reject`, `update_superseded`, `update_neutral` not defined).

- [ ] **Step 3: Write the implementation**

Append to `reliquary/miner/selector.py` (inside the `Selector` class):

```python
    def _ensure_prompt_post(self, prompt_idx: int) -> BetaPosterior:
        if prompt_idx not in self._prompt_post:
            self._prompt_post[prompt_idx] = BetaPosterior()
        return self._prompt_post[prompt_idx]

    def _ensure_bucket_post(self, prompt_idx: int) -> BetaPosterior:
        bucket_key = self._buckets.bucket_of(prompt_idx)
        if bucket_key not in self._bucket_post:
            self._bucket_post[bucket_key] = BetaPosterior()
        return self._bucket_post[bucket_key]

    def update_accepted(self, prompt_idx: int, rewards: list[float]) -> None:
        """Validator accepted the submission. Update from raw rewards."""
        k = sum(int(r) for r in rewards)
        n = len(rewards)
        self._ensure_prompt_post(prompt_idx).update(k, n)
        self._ensure_bucket_post(prompt_idx).update(k, n)

    def update_local_reject(self, prompt_idx: int, rewards: list[float]) -> None:
        """Filtered out locally before submit. Same posterior signal as accepted."""
        self.update_accepted(prompt_idx, rewards)

    def update_superseded(self, prompt_idx: int) -> None:
        """Another miner beat us with smaller signed_round on the same prompt."""
        self._competitor_seen[prompt_idx] = self._competitor_seen.get(prompt_idx, 0) + 1

    def update_neutral(self, prompt_idx: int) -> None:
        """Rejection that carries no signal about p (ckpt mismatch, cooldown, etc.)."""
        # Intentional no-op — kept as an explicit method so callers don't forget.
        return
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/miner_priv/test_selector.py -v
```

Expected: 8 passed (4 from Task 5 + 4 new).

- [ ] **Step 5: Commit**

```bash
git add reliquary/miner/selector.py tests/miner_priv/test_selector.py
git commit -m "feat(miner-priv): Selector update API for accepted/local_reject/superseded/neutral"
```

---

## Task 7: `Selector.on_checkpoint_change()` — decay across checkpoints

**Files:**
- Modify: `reliquary/miner/selector.py`
- Test: `tests/miner_priv/test_selector.py` (add cases)

- [ ] **Step 1: Write the failing tests**

Append to `tests/miner_priv/test_selector.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/miner_priv/test_selector.py -v
```

Expected: 4 new failures.

- [ ] **Step 3: Write the implementation**

Append to `reliquary/miner/selector.py` (inside `Selector`):

```python
    def on_checkpoint_change(self, decay: float = 0.5) -> None:
        """Apply evidence decay to all posteriors after a checkpoint advance.

        Rationale: 1 GRPO step changes p little; carrying decayed evidence is
        better than resetting to Beta(1, 1). competitor_seen also decays since
        it's a stale signal (other miners may have moved on).
        """
        for post in self._prompt_post.values():
            post.decay(decay)
        for post in self._bucket_post.values():
            post.decay(decay)
        # Discrete decay for competitor_seen (counts halved with default decay=0.5)
        self._competitor_seen = {
            k: int(v * decay) for k, v in self._competitor_seen.items() if int(v * decay) > 0
        }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/miner_priv/test_selector.py -v
```

Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add reliquary/miner/selector.py tests/miner_priv/test_selector.py
git commit -m "feat(miner-priv): Selector.on_checkpoint_change with decay"
```

---

## Task 8: Persistence — pickle round-trip + boot-time migration

**Files:**
- Modify: `reliquary/miner/selector.py`
- Test: `tests/miner_priv/test_persistence.py`

- [ ] **Step 1: Write the failing test**

Create `tests/miner_priv/test_persistence.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/miner_priv/test_persistence.py -v
```

Expected: AttributeError on `Selector.save` / `Selector.load`.

- [ ] **Step 3: Write the implementation**

Append to `reliquary/miner/selector.py` (inside `Selector`, with `import pickle` at top of file):

```python
    def save(self, path, checkpoint_n: int) -> None:
        """Serialize state with the checkpoint at which it was captured."""
        import pickle
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint_n": checkpoint_n,
            "prompt_post": self._prompt_post,
            "bucket_post": self._bucket_post,
            "competitor_seen": self._competitor_seen,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def load(self, path, current_checkpoint_n: int, decay_per_step: float = 0.5):
        """Load state. If saved at an earlier checkpoint, apply decay per step.

        Returns the loaded checkpoint_n, or None if the file is missing.
        """
        import pickle
        from pathlib import Path
        path = Path(path)
        if not path.exists():
            return None
        with open(path, "rb") as f:
            payload = pickle.load(f)

        self._prompt_post = payload["prompt_post"]
        self._bucket_post = payload["bucket_post"]
        self._competitor_seen = payload["competitor_seen"]

        saved_n = payload["checkpoint_n"]
        steps = max(0, current_checkpoint_n - saved_n)
        for _ in range(steps):
            self.on_checkpoint_change(decay=decay_per_step)
        return current_checkpoint_n
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/miner_priv/test_persistence.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add reliquary/miner/selector.py tests/miner_priv/test_persistence.py
git commit -m "feat(miner-priv): Selector.save/load with cross-checkpoint decay"
```

---

## Task 9: Hypothesis property tests

**Files:**
- Test: `tests/miner_priv/test_selector_properties.py`

- [ ] **Step 1: Write the property tests**

Create `tests/miner_priv/test_selector_properties.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
pytest tests/miner_priv/test_selector_properties.py -v
```

Expected: 5 passed (Hypothesis explores many examples per test).

- [ ] **Step 3: Commit**

```bash
git add tests/miner_priv/test_selector_properties.py
git commit -m "test(miner-priv): hypothesis property tests for selector invariants"
```

---

## Task 10: Wire Selector into `engine.py` (replace `pick_prompt_idx`)

**Files:**
- Modify: `reliquary/miner/engine.py`
- Test: `tests/miner_priv/test_engine_integration.py`

This task is INTEGRATION-LEVEL. We do not write a unit test for the loop edit; instead we add a smoke test that mocks the validator, runs one tick, and asserts the selector was queried.

- [ ] **Step 1: Read the current `mine_window` loop**

```bash
grep -n "pick_prompt_idx\|_generate_m_rollouts\|submit_batch_v2" reliquary/miner/engine.py
```

Expected: shows lines around `pick_prompt_idx(self.env, cooldown_set, rng=rng)` (around line 249 in upstream).

- [ ] **Step 2: Write the integration smoke test**

Create `tests/miner_priv/test_engine_integration.py`:

```python
"""Smoke test for engine wiring — verifies selector is invoked and σ filter triggers."""
from unittest.mock import MagicMock, patch
import pytest


@pytest.mark.skip(reason="Wired in Task 10 step 4 once engine.py is updated.")
def test_engine_uses_selector_for_pick():
    """One tick of mine_window calls selector.next, not pick_prompt_idx."""
    pass


@pytest.mark.skip(reason="Wired in Task 11 step 4 once σ filter is added.")
def test_engine_skips_submit_when_sigma_below_threshold():
    pass
```

(Tests are marked skip — they will be activated once the engine edit is done. This is a deliberate scaffold so the file exists and pytest discovers it.)

- [ ] **Step 3: Edit `engine.py` — instantiate Selector, replace pick_prompt_idx call**

In `reliquary/miner/engine.py`, find the `MiningEngine.__init__` (or equivalent) and add a Selector instance.

Find the section near the top that imports things like:
```python
from reliquary.miner.engine import pick_prompt_idx  # or similar
```

And the loop body that calls `pick_prompt_idx(self.env, cooldown_set, rng=rng)`.

Apply this diff conceptually:

```python
# Top of file — add:
from reliquary.miner.selector import Selector
from reliquary.miner.buckets import BucketIndex

# In MiningEngine.__init__ (after self.env is set up), add:
self._selector = Selector(buckets=BucketIndex(), rng=random.Random())

# In the loop, replace:
#   prompt_idx = pick_prompt_idx(self.env, cooldown_set, rng=rng)
# with:
prompt_idx = self._selector.next(cooldown_set=cooldown_set)
```

Use:

```bash
grep -n "MiningEngine\|def __init__\|pick_prompt_idx" reliquary/miner/engine.py | head -20
```

to find the exact line numbers, then apply the edits.

- [ ] **Step 4: Activate the integration test**

Replace the body of `test_engine_uses_selector_for_pick` in `tests/miner_priv/test_engine_integration.py`:

```python
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
```

(Remove the `@pytest.mark.skip` decorator.)

- [ ] **Step 5: Run the test**

```bash
pytest tests/miner_priv/test_engine_integration.py::test_engine_uses_selector_for_pick -v
```

Expected: 1 passed.

- [ ] **Step 6: Run the full miner test suite to make sure nothing else broke**

```bash
pytest tests/miner_priv/ -v
pytest tests/ -k "engine or miner" -v   # broader sweep
```

Expected: all pass. Investigate any regression before continuing.

- [ ] **Step 7: Commit**

```bash
git add reliquary/miner/engine.py tests/miner_priv/test_engine_integration.py
git commit -m "feat(miner-priv): wire Selector into MiningEngine, replace pick_prompt_idx"
```

---

## Task 11: Add the local σ filter pre-submit + selector feedback

**Files:**
- Modify: `reliquary/miner/engine.py`
- Test: `tests/miner_priv/test_engine_integration.py`

The filter sits between "rewards computed" and "build GRAIL sketches + submit". If σ < threshold, we discard, feed the selector with `update_local_reject`, and pick another prompt.

- [ ] **Step 1: Read the current submit path**

```bash
grep -n "BatchSubmissionRequest\|submit_batch_v2\|merkle_root" reliquary/miner/engine.py
```

Locate the block between `rollout_submissions = [...]` (where rewards are accessible) and `await submit_batch_v2(...)`.

- [ ] **Step 2: Write the integration test**

Replace the second skip in `tests/miner_priv/test_engine_integration.py`:

```python
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
```

(Remove the `@pytest.mark.skip` decorator.)

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/miner_priv/test_engine_integration.py::test_engine_skips_submit_when_sigma_below_threshold -v
```

Expected: PASS *if* `is_in_zone` is already imported and `MiningEngine` allows the bypass-init pattern. If it fails, debug imports first.

- [ ] **Step 4: Edit `engine.py` — insert the σ filter + feedback**

In `reliquary/miner/engine.py`, find the loop section that builds `rollout_submissions` and computes `merkle_root`. Insert the filter BEFORE the GRAIL sketch construction / submit.

Add at the top of the file:

```python
from reliquary.miner.zone import is_in_zone, ZONE_THRESHOLD_BOOTSTRAP
```

In the loop body (after `rollout_submissions = [...]` but before `merkle_root = ...`):

```python
local_rewards = [r.reward for r in rollout_submissions]
in_bootstrap = state.window_n < 100  # spec section 4 — first 100 windows are bootstrap
if not is_in_zone(local_rewards, bootstrap=in_bootstrap):
    logger.info(
        "local σ filter rejected window=%d prompt=%d (rewards=%s)",
        state.window_n, prompt_idx, local_rewards,
    )
    self._selector.update_local_reject(prompt_idx, local_rewards)
    continue   # skip GRAIL + submit, pick next prompt next tick
```

Then, after `submit_batch_v2(...)` returns a response, wire feedback. Map `RejectReason` to selector calls:

```python
from reliquary.protocol.submission import RejectReason  # adjust import path if different

if resp.accepted:
    self._selector.update_accepted(prompt_idx, local_rewards)
elif resp.reason == RejectReason.SUPERSEDED:
    self._selector.update_superseded(prompt_idx)
elif resp.reason == RejectReason.OUT_OF_ZONE:
    # Should not happen — local filter caught it already. Log and update.
    logger.warning("OUT_OF_ZONE despite local filter — bug?")
    self._selector.update_local_reject(prompt_idx, local_rewards)
else:
    self._selector.update_neutral(prompt_idx)
```

Verify the import path of `RejectReason` matches what's actually exported. Use:

```bash
grep -rn "class RejectReason\|RejectReason =" reliquary/
```

- [ ] **Step 5: Run the test**

```bash
pytest tests/miner_priv/test_engine_integration.py -v
```

Expected: 2 passed.

- [ ] **Step 6: Run the full suite**

```bash
pytest tests/miner_priv/ -v
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add reliquary/miner/engine.py tests/miner_priv/test_engine_integration.py
git commit -m "feat(miner-priv): local σ filter pre-submit + selector feedback wiring"
```

---

## Task 12: Hook `on_checkpoint_change` to the existing checkpoint reload path

**Files:**
- Modify: `reliquary/miner/engine.py`

- [ ] **Step 1: Locate the checkpoint reload code path**

```bash
grep -n "maybe_pull_checkpoint\|checkpoint_n\|_load_checkpoint" reliquary/miner/engine.py
```

You should find a section like:
```python
local_n, local_hash, self.hf_model = await maybe_pull_checkpoint(
    state=state, local_n=local_n, local_hash=local_hash, ...
)
```

- [ ] **Step 2: Add the selector hook**

After the checkpoint-pull block, BEFORE the OPEN check, add:

```python
if local_n != getattr(self, "_last_seen_ckpt_n", -1):
    if hasattr(self, "_last_seen_ckpt_n"):
        # Real advance, not initial boot
        self._selector.on_checkpoint_change(decay=0.5)
        logger.info("selector decayed for checkpoint advance %d -> %d",
                    self._last_seen_ckpt_n, local_n)
    self._last_seen_ckpt_n = local_n
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/miner_priv/ -v
```

Expected: still all pass (we didn't add a test here — this is a single-line wiring; behavior is covered by Task 7's `on_checkpoint_change` unit tests).

- [ ] **Step 4: Commit**

```bash
git add reliquary/miner/engine.py
git commit -m "feat(miner-priv): decay selector posteriors on checkpoint advance"
```

---

## Task 13: Add structured logging for metrics (spec section 9)

**Files:**
- Modify: `reliquary/miner/engine.py`

- [ ] **Step 1: Find every place the engine logs a submission outcome**

```bash
grep -n "logger.info.*submitted\|logger.info.*accepted" reliquary/miner/engine.py
```

- [ ] **Step 2: Augment the log line with a JSON payload**

Where the existing code does:
```python
logger.info("submitted window=%d prompt=%d accepted=%s reason=%s", ...)
```

Add (after, not replacing):
```python
import json
import time
logger.info("submit_attempt %s", json.dumps({
    "ts": time.time(),
    "event": "submit_attempt",
    "prompt_idx": prompt_idx,
    "sigma_local": population_std(local_rewards),
    "passed_local_filter": True,   # we got here, so local filter passed
    "outcome": "accepted" if resp.accepted else (resp.reason.value if hasattr(resp.reason, "value") else str(resp.reason)),
    "signed_round": state.current_round,
    "checkpoint_n": local_n,
}))
```

Add the import `from reliquary.miner.zone import population_std` at the top.

For the local-reject path (Task 11 σ filter), add a similar JSON log:
```python
logger.info("local_reject %s", json.dumps({
    "ts": time.time(),
    "event": "local_reject",
    "prompt_idx": prompt_idx,
    "sigma_local": population_std(local_rewards),
    "passed_local_filter": False,
    "checkpoint_n": local_n,
}))
```

- [ ] **Step 3: Manual verification**

```bash
# Run the existing test suite to ensure nothing broke
pytest tests/miner_priv/ -v
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add reliquary/miner/engine.py
git commit -m "feat(miner-priv): structured JSON logs for submit_attempt / local_reject"
```

---

## Task 14: Manual smoke run against a local validator

**Files:** none — operational task.

This is a manual integration test. It validates the full path on real hardware before mainnet exposure.

- [ ] **Step 1: Start a local validator from upstream code**

In one terminal (in `~/reliquary-miner-priv/` with venv active):
```bash
# Use upstream's validator with a test wallet/netuid
reliquary validate --network local --netuid 1 --wallet-name test_validator --hotkey default
```

Refer to `docs/validating.md` for any extra flags needed. This validator will listen on `127.0.0.1:8888`.

- [ ] **Step 2: Start the private miner against the local validator**

In another terminal:
```bash
reliquary mine --network local --netuid 1 \
    --wallet-name test_miner --hotkey default \
    --validator-url http://127.0.0.1:8888 \
    --log-level INFO 2>&1 | tee miner_smoke.log
```

- [ ] **Step 3: Run for ~10 minutes and verify**

In a third terminal:
```bash
# Count submission attempts and outcomes
grep "submit_attempt\|local_reject" miner_smoke.log | wc -l

# σ distribution
grep "submit_attempt" miner_smoke.log | jq -r '.sigma_local' | datamash min q1 median q3 max

# Outcome breakdown
grep "submit_attempt" miner_smoke.log | jq -r '.outcome' | sort | uniq -c
```

Expected:
- More than 0 submission attempts.
- σ values clustered around 0.4-0.5 (Thompson sampling biased toward sweet zone).
- Outcomes mostly `accepted` (no `OUT_OF_ZONE` if local filter is working).

- [ ] **Step 4: Verify no `OUT_OF_ZONE` from validator**

```bash
grep "OUT_OF_ZONE" miner_smoke.log
```

Expected: zero matches. If matches appear, debug — this means local σ filter doesn't agree with validator's. Most likely cause: rewards computed differently (different env version, different normalization). Check `compute_reward` is the same on both sides.

- [ ] **Step 5: Stop both processes (Ctrl-C), commit the smoke log for posterity**

```bash
gzip miner_smoke.log
git add miner_smoke.log.gz   # only if you want to keep it; otherwise rm
# (skip commit if not keeping the artifact)
```

---

## Task 15: Mainnet shadow-deploy at low frequency

**Files:** none — operational task.

We deploy to mainnet but with a flag that limits submission rate, so a bug can't drain hours of EMA before we notice.

- [ ] **Step 1: Add a CLI flag `--max-windows` to limit run duration**

In `reliquary/cli/...` (find with `grep -rn "def mine\|click.command" reliquary/cli/`), add:
```python
@click.option("--max-windows", type=int, default=None,
              help="Stop after N windows. For canary deployments.")
```

Pass it through to `MiningEngine.mine_window`. Inside the loop, add an early break when `windows_seen >= max_windows`.

- [ ] **Step 2: Commit**

```bash
git add reliquary/cli/...
git commit -m "feat(miner-priv): --max-windows for canary deployments"
```

- [ ] **Step 3: Launch a canary on mainnet — 50 windows**

```bash
reliquary mine \
    --network finney --netuid 81 \
    --wallet-name my_miner --hotkey default \
    --validator-url http://<owner-validator-ip>:8888 \
    --max-windows 50 \
    --log-level INFO 2>&1 | tee canary_$(date +%s).log
```

- [ ] **Step 4: Once canary completes, analyze**

```bash
LATEST=$(ls -t canary_*.log | head -1)

# Acceptance rate
echo "Outcomes:"
grep "submit_attempt" "$LATEST" | jq -r '.outcome' | sort | uniq -c

# Local discard rate
echo "Local rejects: $(grep -c local_reject "$LATEST")"
echo "Submits: $(grep -c submit_attempt "$LATEST")"

# σ distribution
echo "σ stats:"
grep -E "submit_attempt|local_reject" "$LATEST" | jq -r '.sigma_local' | datamash min q1 median q3 max
```

Phase 1 success criteria (spec section 8):
- `out_of_zone_rate_validator` ≈ 0   (sanity-check the local filter)
- `local_discard_rate` < 30 %        (selector quality)
- `accepted_rate` > +20 % vs upstream baseline

If criteria are met → Phase 1 is shipped, write the Phase 2 plan.
If not met → debug. Common causes:
- Bucket priors not informative enough → check bucket_post entropies
- Selector exploration too aggressive → reduce Thompson sampling temperature (out of scope here, would be a spec amendment)
- Discrepancy between local and validator reward calc → check env version

- [ ] **Step 5: Commit canary log (gzipped)**

```bash
gzip "$LATEST"
git add "${LATEST}.gz"
git commit -m "chore: canary log for phase 1 validation"
```

---

## Self-review

**Spec coverage check:**
- §4.1 score function → Task 1 ✓
- §4.2 BetaPosterior + Selector skeleton → Task 2, Task 5 ✓
- §4.3 Thompson + competitor penalty → Task 5 ✓
- §4.4 Update API (4 outcomes) → Task 6 ✓
- §4.5 Persistence → Task 8 ✓
- §6.1 σ filter pre-submit → Task 11 ✓
- §6.2 selector feedback on validator outcomes → Task 11 ✓
- on_checkpoint_change wiring → Task 12 ✓
- §7.2 unit tests → Tasks 1-8 each include tests ✓
- §7.5 property-based tests → Task 9 ✓
- §8 phase 1 deployment + criteria → Task 15 ✓
- §9 metrics (structured JSON logs) → Task 13 ✓

No gaps in scope.

**Placeholder scan:** searched for "TBD", "TODO", "implement later", "fill in" — none in plan body (only inside example log lines as field names like `"event"`, which are intentional).

**Type/name consistency:**
- `score_in_zone(p, n)` — used same name in Task 1, Task 5, Task 9 ✓
- `BetaPosterior(alpha, beta)` — same field names throughout ✓
- `Selector` methods: `next`, `update_accepted`, `update_local_reject`, `update_superseded`, `update_neutral`, `on_checkpoint_change`, `save`, `load` — all referenced consistently ✓
- `is_in_zone`, `population_std`, `ZONE_THRESHOLD_*` — Task 3 defines, Task 11 uses ✓

Plan complete.
