"""Private miner — Bayesian prompt selector.

Two-level Beta posteriors (per-prompt × per-bucket) with Thompson sampling.
See docs/superpowers/specs/2026-05-03-optimized-miner-design.md section 4.
"""
from __future__ import annotations

import pickle
import random
from dataclasses import dataclass
from math import comb, exp
from pathlib import Path


def score_in_zone(p: float, n: int = 8, k_lo: int = 2, k_hi: int = 6) -> float:
    """P(k_lo <= X <= k_hi | X ~ Binomial(n, p)).

    For Reliquary MATH (binary rewards), in-zone means 2 <= X <= 6 successes
    out of n=8, which corresponds to σ >= 0.43.
    """
    return sum(
        comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
        for k in range(k_lo, k_hi + 1)
    )


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
        base = score_in_zone(p_sample, n=8)
        penalty = exp(-self._gamma * self._competitor_seen.get(idx, 0))
        return base * penalty

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
        """Another miner claimed this prompt first (TCP-arrival FIFO, v2.2)."""
        self._competitor_seen[prompt_idx] = self._competitor_seen.get(prompt_idx, 0) + 1

    def update_neutral(self, prompt_idx: int) -> None:
        """Rejection that carries no signal about p (ckpt mismatch, cooldown, etc.)."""
        # Intentional no-op — kept as an explicit method so callers don't forget.
        return

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
            k: int(v * decay) for k, v in self._competitor_seen.items()
        }

    def next(
        self,
        cooldown_set: set[int],
        eligible: set[int] | None = None,
    ) -> int:
        """Pick the prompt_idx with the highest Thompson-sampled in-zone score.

        ``eligible`` (when provided) restricts the candidate pool — used to
        pre-filter prompts whose static difficulty (e.g. MATH level) sits
        outside the Goldilocks band, so the selector spends 0 windows
        learning what an external prior already knows.
        """
        n = len(self._buckets)
        if eligible is None:
            candidates = [i for i in range(n) if i not in cooldown_set]
        else:
            candidates = [i for i in eligible if i not in cooldown_set]
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

    def save(self, path, checkpoint_n: int) -> None:
        """Serialize state with the checkpoint at which it was captured.

        Atomic write (tempfile + rename) so concurrent miner processes
        sharing the same path can't corrupt each other's writes.
        """
        import os
        import tempfile
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint_n": checkpoint_n,
            "prompt_post": self._prompt_post,
            "bucket_post": self._bucket_post,
            "competitor_seen": self._competitor_seen,
        }
        fd, tmp_path = tempfile.mkstemp(
            prefix=path.name + ".", dir=str(path.parent),
        )
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(payload, f)
            os.replace(tmp_path, path)   # atomic on POSIX
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def load(self, path, current_checkpoint_n: int, decay_per_step: float = 0.5):
        """Load state. If saved at an earlier checkpoint, apply decay per step.

        Returns the loaded checkpoint_n, or None if the file is missing.
        """
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
