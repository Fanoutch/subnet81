"""Adaptive slot allocator across envs, driven by our own /verdicts outcomes.

Pure logic, no I/O. The engine feeds outcomes via ``record_outcome(env,
rewarded)`` (mapping each verdict back to the env it submitted), and reads
``target_slots()`` at window open to decide how many of the global slot budget
go to each env. Allocation is proportional to the EMA reward-rate per env, with
an exploration floor so no env is ever starved (we keep measuring it, and catch
the moment it becomes lucrative).
"""
from __future__ import annotations

import itertools

_NEUTRAL_YIELD = 0.5  # cold-start prior so unseen envs get a fair share


class MixController:
    def __init__(self, envs, total_slots: int = 8, slot_floor: int = 1,
                 alpha: float = 0.1) -> None:
        self.envs = list(envs)
        self.total = int(total_slots)
        self.floor = int(slot_floor)
        self.alpha = float(alpha)
        self._yield: dict[str, float | None] = {e: None for e in self.envs}

    def record_outcome(self, env: str, rewarded: bool) -> None:
        """Update the EMA reward-rate for ``env``. Unknown envs are ignored."""
        if env not in self._yield:
            return
        x = 1.0 if rewarded else 0.0
        prev = self._yield[env]
        self._yield[env] = x if prev is None else (1 - self.alpha) * prev + self.alpha * x

    def _effective_yield(self, env: str) -> float:
        y = self._yield[env]
        return _NEUTRAL_YIELD if y is None else y

    def target_slots(self) -> dict[str, int]:
        """Return {env: slots}, summing exactly to ``total``, floor-respecting."""
        n = len(self.envs)
        alloc = {e: self.floor for e in self.envs}
        remaining = self.total - self.floor * n
        if remaining <= 0:
            if remaining < 0:
                # floors over-subscribe the budget — clamp to an even split
                base = self.total // n
                out = {e: base for e in self.envs}
                for e in self.envs[: self.total - base * n]:
                    out[e] += 1
                return out
            return alloc
        weights = {e: max(self._effective_yield(e), 1e-9) for e in self.envs}
        wsum = sum(weights.values())
        quota = {e: remaining * weights[e] / wsum for e in self.envs}
        base = {e: int(quota[e]) for e in self.envs}
        for e in self.envs:
            alloc[e] += base[e]
        leftover = remaining - sum(base.values())
        order = sorted(self.envs, key=lambda e: quota[e] - base[e], reverse=True)
        for e in order[:leftover]:
            alloc[e] += 1
        return alloc


_TIE_ROTATION = itertools.count()


def pick_bake_env(target_slots: dict[str, int], pool_counts: dict[str, int]) -> str:
    """Env to bake next: the one furthest below its target share in the pool.

    deficit(env) = target_slots[env] - pool_counts.get(env, 0). The largest
    deficit wins; TIES ROTATE between the tied envs.

    Rotation matters because ties are the normal case, not an edge case: while
    the sigma filter rejects every group, nothing ever reaches the pool, so
    pool_counts stays all-zero and the deficits are permanently equal. The old
    strict ``>`` then returned the FIRST env every time — measured 2026-07-21 as
    130 math bakes against 5 code bakes (26:1), effectively starving the code
    env even though it is far less contested and its unmined emissions burn.

    With a single env this still always returns that env.
    """
    best_deficit: int | None = None
    tied: list[str] = []
    for env, target in target_slots.items():
        deficit = target - pool_counts.get(env, 0)
        if best_deficit is None or deficit > best_deficit:
            best_deficit, tied = deficit, [env]
        elif deficit == best_deficit:
            tied.append(env)
    assert tied, "target_slots must be non-empty"
    if len(tied) == 1:
        return tied[0]
    return tied[next(_TIE_ROTATION) % len(tied)]


def entry_env_name(entry: dict, default: str) -> str:
    """Env that baked a pool entry; falls back to ``default`` for legacy
    disk-reloaded entries (pre-multi-env persistence) that lack the key."""
    return entry.get("env_name") or default
