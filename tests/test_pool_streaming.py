"""Feed the pool as each prompt finishes, not once the whole bake is done.

Root cause of zero submissions (measured 2026-07-21): `_pre_bake_batch` returns
all entries only after every prompt is baked (~185s for 6). By then the window
has flipped, the prompt-range slice has moved, and the fire path drops every
entry as an "out-of-slice straggler" — hence `pool empty (kept=0 after cooldown
filter)` on every fire. The first prompt was ready at 44s, well inside the
window, and was thrown away because we waited for the other five.

Streaming each entry into the pool the moment it is baked lets the trigger loop
fire it mid-window, while its slice is still current.
"""

from __future__ import annotations

import asyncio

import pytest


class _Engine:
    """Minimal stand-in wired like MiningEngine for the streaming path."""

    def __init__(self, n_prompts=4):
        from reliquary.miner.engine import MiningEngine

        self.e = MiningEngine.__new__(MiningEngine)
        self.e._pool = []
        self.e._pool_lock = asyncio.Lock()
        self.e._phase1_cache = {}
        self.e._cached_randomness = "ab" * 32
        self.e._local_hash = "ckpt"
        # pool size observed at the START of each per-prompt bake
        self.observed = []
        self.n = n_prompts

        def _prefetch(problems, prompt_indices, *, randomness, env):
            self.prefetched = list(prompt_indices)
            return len(prompt_indices)

        def _entry(idx, prob, expected_ckpt_n, env):
            self.observed.append(len(self.e._pool))
            return {"prompt_idx": idx}

        self.e._prefetch_phase1 = _prefetch
        self.e._pre_bake_entry = _entry


PROBLEMS = [{"prompt": f"p{i}"} for i in range(4)]
IDXS = [10, 11, 12, 13]


def _run(eng):
    return asyncio.run(
        eng.e._bake_streaming(PROBLEMS, IDXS, expected_ckpt_n=1, env=None)
    )


def test_generation_is_still_batched_once_for_all_prompts():
    """The phase-1 win must survive: one prefetch covering every prompt."""
    eng = _Engine()
    _run(eng)
    assert eng.prefetched == IDXS


def test_each_entry_lands_in_the_pool_before_the_next_prompt_is_baked():
    """This is the fix: entry k is fireable while prompt k+1 is still baking."""
    eng = _Engine()
    _run(eng)
    # pool size seen at the start of each bake: 0, 1, 2, 3
    assert eng.observed == [0, 1, 2, 3]


def test_all_entries_end_up_in_the_pool_in_order():
    eng = _Engine()
    _run(eng)
    assert [e["prompt_idx"] for e in eng.e._pool] == IDXS


def test_a_failed_prompt_does_not_block_the_rest():
    """One bad prompt must not cost the whole bake — the others still fire."""
    eng = _Engine()
    orig = eng.e._pre_bake_entry

    def _entry(idx, prob, expected_ckpt_n, env):
        if idx == IDXS[1]:
            raise RuntimeError("boom")
        return orig(idx, prob, expected_ckpt_n, env)

    eng.e._pre_bake_entry = _entry
    _run(eng)
    assert [e["prompt_idx"] for e in eng.e._pool] == [IDXS[0], IDXS[2], IDXS[3]]


def test_a_none_entry_is_skipped_not_appended():
    """_pre_bake_entry returns None for out-of-zone groups."""
    eng = _Engine()

    def _entry(idx, prob, expected_ckpt_n, env):
        return None if idx == IDXS[0] else {"prompt_idx": idx}

    eng.e._pre_bake_entry = _entry
    _run(eng)
    assert [e["prompt_idx"] for e in eng.e._pool] == IDXS[1:]


def test_stale_checkpoint_entries_are_dropped_when_the_policy_says_so():
    """DROP_POOL_ON_CKPT=1: an entry baked under an older checkpoint is invalid
    under forced-seed and must not reach the pool."""
    import reliquary.miner.engine as eng

    e = _Engine()
    e.e._local_n = 5
    e.e._prefetch_phase1 = lambda *a, **k: 0

    def _entry(idx, prob, expected_ckpt_n, env):
        # prompt 11 was baked one checkpoint behind
        return {"prompt_idx": idx, "checkpoint_n": 4 if idx == 11 else 5}

    e.e._pre_bake_entry = _entry
    orig = eng.drop_pool_on_ckpt_advance
    eng.drop_pool_on_ckpt_advance = lambda: True
    try:
        _run(e)
    finally:
        eng.drop_pool_on_ckpt_advance = orig
    assert [x["prompt_idx"] for x in e.e._pool] == [10, 12, 13]


def test_the_event_loop_gets_control_between_prompts():
    """Without an await boundary the trigger loop could never fire mid-bake."""
    eng = _Engine()
    ticks = [0]

    async def _beat():
        while True:
            ticks[0] += 1
            await asyncio.sleep(0)

    async def _main():
        hb = asyncio.create_task(_beat())
        await eng.e._bake_streaming(PROBLEMS, IDXS, expected_ckpt_n=1, env=None)
        hb.cancel()

    asyncio.run(_main())
    assert ticks[0] >= len(IDXS), (
        "bake never yielded to the loop; the trigger loop cannot fire mid-window"
    )
