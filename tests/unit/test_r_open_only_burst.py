"""Unit tests for the R_open-only burst policy.

See docs/superpowers/specs/2026-05-16-r-open-only-burst-design.md for the
full design rationale. Each test pins one behavior of the simplified
trigger loop: fire exactly once per window when pool is non-empty and
state is OPEN-with-randomness; hold otherwise.
"""

import asyncio
import inspect
from types import SimpleNamespace

import pytest

from reliquary.miner import engine as engine_module
from reliquary.miner.engine import _should_fire_for_window
from reliquary.protocol.submission import WindowState


def _state(window_n: int = 7, state: WindowState = WindowState.OPEN,
           randomness: str = "deadbeef") -> SimpleNamespace:
    return SimpleNamespace(
        window_n=window_n,
        state=state,
        randomness=randomness,
        cooldown_prompts=[],
    )


def test_fire_gate_open_with_pool_not_yet_fired():
    assert _should_fire_for_window(_state(), set(), set(), pool_size=1) is True


def test_fire_gate_holds_when_window_already_fired():
    assert _should_fire_for_window(_state(window_n=7), {7}, set(), pool_size=1) is False


def test_fire_gate_holds_when_window_forfeited():
    """If we saw OPEN with empty pool earlier in the window (forfeit set),
    a later pool refill within the SAME window must NOT trigger a fire.
    Under the R_open-only policy, a forfeited window is committed-to-skip
    rather than fired late at R_open+k — late fires sit behind R_open
    competitors at seal and routinely return batch_filled.
    """
    assert _should_fire_for_window(_state(window_n=7), set(), {7}, pool_size=1) is False


def test_fire_gate_holds_when_pool_empty():
    assert _should_fire_for_window(_state(), set(), set(), pool_size=0) is False


def test_fire_gate_holds_when_not_open():
    s = _state(state=WindowState.PUBLISHING)
    assert _should_fire_for_window(s, set(), set(), pool_size=4) is False


def test_fire_gate_holds_when_randomness_empty():
    s = _state(randomness="")
    assert _should_fire_for_window(s, set(), set(), pool_size=4) is False


def test_compute_offset_sub_second_anchor_is_centered():
    """Regression test for the WIP fix that moved the anchor from
    period/4 to period/2. A neutral anchor means: across a sweep of
    t_fetch values uniformly covering one drand period, the mean
    estimation error is zero and the worst-case |error| is bounded by
    period/2. period/4 anchor would skew the mean toward STALE_ROUND
    by ~0.75 s on quicknet — incompatible with the validator's
    BACKWARD_TOLERANCE=0 (Catalyst commit 2d8ac38).
    """
    from reliquary.miner.engine import _compute_offset_sub_second

    period = 3
    genesis_time = 1_700_000_000
    r_drand = 1000
    t_round_start = genesis_time + (r_drand - 1) * period

    # True offset is zero — sweep t_fetch across the round; the estimate
    # is t_anchor - t_fetch where t_anchor is fixed at round_start + period/2.
    samples = [
        _compute_offset_sub_second(
            r_drand, t_round_start + k / 100 * period,
            {"genesis_time": genesis_time, "period": period},
        )
        for k in range(101)
    ]
    mean_err = sum(samples) / len(samples)
    assert abs(mean_err) < 1e-6, (
        f"sub-second offset estimator is biased (mean={mean_err:+.4f}); "
        "anchor must be at t_round_start + period/2, not period/4."
    )
    assert max(abs(s) for s in samples) <= period / 2 + 1e-6


class _StubMiningEngine:
    """Minimal MiningEngine surface needed to exercise _trigger_loop's
    fire decision in a tight loop. Real fields the loop touches:
    _pool, _pool_lock, _fired_windows, _cached_cooldown, _local_n,
    _local_hash, hf_model. Methods called: _fire_for_window, _load_checkpoint.
    """
    def __init__(self):
        self._pool = [{"prompt_idx": 0, "marker": "entry"}]
        self._pool_lock = asyncio.Lock()
        self._fired_windows: set[int] = set()
        self._logged_empty_windows: set[int] = set()
        self._inflight_fire_tasks: set = set()
        self._cached_cooldown: set[int] = set()
        self._local_n = 1
        self._local_hash = "hash"
        self.hf_model = None
        self.fire_calls: list[int] = []
        self._pool_dir = None

    async def _fire_for_window(self, state, url, client, results):
        self.fire_calls.append(state.window_n)


@pytest.mark.asyncio
async def test_trigger_loop_fires_once_per_window(monkeypatch):
    """Ten ticks reporting OPEN for the same window_n must trigger exactly
    one _fire_for_window call.
    """
    me = _StubMiningEngine()
    state = _state(window_n=11)

    poll_count = {"n": 0}

    async def fake_get_state(url, *, client):
        poll_count["n"] += 1
        if poll_count["n"] >= 10:
            raise StopAsyncIteration  # break the infinite loop
        # Return (state, resp, t_send, t_recv)
        return state, SimpleNamespace(headers={}), 0.0, 0.0

    # Patch at the submitter module so the local import inside _trigger_loop
    # picks up the stub (the function re-imports on every call).
    monkeypatch.setattr(
        "reliquary.miner.submitter.get_window_state_v2_with_resp",
        fake_get_state,
    )

    # maybe_pull_checkpoint must not advance the checkpoint
    async def fake_pull(**kwargs):
        return kwargs["local_n"], kwargs["local_hash"], kwargs["local_model"]
    monkeypatch.setattr(engine_module, "maybe_pull_checkpoint", fake_pull)

    with pytest.raises(StopAsyncIteration):
        await engine_module.MiningEngine._trigger_loop(
            me, url="http://stub", client=None, results=[],
        )

    assert me.fire_calls == [11], (
        f"expected exactly one fire for window 11, got {me.fire_calls}"
    )


@pytest.mark.asyncio
async def test_trigger_loop_fires_once_per_new_window(monkeypatch):
    """Across a window transition (11 → 12), each window must fire exactly
    once. This pins the per-new-window half of the single-fire invariant
    that the previous test does not cover.
    """
    me = _StubMiningEngine()

    # Ticks 1-3 see window 11; ticks 4-8 see window 12; tick 9 breaks.
    states = {
        1: _state(window_n=11), 2: _state(window_n=11), 3: _state(window_n=11),
        4: _state(window_n=12), 5: _state(window_n=12), 6: _state(window_n=12),
        7: _state(window_n=12), 8: _state(window_n=12),
    }
    poll_count = {"n": 0}

    async def fake_get_state(url, *, client):
        poll_count["n"] += 1
        if poll_count["n"] >= 9:
            raise StopAsyncIteration
        return states[poll_count["n"]], SimpleNamespace(headers={}), 0.0, 0.0

    monkeypatch.setattr(
        "reliquary.miner.submitter.get_window_state_v2_with_resp",
        fake_get_state,
    )

    async def fake_pull(**kwargs):
        return kwargs["local_n"], kwargs["local_hash"], kwargs["local_model"]
    monkeypatch.setattr(engine_module, "maybe_pull_checkpoint", fake_pull)

    with pytest.raises(StopAsyncIteration):
        await engine_module.MiningEngine._trigger_loop(
            me, url="http://stub", client=None, results=[],
        )

    assert me.fire_calls == [11, 12], (
        f"expected one fire per new window, got {me.fire_calls}"
    )


def test_no_boundary_safety_in_submit_entry():
    """The RELIQUARY_DRAND_BOUNDARY_SAFETY_S sleep block must be deleted
    from _submit_entry. Structural test rather than behavioral because
    the block is deep inside a function that requires a real baked
    entry + httpx client to exercise behaviorally; the structural
    assertion is precise enough for a one-shot refactor.
    """
    src = inspect.getsource(engine_module.MiningEngine._submit_entry)
    assert "BOUNDARY_SAFETY" not in src, (
        "RELIQUARY_DRAND_BOUNDARY_SAFETY_S env-var read still present "
        "in _submit_entry"
    )
    assert "seconds_until_next_drand_boundary" not in src, (
        "boundary-cross detection still present in _submit_entry"
    )


@pytest.mark.asyncio
async def test_fire_for_window_empty_pool_returns_no_post():
    """With pool empty, _fire_for_window must return cleanly without
    calling _submit_entry, without raising, and without populating
    results. Preserves the existing 'no entries → no work' behavior
    after the max_fires-parameter removal.
    """
    me = _StubMiningEngine()
    me._pool = []
    state = _state(window_n=5)
    results: list = []
    await engine_module.MiningEngine._fire_for_window(
        me, state, "http://stub", None, results,
    )
    assert results == []


@pytest.mark.asyncio
async def test_pool_empty_at_flip_logged_once_per_window(monkeypatch, caplog):
    """When state is OPEN but the pool is empty, the trigger loop must log
    once per window_n — not every 5 ms tick.
    """
    import logging

    me = _StubMiningEngine()
    me._pool = []  # empty pool

    state = _state(window_n=99)
    poll_count = {"n": 0}

    async def fake_get_state(url, *, client):
        poll_count["n"] += 1
        if poll_count["n"] >= 10:
            raise StopAsyncIteration
        return state, SimpleNamespace(headers={}), 0.0, 0.0

    monkeypatch.setattr(
        "reliquary.miner.submitter.get_window_state_v2_with_resp",
        fake_get_state,
    )

    async def fake_pull(**kwargs):
        return kwargs["local_n"], kwargs["local_hash"], kwargs["local_model"]
    monkeypatch.setattr(engine_module, "maybe_pull_checkpoint", fake_pull)

    with caplog.at_level(logging.WARNING, logger="reliquary.miner.engine"):
        with pytest.raises(StopAsyncIteration):
            await engine_module.MiningEngine._trigger_loop(
                me, url="http://stub", client=None, results=[],
            )

    empty_logs = [
        r for r in caplog.records
        if "pool empty at OPEN" in r.message and "window=99" in r.message
    ]
    assert len(empty_logs) == 1, (
        f"expected exactly one 'pool empty at OPEN window=99' log line, "
        f"got {len(empty_logs)}: {[r.message for r in empty_logs]}"
    )


@pytest.mark.asyncio
async def test_pool_empty_log_suppressed_after_successful_fire(monkeypatch, caplog):
    """After a window has fired successfully and the pool drained, subsequent
    ticks of the SAME window must NOT log 'pool empty at OPEN'. The 'window
    already fired' check on the elif branch is what suppresses the spurious
    warning; if it regresses, this test fails.
    """
    import logging

    me = _StubMiningEngine()
    # Pool starts with one entry — first tick fires and drains it.
    me._pool = [{"prompt_idx": 0, "marker": "entry"}]

    # _fire_for_window stub drains the pool on call to simulate a real burst.
    async def fake_fire(state, url, client, results):
        async with me._pool_lock:
            me._pool = []
        me.fire_calls.append(state.window_n)
    me._fire_for_window = fake_fire

    state = _state(window_n=77)
    poll_count = {"n": 0}

    async def fake_get_state(url, *, client):
        poll_count["n"] += 1
        if poll_count["n"] >= 10:
            raise StopAsyncIteration
        return state, SimpleNamespace(headers={}), 0.0, 0.0

    monkeypatch.setattr(
        "reliquary.miner.submitter.get_window_state_v2_with_resp",
        fake_get_state,
    )

    async def fake_pull(**kwargs):
        return kwargs["local_n"], kwargs["local_hash"], kwargs["local_model"]
    monkeypatch.setattr(engine_module, "maybe_pull_checkpoint", fake_pull)

    with caplog.at_level(logging.WARNING, logger="reliquary.miner.engine"):
        with pytest.raises(StopAsyncIteration):
            await engine_module.MiningEngine._trigger_loop(
                me, url="http://stub", client=None, results=[],
            )

    spurious = [
        r for r in caplog.records
        if "pool empty at OPEN" in r.message and "window=77" in r.message
    ]
    assert me.fire_calls == [77], "expected exactly one fire for window 77"
    assert spurious == [], (
        f"after a successful fire drained the pool, subsequent ticks must "
        f"NOT log 'pool empty at OPEN'. Got {len(spurious)} spurious logs: "
        f"{[r.message for r in spurious]}"
    )
