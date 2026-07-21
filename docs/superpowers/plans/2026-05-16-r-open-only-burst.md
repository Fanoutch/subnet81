# R_open-only burst implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the single-fire-per-window R_open burst policy described in `docs/superpowers/specs/2026-05-16-r-open-only-burst-design.md`.

**Architecture:** Surgical edits to `reliquary/miner/engine.py`. Extract the fire gate into a pure helper, replace the multi-fire state machine in `_trigger_loop` with a `_fired_windows: set[int]`, drop the boundary-safety sleep from `_fire_for_window`. Tests use unittest-style with stubbed `WindowState`.

**Tech Stack:** Python 3.11, asyncio, pytest, no new dependencies.

## Working-tree note

At plan creation the user had two unstaged edits:
- `M reliquary/miner/engine.py` — `_compute_offset_sub_second` anchor moved to `period/2` and `_apply_offset_from_validator_response` floor compensation raised to `+0.5`. These are pre-requisites of this plan but are an independent change. **Commit them first** (suggested message: `fix(miner): center sub-second drand anchor + raise floor comp to 0.5`) before starting Task 1.
- `?? tests/unit/test_clock_offset.py` — regression test for the floor comp. Goes with the engine.py commit above.

If those have already been committed, skip this note.

## File map

- Modify: `~/reliquary-miner-priv/reliquary/miner/engine.py`
  - Lines 407-408 (state init): remove `_fires_per_window`, `_last_fire_ts`; add `_fired_windows` set.
  - Lines 544-693 (`_trigger_loop`): replace multi-fire logic with helper-based single fire.
  - Lines 695-752 (`_fire_for_window`): drop `max_fires` parameter, drop boundary-safety sleep, drop `_fires_per_window` accounting.
  - Module-level: add `_should_fire_for_window(state, fired_windows, pool_size)` helper near `_current_drand_round_at_send`.
- Create: `~/reliquary-miner-priv/tests/unit/test_r_open_only_burst.py` — all unit tests for this work.

---

### Task 1: Pure fire gate helper

**Files:**
- Create: `tests/unit/test_r_open_only_burst.py`
- Modify: `reliquary/miner/engine.py` (add module-level helper near `_current_drand_round_at_send`)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_r_open_only_burst.py` with:

```python
"""Unit tests for the R_open-only burst policy.

See docs/superpowers/specs/2026-05-16-r-open-only-burst-design.md for the
full design rationale. Each test pins one behavior of the simplified
trigger loop: fire exactly once per window when pool is non-empty and
state is OPEN-with-randomness; hold otherwise.
"""

from types import SimpleNamespace

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
    assert _should_fire_for_window(_state(), set(), pool_size=1) is True


def test_fire_gate_holds_when_window_already_fired():
    assert _should_fire_for_window(_state(window_n=7), {7}, pool_size=1) is False


def test_fire_gate_holds_when_pool_empty():
    assert _should_fire_for_window(_state(), set(), pool_size=0) is False


def test_fire_gate_holds_when_not_open():
    s = _state(state=WindowState.PUBLISHING)
    assert _should_fire_for_window(s, set(), pool_size=4) is False


def test_fire_gate_holds_when_randomness_empty():
    s = _state(randomness="")
    assert _should_fire_for_window(s, set(), pool_size=4) is False


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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/reliquary-miner-priv && pytest tests/unit/test_r_open_only_burst.py -v
```

Expected: `ImportError: cannot import name '_should_fire_for_window' from 'reliquary.miner.engine'`

- [ ] **Step 3: Add the helper**

In `reliquary/miner/engine.py`, insert this function just **after** `_current_drand_round_at_send` (around line 180), before `_apply_offset_from_validator_response`:

```python
def _should_fire_for_window(
    state, fired_windows: set[int], pool_size: int,
) -> bool:
    """True iff the trigger loop should fire a burst right now.

    Pure function so the gate is unit-testable without spinning up an
    event loop. The four conditions mirror the design doc
    (specs/2026-05-16-r-open-only-burst-design.md):
      * the window hasn't already been fired,
      * /state reports OPEN,
      * the validator has published randomness,
      * the pool has at least one bakeable entry.
    """
    from reliquary.protocol.submission import WindowState
    return (
        state.window_n not in fired_windows
        and state.state == WindowState.OPEN
        and bool(state.randomness)
        and pool_size > 0
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd ~/reliquary-miner-priv && pytest tests/unit/test_r_open_only_burst.py -v
```

Expected: 6 passed (5 gate tests + the sub-second anchor regression test).

- [ ] **Step 5: Commit**

```bash
cd ~/reliquary-miner-priv && git add tests/unit/test_r_open_only_burst.py reliquary/miner/engine.py && git commit -m "feat(miner): extract pure fire gate _should_fire_for_window

Helper makes the trigger-loop fire decision unit-testable without
spinning up an event loop. No behavior change yet — _trigger_loop
still uses its existing multi-fire state machine; the next commit
replaces it.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Single-fire-per-window in `_trigger_loop`

**Files:**
- Modify: `reliquary/miner/engine.py` lines 407-408 (state init) and lines 544-693 (`_trigger_loop`)
- Modify: `tests/unit/test_r_open_only_burst.py` (add integration test)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_r_open_only_burst.py`:

```python
import asyncio
import pytest

from reliquary.miner import engine as engine_module


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
        self._cached_cooldown: set[int] = set()
        self._local_n = 1
        self._local_hash = "hash"
        self.hf_model = None
        self.fire_calls: list[int] = []

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

    monkeypatch.setattr(
        engine_module, "get_window_state_v2_with_resp", fake_get_state,
        raising=False,  # imported inside function, monkeypatch at module
    )
    # Stub the in-function imports the loop performs
    monkeypatch.setattr(
        "reliquary.miner.submitter.get_window_state_v2_with_resp",
        fake_get_state,
    )
    monkeypatch.setattr(
        "reliquary.miner.submitter.SubmissionError", Exception,
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/reliquary-miner-priv && pytest tests/unit/test_r_open_only_burst.py::test_trigger_loop_fires_once_per_window -v
```

Expected: FAIL — the current `_trigger_loop` either fires multiple times (the existing multi-fire logic) or `_fired_windows` is missing from `_StubMiningEngine` defaults and the test asserts on what the new code produces. The exact failure reason is the assertion `me.fire_calls == [11]` reporting either `[]` or `[11, 11, 11, ...]`.

- [ ] **Step 3: Replace state-init lines**

In `mine_window` (around lines 407-408), replace:

```python
        # Window-local POST counter — caps at MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW.
        # Tracked locally because the validator's rate-limit gate doesn't tell
        # us how many slots we have left; we just mirror its cap.
        # Reset by the trigger loop on every new window.
        self._fires_per_window: int = 0
        self._last_fire_ts: float = 0.0
```

with:

```python
        # Set of window_n already fired. Single-shot per window under the
        # R_open-only burst policy (specs/2026-05-16-r-open-only-burst-design.md).
        # Pruned in _trigger_loop to bound growth.
        self._fired_windows: set[int] = set()
```

- [ ] **Step 4: Replace the fire decision block in `_trigger_loop`**

In `_trigger_loop` (around lines 649-691), replace the block from the comment `# Fire path: retry up to MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW` down to the closing `asyncio.create_task(...)` with:

```python
            # Fire path: one burst per window at the OPEN flip, up to 8 entries.
            # No retries within a window — entries baked after the flip wait
            # for the next window's R_open (specs/2026-05-16-r-open-only-burst-design.md).
            async with self._pool_lock:
                pool_size = len(self._pool)

            if _should_fire_for_window(state, self._fired_windows, pool_size):
                # Mark BEFORE scheduling so the next 5 ms tick can't double-fire
                # while the task is in flight.
                self._fired_windows.add(state.window_n)
                asyncio.create_task(
                    self._fire_for_window(state, url, client, results),
                    name=f"fire_window_{state.window_n}",
                )
            # Prune old entries to bound memory growth — 64 windows back is
            # well beyond any realistic /state rollback.
            self._fired_windows = {
                w for w in self._fired_windows if w >= state.window_n - 64
            }
```

Also remove the leftover `if state.window_n != last_window_n:` reset block (just above), and the `_MIN_FIRE_INTERVAL_S` env-var read, and any `last_window_n = -1` initialization at the top of the function.

- [ ] **Step 5: Run test to verify it passes**

```bash
cd ~/reliquary-miner-priv && pytest tests/unit/test_r_open_only_burst.py -v
```

Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
cd ~/reliquary-miner-priv && git add reliquary/miner/engine.py tests/unit/test_r_open_only_burst.py && git commit -m "feat(miner): single-fire-per-window trigger loop

Replaces the multi-fire state machine (_fires_per_window /
_last_fire_ts / _MIN_FIRE_INTERVAL_S) with a _fired_windows set.
Each window_n is fired exactly once at the OPEN flip; subsequent
ticks within the same window are no-ops. Entries baked mid-window
wait for the next window's flip.

Implements the policy half of
docs/superpowers/specs/2026-05-16-r-open-only-burst-design.md.
The boundary-safety sleep in _fire_for_window is removed in the
next commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Drop boundary-safety sleep from `_fire_for_window`

**Files:**
- Modify: `reliquary/miner/engine.py` lines 695-752 (signature, body, drop sleep)
- Modify: `tests/unit/test_r_open_only_burst.py` (add timing test)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_r_open_only_burst.py`:

```python
import inspect


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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/reliquary-miner-priv && pytest tests/unit/test_r_open_only_burst.py::test_no_boundary_safety_in_submit_entry tests/unit/test_r_open_only_burst.py::test_fire_for_window_empty_pool_returns_no_post -v
```

Expected:
- `test_no_boundary_safety_in_submit_entry` → FAIL (the BOUNDARY_SAFETY block is still in `_submit_entry`).
- `test_fire_for_window_empty_pool_returns_no_post` → may PASS already since the existing `if not fire: return` block at line 734 handles this; this test pins the behavior across the upcoming `max_fires` removal so a future refactor doesn't break it. If it passes at this point, that's fine — proceed to Step 3.

- [ ] **Step 3: Drop the boundary-safety block + simplify signature**

In `_fire_for_window` (line 695), change the signature from:

```python
    async def _fire_for_window(self, state, url, client, results, max_fires: int = 8):
```

to:

```python
    async def _fire_for_window(self, state, url, client, results):
```

Inside the body, delete:

1. Lines 705-707 of the docstring (the `Compensates self._fires_per_window ...` paragraph).
2. The drain block uses `max_fires` — replace `if len(fire) < max_fires:` with the module constant from `reliquary.constants`:

   ```python
   from reliquary.constants import MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW
   # ... inside the async with:
       if len(fire) < MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW:
           fire.append(entry)
   ```

3. The `actually_fired` / `_fires_per_window -=` refund block (lines 727-732).
4. The boundary-safety sleep block in `_submit_entry` (the block I read at lines ~792-815 starting with `_BOUNDARY_SAFETY_S = float(_os.environ.get("RELIQUARY_DRAND_BOUNDARY_SAFETY_S"...` and ending with the `await asyncio.sleep(...)`). Delete the `from reliquary.infrastructure.chain import seconds_until_next_drand_boundary` line that imported the helper for it — leave the `from reliquary.infrastructure.drand import get_current_chain as _gc` import alone if it's used elsewhere in the function; if not, also remove.

5. The log line at ~lines 741-746 referenced `self._fires_per_window` — replace with:

   ```python
   logger.info(
       "fire_for_window=%d: finalizing %d entries (pool kept=%d) randomness=%s",
       state.window_n, actually_fired, len(kept), randomness[:16],
   )
   ```

   (Drop the `fires_so_far=%d` placeholder and the `_fires_per_window` argument; `actually_fired` and `kept` are already in scope.)

- [ ] **Step 4: Run the new tests**

```bash
cd ~/reliquary-miner-priv && pytest tests/unit/test_r_open_only_burst.py::test_no_boundary_safety_in_submit_entry tests/unit/test_r_open_only_burst.py::test_fire_for_window_empty_pool_returns_no_post -v
```

Expected: both PASS.

- [ ] **Step 5: Run all R_open tests + smoke the engine import**

```bash
cd ~/reliquary-miner-priv && pytest tests/unit/test_r_open_only_burst.py -v && python -c "from reliquary.miner.engine import MiningEngine; print('import ok')"
```

Expected: all tests pass, `import ok` printed.

- [ ] **Step 6: Commit**

```bash
cd ~/reliquary-miner-priv && git add reliquary/miner/engine.py tests/unit/test_r_open_only_burst.py && git commit -m "feat(miner): drop boundary-safety sleep + max_fires from _fire_for_window

The validator now enforces drand_round zero-tolerance in BOTH
directions (commit 2d8ac38 on Catalyst). The old strategy of
sleeping past the next drand boundary to attach R_open+1 cost
chronological priority without buying any acceptance margin.
Under the strict-equality check, R_open+1 is still accepted but
sits behind every competitor that landed in R_open.

Removed:
  * RELIQUARY_DRAND_BOUNDARY_SAFETY_S env var (no-op now; documented)
  * seconds_until_next_drand_boundary import
  * max_fires parameter (always 8 from MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW)
  * _fires_per_window refund accounting (the field is gone)

Implements the no-sleep half of
docs/superpowers/specs/2026-05-16-r-open-only-burst-design.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Pool-empty-at-flip log (once per window)

**Files:**
- Modify: `reliquary/miner/engine.py` (`_trigger_loop`)
- Modify: `tests/unit/test_r_open_only_burst.py` (add log test)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_r_open_only_burst.py`:

```python
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
    monkeypatch.setattr(
        "reliquary.miner.submitter.SubmissionError", Exception,
    )

    async def fake_pull(**kwargs):
        return kwargs["local_n"], kwargs["local_hash"], kwargs["local_model"]
    monkeypatch.setattr(engine_module, "maybe_pull_checkpoint", fake_pull)

    with caplog.at_level(logging.INFO, logger="reliquary.miner.engine"):
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/reliquary-miner-priv && pytest tests/unit/test_r_open_only_burst.py::test_pool_empty_at_flip_logged_once_per_window -v
```

Expected: FAIL — `len(empty_logs) == 0`, no such log line exists yet.

- [ ] **Step 3: Add the log, gated by a separate set**

In `MiningEngine.mine_window` near the `_fired_windows` init (which Task 2 added), also add:

```python
        # Windows where we already logged "pool empty at OPEN" so the
        # 200 Hz tick doesn't spam the log. Same pruning as _fired_windows.
        self._logged_empty_windows: set[int] = set()
```

In `_trigger_loop`, after the `_should_fire_for_window` check and the prune, add:

```python
            elif (
                state.window_n not in self._fired_windows
                and state.window_n not in self._logged_empty_windows
                and state.state == WindowState.OPEN
                and state.randomness
                and pool_size == 0
            ):
                logger.warning(
                    "pool empty at OPEN window=%d — skipping fire, entries "
                    "baked later in this window will wait for the next flip",
                    state.window_n,
                )
                self._logged_empty_windows.add(state.window_n)
            self._logged_empty_windows = {
                w for w in self._logged_empty_windows if w >= state.window_n - 64
            }
```

(The `WindowState` is imported at the top of `_trigger_loop` already; if not, add it next to the existing protocol imports.)

- [ ] **Step 4: Run test to verify it passes**

```bash
cd ~/reliquary-miner-priv && pytest tests/unit/test_r_open_only_burst.py::test_pool_empty_at_flip_logged_once_per_window -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd ~/reliquary-miner-priv && git add reliquary/miner/engine.py tests/unit/test_r_open_only_burst.py && git commit -m "feat(miner): log pool-empty-at-OPEN once per window

Operator-visible signal that the miner saw the OPEN flip but had
no bakeable entries to burst — i.e. the window is being skipped
by design under the R_open-only policy. Gated by a per-window
set so the 200 Hz tick doesn't spam the log.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Verify env-var no-ops are documented + regression-test removal

**Files:**
- Modify: `reliquary/miner/engine.py` (drop unused env-var reads)
- No new tests; this is cleanup.

- [ ] **Step 1: Grep for any remaining references**

```bash
cd ~/reliquary-miner-priv && grep -n "RELIQUARY_MIN_FIRE_INTERVAL_S\|RELIQUARY_DRAND_BOUNDARY_SAFETY_S\|_fires_per_window\|_last_fire_ts\|max_fires" reliquary/miner/engine.py
```

Expected: no matches. If any remain, they are dead code; delete them.

- [ ] **Step 2: Run the full miner-priv test suite to catch any test that referenced the removed names**

```bash
cd ~/reliquary-miner-priv && pytest tests/unit/ -x --ignore=tests/unit/test_archive_window_content.py 2>&1 | tail -30
```

`test_archive_window_content.py` is a pre-existing failure (PermissionError, unrelated to this work) — ignore it.

Expected: all tests pass OR the failures point to a test that pinned the old multi-fire/boundary-safety contracts. For each such failure, decide:
- The test pinned old behavior that the spec explicitly removed → delete the test.
- The test was incidentally broken by a rename → update the test.

Document any deletions in the commit message.

- [ ] **Step 3: Smoke the engine module import + a dry-run of MiningEngine instantiation**

```bash
cd ~/reliquary-miner-priv && python -c "
from reliquary.miner.engine import MiningEngine, _should_fire_for_window
from reliquary.protocol.submission import WindowState
state = type('S', (), {'window_n': 1, 'state': WindowState.OPEN, 'randomness': 'x'})()
assert _should_fire_for_window(state, set(), 1) is True
assert _should_fire_for_window(state, {1}, 1) is False
print('smoke ok')
"
```

Expected: `smoke ok`.

- [ ] **Step 4: Commit (only if changes were needed)**

If Step 2 surfaced test deletions/updates:

```bash
cd ~/reliquary-miner-priv && git add tests/ && git commit -m "test(miner): drop tests pinning the multi-fire / boundary-safety contracts

These pinned behavior that's gone under the R_open-only policy
(specs/2026-05-16-r-open-only-burst-design.md). Removing rather
than updating because the underlying contract no longer exists,
not because the test was wrong.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

If Step 1 surfaced no remaining references and Step 2 was clean, no commit needed for this task.

---

### Task 6: Final integration smoke

**Files:** none modified; this is verification.

- [ ] **Step 1: Run the full new test file**

```bash
cd ~/reliquary-miner-priv && pytest tests/unit/test_r_open_only_burst.py -v
```

Expected: 10 passed (5 gate tests + 1 sub-second anchor + 1 single-fire + 1 no-boundary-safety structural + 1 fire-empty-pool + 1 pool-empty-log).

- [ ] **Step 2: Run any test files that touch engine.py**

```bash
cd ~/reliquary-miner-priv && pytest tests/unit/test_miner_engine_v2.py tests/miner_priv/ -v 2>&1 | tail -20
```

Expected: all pass.

- [ ] **Step 3: Inspect the diff one more time**

```bash
cd ~/reliquary-miner-priv && git log --oneline -8 && git diff HEAD~5..HEAD -- reliquary/miner/engine.py | wc -l
```

Expected: 4–5 commits since the prep WIP commit; engine.py diff is ~150-200 lines.

- [ ] **Step 4: Stop here for review**

Hand back to the user. Do NOT deploy to the prod miner box (86.38.238.199) without a manual restart procedure — the `project_miner_setup` memory has the orphan-VLLM-VRAM gotcha, and the user owns that operation.

## Self-review notes

- Spec section "Removed" lists `RELIQUARY_MIN_FIRE_INTERVAL_S` and `RELIQUARY_DRAND_BOUNDARY_SAFETY_S` — both verified in Task 5 Step 1.
- Spec section "Edge cases / Pool empty at flip" → Task 4 covers the log line; the design rule "entries baked later in this window will wait for the next flip" is implicit in the single-fire-per-window invariant proven by Task 2.
- Spec section "Edge cases / Finalize fails on a single entry" → already covered by existing `asyncio.gather(..., return_exceptions=True)` at line 749; no new test needed.
- Spec section "Edge cases / EMA not warm at startup" → unaffected by this change set; covered by the pre-req WIP `test_clock_offset.py`.
- Spec section "Out of scope" — no tasks for predictive pre-arm or adaptive latency budget, as required.
