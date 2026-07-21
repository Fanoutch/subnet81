# R_open-only burst — miner submit policy

**Date:** 2026-05-16
**Branch target:** `main` of `~/reliquary-miner-priv`
**Status:** design approved, ready for implementation plan

## Goal

Every POST the miner emits should carry `drand_round == R_open`, where
`R_open` is the drand round in progress at the validator's window OPEN
flip. When that's not achievable in a given window, hold the pool entries
for the next window rather than firing them at `R_open + k` and losing
chronological priority.

## Why now

The validator tightened `DRAND_ROUND_BACKWARD_TOLERANCE` to `0`
(commit `2d8ac38`). Both directions are now strict equality:

- `drand_round > current` → `FUTURE_ROUND`
- `drand_round < current` → `STALE_ROUND`
- `drand_round == current` → accepted, enters that round's chronological
  bucket. Seal-time selection sorts buckets ascending and the earliest
  bucket gets the slots first (`reliquary/validator/batch_selection.py:148`).

Two pieces of miner logic that made sense under the old wider tolerance
are now actively harmful:

1. The boundary-safety sleep in `_fire_for_window` (`engine.py:802-815`)
   waits past the next drand boundary and deliberately attaches
   `R_open + 1`. Under tolerance=0, `R_open + 1` is still accepted but
   loses the chronological priority — we sit behind every competitor
   that landed in `R_open`.

2. `_compute_offset_sub_second` anchors the clock-offset estimate at
   `t_round_start + period/4` (`engine.py:248`), biasing the corrected
   clock toward `R - 1`. The commit (`8ab8aad`) explicitly justified the
   bias as "STALE_ROUND, which the validator's 1-round backward tolerance
   accepts". That backward tolerance is now zero, so the bias produces
   straight rejections.

The retry-within-window code path (`5522070`) also conflicts: by
construction every retry within the same window lands in `R_open + k`,
which we now treat as a loss-of-priority and refuse.

## Architecture

One rule:

> On each `_trigger_loop` tick (200 Hz), if `state.window_n` has not yet
> been fired AND `state.state == OPEN` AND `state.randomness` is set
> AND `pool_size > 0`, fire **one** burst of up to 8 entries in
> parallel. Mark the window as fired. All subsequent ticks for the same
> `window_n` are no-ops.

That's the whole policy. No boundary check, no fire-gate budget, no
retry counter.

## Components

### Unchanged

- `_generator_loop` (bg) — continues baking entries into `self._pool`.
- `_apply_offset_from_validator_response` — keeps updating
  `_DRAND_CLOCK_OFFSET_S` via the validator HTTP Date header EMA on
  every /state poll.
- 200 Hz `/state` polling interval.
- `_refresh_drand_offset_loop` — drand-network fallback every 60 s
  (with the bias removed; see below).
- `_current_drand_round_at_send` — still computed at the POST instant
  using the corrected clock.

### Modified

**`_trigger_loop` (`engine.py:544`)**

Replace the multi-fire state machine
(`_fires_per_window`, `_last_fire_ts`, `_MIN_FIRE_INTERVAL_S`,
`posts_left` accounting) with a single set:

```python
self._fired_windows: set[int] = set()
```

The fire condition collapses to:

```python
if (
    state.window_n not in self._fired_windows
    and state.state == WindowState.OPEN
    and state.randomness
    and pool_size > 0
):
    self._fired_windows.add(state.window_n)
    asyncio.create_task(
        self._fire_for_window(state, url, client, results),
        name=f"fire_window_{state.window_n}",
    )
```

We add to `_fired_windows` *before* scheduling the task, so the next
5 ms tick can't double-fire while the task is in flight. The set grows
unbounded over time; prune entries older than `state.window_n - 64`
once per tick (one liner) to keep it bounded — 64 is well beyond any
realistic late-state-rollback the validator could send.

**`_fire_for_window` (`engine.py:695`)**

- Drop the `max_fires` parameter — always `min(8, pool_size)`.
- Delete the boundary-safety sleep block (lines 802-815) and its
  surrounding imports of `seconds_until_next_drand_boundary` /
  `get_current_chain`.
- Keep everything else: pool snapshot, finalize on threads, drand
  round computation just before POST, parallel `asyncio.gather` POST.

**`_compute_offset_sub_second` (`engine.py:229`)**

Change one line:

```python
t_anchor = t_round_start + period / 2   # was: period / 4
```

The anchor at midpoint is unbiased — residual estimation error is
symmetric around zero. With the validator now strict in both
directions, asymmetric bias is pure rejection probability.

Update the docstring accordingly (drop the "biases toward STALE_ROUND
which the validator tolerates" rationale; reference this design doc
or commit `2d8ac38`).

### Removed

- Env vars: `RELIQUARY_MIN_FIRE_INTERVAL_S`,
  `RELIQUARY_DRAND_BOUNDARY_SAFETY_S`.
- State fields: `_fires_per_window`, `_last_fire_ts`.
- The `max_fires` parameter on `_fire_for_window`.
- The boundary-safety sleep block.
- The `from reliquary.infrastructure.chain import
  seconds_until_next_drand_boundary` import (now unused).

Document removals in the commit message — operators with these env
vars set in their launch scripts need to know they're now no-ops.

## Data flow

```
_generator_loop ──► self._pool (append baked entries)
                        │
                        ▼
/state poll (200 Hz) ── _apply_offset_from_validator_response (EMA)
                        │
                        ▼
                   gate: window_n not yet fired
                         AND state == OPEN
                         AND randomness set
                         AND pool non-empty
                        │
                        ▼
                   _fire_for_window:
                     snapshot pool (≤8)
                     finalize on threads (≈50 ms)
                     drand_round = corrected clock NOW
                     asyncio.gather POST in parallel
                     remove fired entries from pool
```

## Edge cases

- **Pool empty at flip.** No fire. Log
  `pool empty at OPEN window=N` once per window. Entries baked
  later in this window wait for `window_n + 1`'s flip.
- **First window after launch.** Generator hasn't yielded an entry
  yet; pool is empty; first window is skipped. Expected.
- **Pool dropped on checkpoint advance** (default optimistic-keep
  unless `RELIQUARY_DROP_POOL_ON_CKPT=1` is set). Same as cold start
  if drop happens close to flip.
- **Finalize fails on a single entry.** That entry is dropped; the
  remaining entries POST in the same burst.
- **POST returns `STALE_ROUND` / `FUTURE_ROUND`.** The flip detection
  landed too close to a drand boundary for the finalize+POST latency
  to clear it. Log the rejection. The entry is *not* re-queued — one
  shot per entry per window.
- **EMA not warm at startup.** At 200 Hz with α = 0.2 the corrected
  clock converges in roughly 50 ms wall time. The first 10 ticks of
  the first launch may use a coarsely-calibrated clock; if the
  validator HTTP Date header is reachable on poll #1 the EMA seeds
  from that single sample.
- **State rollback (`state.window_n` regresses).** Should not happen
  in practice, but if it does the `_fired_windows` set will still
  contain the older N and the fire is skipped. Safe.

## Testing

- **Unit `_trigger_loop`:** mock /state to flip `state.state` between
  `READY` and `OPEN` across two `window_n` values; assert exactly one
  `_fire_for_window` task is created per window even after many
  ticks; assert no fire when `pool_size == 0`.
- **Unit `_fire_for_window` empty pool:** call with empty pool, expect
  no POST, no exception, results list unchanged.
- **Unit `_compute_offset_sub_second` neutral anchor:** sweep `t_fetch`
  across one drand period at constant `r_drand`; assert mean offset
  error is zero (within float tolerance) and the error distribution is
  symmetric around zero (max overshoot == max undershoot in magnitude).
- **Regression deletion:** the existing tests pinning
  `RELIQUARY_DRAND_BOUNDARY_SAFETY_S` behavior and the
  retry-within-window MAX_SUBMISSIONS pacing must be removed or
  inverted — their old contracts are gone.
- **Integration (testnet):** start miner, observe `submitted ... drand_round=R`
  log lines across 10 windows; cross-reference each `R` against the
  validator's `accepted ... drand_round=R` (commit `e9a57bf` surfaces
  this in validator logs). Target: > 95 % of bursts where every entry
  in the burst carries the same `R == R_open`.

## Operator impact

- Two env vars become no-ops. Document in the commit message.
- Behavior change: under load where the pool refills mid-window, the
  miner used to send a second/third sub-burst. It no longer does. This
  is intentional under the strict-equality validator; later sub-bursts
  could not have landed in `R_open` anyway.
- Cold-start: the first window after restart is essentially guaranteed
  to skip (pool empty). Operators restarting mid-mining lose one
  window. Acceptable.

## Out of scope (explicit non-goals)

- Predictive pre-arm (firing on a timer at predicted T_open before
  /state confirms OPEN) — deferred. The strict-equality + 200 Hz poll
  + accurate EMA should already hit R_open in the vast majority of
  windows. If empirical data shows a high miss rate due to /state lag,
  pre-arm becomes the next iteration.
- Adaptive latency budget — not needed because we removed the budget
  itself.
- Inference-speed optimization, reward prediction — separate workstreams,
  unaffected by this change.
