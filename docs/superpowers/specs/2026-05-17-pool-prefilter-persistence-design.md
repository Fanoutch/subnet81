# Pool pre-filter + persistence — miner design

**Date:** 2026-05-17
**Branch target:** `main` of `~/reliquary-miner-priv`
**Status:** design approved, ready for implementation plan

## Goal

Stop wasting GPU on bakes that the validator will reject with
`out_of_zone`, and stop losing pre-baked entries across miner restarts.
Specifically:

1. **Pre-filter `out_of_zone`** — discard entries whose reward
   standard deviation is below `SIGMA_MIN` (=0.43, or
   `BOOTSTRAP_SIGMA_MIN` =0.33 in bootstrap) *before* they enter the
   pool. These entries are guaranteed-reject by the validator and
   currently consume a slot in the per-window fire burst for nothing.

2. **Raise pool cap** from 16 to 200 entries. With the pre-filter
   in place, more entries actually have a shot at acceptance, and the
   bg generator should not stop baking just because 16 are queued.

3. **Persist the pool across restart**. Entries are expensive to
   bake (~30-60 s GPU each) and currently disappear on every kill.
   A cold-start now costs ~1-2 forfeit windows; persistence cuts
   this to zero except for the very first launch of the box.

## Why now

The validator's `out_of_zone` check (`reliquary.validator.verifier.is_in_zone`)
is the single most common rejection reason for our hotkey
`5C8V7S8VL4Pe`. From the 2026-05-16 prod run, ~80 % of our submissions
that made it past the drand check were rejected on `out_of_zone` —
all of them with reward distributions like `[1,1,1,1,1,1,0,1]` whose
population standard deviation is ~0.33, just below the 0.43 cutoff.

The check is **trivial** to replicate miner-side: it's the population
std of the 8-entry reward vector. The miner already has the rewards
in hand at the end of `_pre_bake_entry` (line 1086 of `engine.py`).
Adding a one-line gate there saves a full bake cycle for every
out-of-zone group.

The 16-entry pool cap (`RELIQUARY_POOL_MAX_SIZE`) was sized for the
old V30 exploit miner that recycled cached rollouts; under v2.3
pipelined honest mining, every fire consumes up to 8 fresh entries
and the bg generator needs more headroom to avoid stalling between
fires.

The lack of persistence means each restart loses the entire pool
plus 30-90 s of vLLM warmup, and historically each restart has cost
1-3 windows of forfeit while the first bake completes. The miner has
been restarted 5+ times in the last 24 hours during the drand /
envelope-signature fix cycle; persistence would have saved the
forfeits.

## Architecture

Three independent changes, all in `reliquary/miner/`. Each can be
verified in isolation.

### Change A — out_of_zone pre-filter in `_pre_bake_entry`

The 8 rewards are computed at `engine.py:1086` inside the per-rollout
loop and stored in `rollouts_cache[i]["reward"]`. At the end of
`_pre_bake_entry`, just before the `return` of the entry dict, gather
the reward vector and compute the population std:

```python
rewards = [r["reward"] for r in rollouts_cache]
from reliquary.validator.verifier import rewards_std, is_in_zone
sigma = rewards_std(rewards)
if not is_in_zone(sigma, bootstrap=False):
    logger.info(
        "pre_bake[out_of_zone] skipping prompt=%d sigma=%.3f rewards=%s",
        prompt_idx, sigma, rewards,
    )
    return None
```

Returning `None` is already a valid outcome for `_pre_bake_entry`
(it's what generation-underflow does at line 1075) — the caller
(`_generator_loop`) treats `None` as "skip pool append, pick next
prompt" automatically. No changes needed at the call site.

The `bootstrap` parameter mirrors the validator's `self.bootstrap`
flag (`batcher.py:476`). Bootstrap windows allow σ ≥ 0.33 instead of
0.43. The miner does NOT see this flag via `/state` (the
`GrpoBatchState` schema doesn't expose it). We hardcode
`bootstrap=False` (strict, 0.43 threshold). The cost: during a real
bootstrap phase (first BOOTSTRAP_WINDOWS=N of a new subnet), the
miner is slightly more conservative than the validator and may
discard entries the validator would have accepted. This is
acceptable — bootstrap phases are rare and the miner re-bakes
quickly.

**Why import from `reliquary.validator.verifier`** — the helpers
(`rewards_std`, `is_in_zone`) are pure Python, no torch / no network
dependency, and live behind a `from reliquary.constants import ...`
that's already in the miner. Importing keeps the threshold in
exactly one place; if the validator bumps `SIGMA_MIN`, the miner
follows.

### Change B — pool cap 16 → 200

In `mine_window` (around `engine.py:427`):

```python
self._pool_max_size = int(_os.environ.get("RELIQUARY_POOL_MAX_SIZE", "200"))
```

Just the default literal bumps from `"16"` to `"200"`. Operators who
hit a memory ceiling can lower it via env var. All existing logic in
`_generator_loop` that checks `pool_full = len(self._pool) >= self._pool_max_size`
(line 511) keeps working unchanged.

Memory budget at 200 entries: each rollout's `hidden_states_cpu` is a
`torch.Tensor` of shape `[seq_len, hidden_dim]` in float32 — typical
`seq_len ≈ 200-1000` tokens × `hidden_dim = 4096` × 4 bytes = 3-16 MB
per rollout. 8 rollouts × 16 MB = ~130 MB max per entry. 200 entries
× 130 MB ≈ 26 GB CPU RAM. The H200 box has 754 GB system RAM
(`free -g` confirmed during the disk-full diagnostic on 2026-05-16),
so 26 GB is well within budget.

### Change C — pool persistence to disk

New module: `reliquary/miner/pool_persistence.py`. Three functions,
roughly 80 LOC total. Pure synchronous code, called from inside the
existing pool lock so there's no concurrency surface added.

```python
def save_entry(entry: dict, pool_dir: Path) -> Path:
    """Atomically persist a baked entry. Returns the final path.

    Layout: pool_dir/<prompt_idx>_<timestamp_ns>.pt
    Atomicity: torch.save to .tmp then os.rename — survives crash
    mid-write because rename is atomic on POSIX.
    """

def delete_entry(path: Path) -> None:
    """Remove a persisted entry, idempotent — missing file is silently
    skipped (race with a concurrent restart's reload is benign)."""

def load_pool(pool_dir: Path, local_checkpoint_n: int) -> list[dict]:
    """Scan pool_dir and torch.load every .pt. Skip files that fail
    to deserialize (logged WARNING, file left in place for forensic
    inspection). Entries whose baked checkpoint_n != local_checkpoint_n
    are KEPT (optimistic policy, same as the live RELIQUARY_DROP_POOL_ON_CKPT=0
    default) but tagged with a WARNING line so operators see the
    optimistic-keep count after restart. Returns the list of entries
    sorted by file mtime so older bakes drain first."""
```

Integration in `engine.py`:

- **Init** in `mine_window` (right after `self._pool = []` at line ~388
  but before the bg generator starts):

  ```python
  pool_dir = Path(_os.environ.get(
      "RELIQUARY_POOL_DIR", "/root/reliquary-state/pool",
  ))
  pool_dir.mkdir(parents=True, exist_ok=True)
  self._pool_dir = pool_dir
  reloaded = load_pool(pool_dir, self._local_n)
  if reloaded:
      async with self._pool_lock:
          self._pool.extend(reloaded)
      logger.info("pool: reloaded %d entries from %s", len(reloaded), pool_dir)
  ```

  Note: `self._local_n = -1` at this point (sentinel), so the
  staleness check inside `load_pool` doesn't kick in until later
  when the first `/state` poll comes back with `state.checkpoint_n`.
  That's fine — optimistic-keep means the entries are loaded
  regardless; the staleness tag is informational.

- **After bake success** in `_generator_loop` (after the
  `self._pool.append(new_entry)` at engine.py:560):

  ```python
  save_entry(new_entry, self._pool_dir)
  new_entry["_persist_path"] = ...  # set inside save_entry, returned
  ```

  Track the on-disk path on the in-memory entry so the fire path
  knows which file to delete.

- **After fire success** in `_fire_for_window` (after `asyncio.gather`
  completes, around engine.py:756): for each fired entry, call
  `delete_entry(entry["_persist_path"])`. Idempotent — if the file
  was already deleted by a concurrent process (shouldn't happen
  with single-miner setup, but safe), the missing-file branch hits.

- **Pool drop on checkpoint advance** (existing path at engine.py:622
  when `RELIQUARY_DROP_POOL_ON_CKPT=1`): after `self._pool = []`,
  also clear the on-disk directory (`shutil.rmtree(pool_dir);
  pool_dir.mkdir()`). The default keeps the optimistic-keep behavior
  including for on-disk entries.

## File map

- Modify: `reliquary/miner/engine.py`
  - Line ~427: env-var default `16` → `200`
  - Line ~1086 area (inside `_pre_bake_entry`): add std + is_in_zone gate, return None on fail
  - Line ~388-395 (inside `mine_window`): pool_dir setup + reload
  - Line ~560 (inside `_generator_loop`): call `save_entry` after pool append
  - Line ~756 (inside `_fire_for_window`): call `delete_entry` for each fired entry
  - Line ~622-630 (the `drop_on_ckpt` branch): clear on-disk directory too
- Create: `reliquary/miner/pool_persistence.py` — `save_entry`,
  `delete_entry`, `load_pool` (~80 LOC, no external deps beyond `torch`
  and `pathlib`)
- Create: `tests/unit/test_pool_persistence.py`
  - `save_entry` writes a .pt file at the expected path, atomic
  - `delete_entry` removes a known file, idempotent on missing
  - `load_pool` returns the entries in mtime order, skips corrupt
    files, logs warning on ckpt mismatch
  - `_pre_bake_entry` returns `None` when σ < 0.43, returns entry
    when σ ≥ 0.43 (mock the bake outputs to control rewards)

## Edge cases

- **Empty pool dir on first launch.** `mkdir(parents=True, exist_ok=True)`
  creates it; `load_pool` returns `[]`; bg generator starts from cold,
  same as today.

- **Corrupt .pt file.** `torch.load` raises. `load_pool` catches,
  logs `WARNING` with filename, continues. The file is left on disk
  so the operator can `rm` it manually after inspection.

- **Checkpoint advance during reload.** Reload happens early in
  `mine_window`, before `_trigger_loop` polls `/state` for the first
  time. So `local_n = -1` at reload — every entry's `checkpoint_n`
  differs. The staleness log emits but every entry is kept (optimistic).
  When the first `/state` poll updates `local_n` to the real value
  (e.g. 19), the in-memory pool stays as-is — the live
  `maybe_pull_checkpoint` path handles any subsequent advance the
  same way it does today.

- **Disk full during `save_entry`.** `torch.save` raises `OSError`.
  Wrap in try/except in the generator loop — log `ERROR`, skip
  the save, keep the entry in memory only. Next bake retries
  the save (clean dir might have freed up via fire-delete by then).
  Avoid crashing the generator on disk-full.

- **Restart-induced reload duplicates an entry that was fired but
  whose delete didn't make it to disk.** The validator would
  hash-dedupe via `RolloutHashSet`. We'd get `hash_duplicate` on the
  re-fire. Acceptable cost. To prevent it cleanly, `delete_entry`
  must complete BEFORE the fire's `submit_batch_v2` returns
  successfully. The current draining-then-POST order in
  `_fire_for_window` already establishes this: pool drain happens
  before POST, and we'll add the `delete_entry` between drain and
  POST (or in the `done_callback` of each `_submit_entry`, but
  simpler to do it inline after gather).

- **`/state` not yet available at reload time.** `local_n = -1` sentinel.
  Reload runs without staleness pruning. Live miner's first `/state`
  poll syncs `_local_n` and the live checkpoint-advance path takes
  over. No new code needed.

## Testing

- **Unit `test_pool_persistence.py`** — listed above. ~10 cases.

- **Unit `test_pre_bake_out_of_zone.py`** — pin the σ < 0.43 → None
  contract. Mock `_generate_m_rollouts` to return 8 fixed rollouts;
  override their `reward` after the env compute step (or mock
  `self.env.compute_reward` to return controlled values).
  Cases: all-1.0 (σ=0 → drop), [1,1,1,1,1,1,0,1] (σ=0.33 → drop),
  [1,0,1,0,1,0,1,0] (σ=0.5 → keep), σ exactly at 0.43 boundary
  (keep — strict-equal accepts).

- **Integration smoke** — run miner under TestClient with a stubbed
  env that produces controllable reward distributions; assert the
  pool gets only in-zone entries.

- **Regression** — the existing `test_r_open_only_burst.py` tests
  pass without changes. The new `_pool_dir` field on `_StubMiningEngine`
  is added to keep them green.

## Operator impact

- New env var: `RELIQUARY_POOL_DIR` (default `/root/reliquary-state/pool`).
  Operator can point this elsewhere (different disk) or change it to
  trigger a clean start.
- Env var: `RELIQUARY_POOL_MAX_SIZE` default changes `16` → `200`.
  Operators who explicitly set the lower value retain it. Document
  the change in the commit message.
- Disk usage: up to ~26 GB for a full pool (200 entries × ~130 MB),
  in `RELIQUARY_POOL_DIR`. Operator should ensure free space;
  existing HF cache cleanup is independent.

## Out of scope (deliberate non-goals)

- **Other pre-filters** (`bad_termination`, `distribution_suspicious`).
  Operator explicitly limited scope to `out_of_zone`. Adding them
  later is a one-function change per filter.
- **Multi-process pool sharing.** Single miner instance per hotkey
  is the production setup; cross-process consistency is not needed.
- **Compression of persisted entries.** A 130 MB entry torch-saves
  to ~130 MB; gzip would shave ~40 % at the cost of CPU during save.
  Defer until disk usage actually bites.
- **Periodic stale-entry sweep.** The on-disk pool is naturally
  bounded by `RELIQUARY_POOL_MAX_SIZE` (the in-memory cap pushes
  back on the generator, so save_entry runs at most pool_max_size
  times before the next fire-delete). No separate GC needed.
- **Migration of an existing live pool.** First deploy with this
  spec starts from an empty `RELIQUARY_POOL_DIR`; the in-memory
  pool at restart time is lost as today. Acceptable one-time cost.
