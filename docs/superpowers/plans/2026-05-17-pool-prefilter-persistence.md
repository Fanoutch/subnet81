# Pool pre-filter + persistence implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the three changes from `docs/superpowers/specs/2026-05-17-pool-prefilter-persistence-design.md`: out_of_zone pre-filter in `_pre_bake_entry`, raise pool cap default 16 → 200, and persist the pool to disk across restarts.

**Architecture:** Surgical edits to `reliquary/miner/engine.py` plus one new module `reliquary/miner/pool_persistence.py`. The validator's `rewards_std` and `is_in_zone` are imported directly (single source of truth for σ thresholds). Persistence is one .pt file per entry, atomic via tmp+rename, loaded sorted by mtime at startup.

**Tech Stack:** Python 3.11, torch (for save/load of tensors), pytest, no new dependencies.

## Working-tree note

At plan creation the user had no unstaged edits. The previous commit is `36307a1` (the spec doc). All work starts from there.

## File map

- Modify: `~/reliquary-miner-priv/reliquary/miner/engine.py`
  - Line ~427: env-var default `"16"` → `"200"` (Task 2)
  - Line ~882 (top of `_pre_bake_entry`): add zone-gate helper call (Task 1)
  - Line ~1086 area (end of `_pre_bake_entry`, just before `return entry_dict`): compute σ, return None if out of zone (Task 1)
  - Line ~388-395 (`mine_window`, just after `self._pool = []`): pool_dir setup + reload (Task 4)
  - Line ~560 (`_generator_loop`, after `self._pool.append(new_entry)`): save_entry call (Task 4)
  - Line ~756 (`_fire_for_window`, after `asyncio.gather` returns): delete_entry per fired entry (Task 4)
  - Line ~622-630 (`drop_on_ckpt` branch): also wipe on-disk pool_dir (Task 4)
- Create: `~/reliquary-miner-priv/reliquary/miner/pool_persistence.py` — `save_entry`, `delete_entry`, `load_pool` (Task 3)
- Create: `~/reliquary-miner-priv/tests/unit/test_pre_bake_out_of_zone.py` — Task 1's tests
- Create: `~/reliquary-miner-priv/tests/unit/test_pool_persistence.py` — Task 3's tests
- Modify: `~/reliquary-miner-priv/tests/unit/test_r_open_only_burst.py` — Task 4 adds `_pool_dir` to the `_StubMiningEngine` so existing tests keep passing

---

### Task 1: out_of_zone pre-filter helper + integration

**Files:**
- Create: `tests/unit/test_pre_bake_out_of_zone.py`
- Modify: `reliquary/miner/engine.py` — add helper function + call from `_pre_bake_entry`

- [ ] **Step 1: Write failing tests for the gate helper**

Create `tests/unit/test_pre_bake_out_of_zone.py`:

```python
"""Tests for the out_of_zone pre-filter in _pre_bake_entry.

The miner drops a baked group whose reward std σ would be rejected by
the validator's `is_in_zone(σ, bootstrap=False)` check. This saves the
GPU cost of finalize and the per-window slot of firing an entry that
the validator guarantees to reject.

The decision lives in a small pure helper so the threshold logic is
trivially testable without mocking vLLM or HF.
"""

from reliquary.miner.engine import _skip_for_out_of_zone


def test_all_ones_zero_std_skips():
    """σ=0 (degenerate, all-1.0 rewards) — must skip."""
    assert _skip_for_out_of_zone([1.0] * 8) is True


def test_all_zeros_zero_std_skips():
    """σ=0 (degenerate, all-0.0 rewards) — must skip."""
    assert _skip_for_out_of_zone([0.0] * 8) is True


def test_7_of_8_correct_under_threshold_skips():
    """σ≈0.33 for [1,1,1,1,1,1,0,1] — below 0.43 cutoff → skip.
    
    This is the exact distribution that caused the bulk of our prod
    rejections (rewards=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0]
    rejected with reason=out_of_zone, prompt=695470 on 2026-05-16).
    """
    assert _skip_for_out_of_zone([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0]) is True


def test_4_of_8_correct_keeps():
    """σ≈0.5 for [1,0,1,0,1,0,1,0] — above 0.43 → keep."""
    assert _skip_for_out_of_zone([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]) is False


def test_5_of_8_correct_keeps():
    """σ≈0.484 for [1,1,1,1,1,0,0,0] — above 0.43 → keep."""
    assert _skip_for_out_of_zone([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]) is False


def test_empty_rewards_skips():
    """Degenerate empty list — rewards_std returns 0.0, must skip
    (refusing to mint an empty group is the safe default).
    """
    assert _skip_for_out_of_zone([]) is True


def test_threshold_uses_strict_zone():
    """The miner hardcodes bootstrap=False (strict, SIGMA_MIN=0.43).
    
    During a real bootstrap phase the miner is slightly more
    conservative than the validator — acceptable per the spec.
    """
    # σ for [1,1,0,1,1,1,1,1] = sqrt(0.875*0.125) ≈ 0.331 — below 0.43.
    # In bootstrap (0.33 cutoff) this would barely keep; in strict it skips.
    assert _skip_for_out_of_zone([1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]) is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/reliquary-miner-priv && pytest tests/unit/test_pre_bake_out_of_zone.py -v
```

Expected: `ImportError: cannot import name '_skip_for_out_of_zone' from 'reliquary.miner.engine'`

- [ ] **Step 3: Add the helper to engine.py**

In `reliquary/miner/engine.py`, find the line `def _should_fire_for_window(` (this is where the existing module-level pure helpers live — we put the new one next to it for discoverability). Insert this function **immediately before** `_should_fire_for_window`:

```python
def _skip_for_out_of_zone(rewards: list[float]) -> bool:
    """Return True iff the validator would reject this rollout group as
    ``out_of_zone`` (rewards_std < SIGMA_MIN = 0.43, strict, no bootstrap).

    Mirrors ``reliquary.validator.verifier.is_in_zone`` with
    ``bootstrap=False``. The miner does not see the validator's
    bootstrap flag via /state, so we hardcode the strict threshold.
    During a real bootstrap phase the miner is slightly more
    conservative than the validator — acceptable per the design doc.

    Called from ``_pre_bake_entry`` after rewards are computed; entries
    that would be rejected are dropped before they enter the pool,
    saving the per-window fire slot and the finalize GPU cost.
    """
    from reliquary.validator.verifier import is_in_zone, rewards_std
    sigma = rewards_std(rewards)
    return not is_in_zone(sigma, bootstrap=False)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd ~/reliquary-miner-priv && pytest tests/unit/test_pre_bake_out_of_zone.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Integrate into `_pre_bake_entry`**

In `reliquary/miner/engine.py`, find the end of `_pre_bake_entry`. The function currently ends with a `return` of an entry dict (around line 1113 — locate it by grepping for `return {` inside the function body).

Right **before** the final `return {`, add:

```python
        rewards_for_zone = [r["reward"] for r in rollouts_cache]
        if _skip_for_out_of_zone(rewards_for_zone):
            logger.info(
                "pre_bake[out_of_zone] skipping prompt=%d sigma=%.3f rewards=%s",
                prompt_idx,
                __import__("reliquary.validator.verifier", fromlist=["rewards_std"]).rewards_std(rewards_for_zone),
                rewards_for_zone,
            )
            return None
```

Note: the lazy `__import__` keeps the import path explicit at the call site without polluting the module top-level imports — same pattern as the existing inside-function imports in `_pre_bake_entry`. (If you prefer, hoist `from reliquary.validator.verifier import is_in_zone, rewards_std` to the top of `_pre_bake_entry`'s body — both work.)

The caller (`_generator_loop` around line 540) already handles a `None` return as "skip this prompt, pick a new one" — same as the existing generation-underflow path at line 1075. No call-site change needed.

- [ ] **Step 6: Smoke-check existing tests still pass**

```bash
cd ~/reliquary-miner-priv && pytest tests/unit/test_r_open_only_burst.py tests/unit/test_clock_offset.py tests/unit/test_pre_bake_out_of_zone.py -v
```

Expected: 14 + 7 = at least 21 passed (count depends on exact prior state).

- [ ] **Step 7: Commit**

```bash
cd ~/reliquary-miner-priv && git add tests/unit/test_pre_bake_out_of_zone.py reliquary/miner/engine.py && git commit -m "feat(miner): pre-filter out_of_zone bakes before they enter the pool

The validator rejects ~80% of our submissions with reason=out_of_zone
when the 8-rollout group's reward std is below SIGMA_MIN (=0.43). The
check is trivial — population std of 8 floats — and we have the
rewards in hand at the end of _pre_bake_entry. Compute σ there and
return None for groups that would be rejected, saving the GPU cost
of finalize and the per-window fire slot.

Implements section 'Change A' of
docs/superpowers/specs/2026-05-17-pool-prefilter-persistence-design.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Pool cap 16 → 200

**Files:**
- Modify: `reliquary/miner/engine.py` line ~427

- [ ] **Step 1: Find the current default**

```bash
cd ~/reliquary-miner-priv && grep -n "RELIQUARY_POOL_MAX_SIZE" reliquary/miner/engine.py
```

Expected: one match around line 427:
```python
self._pool_max_size = int(_os.environ.get("RELIQUARY_POOL_MAX_SIZE", "16"))
```

- [ ] **Step 2: Bump the default to 200**

In `reliquary/miner/engine.py`, change that line to:

```python
        # Default 200 (was 16). Sized for v2.3 honest mining where every
        # fire consumes 8 fresh entries; the bg generator needs more
        # headroom between fires. Memory budget ~26 GB CPU RAM at full
        # pool (200 × ~130 MB hidden_states per entry), within the 754 GB
        # on the H200 box. Disk budget same magnitude — see
        # RELIQUARY_POOL_DIR. Operators can lower via env var.
        self._pool_max_size = int(_os.environ.get("RELIQUARY_POOL_MAX_SIZE", "200"))
```

- [ ] **Step 3: Smoke-check all tests still pass (no test should care about this constant)**

```bash
cd ~/reliquary-miner-priv && pytest tests/unit/ -x --ignore=tests/unit/test_archive_window_content.py 2>&1 | tail -5
```

Expected: tests pass (or pre-existing failures unchanged). The bumped default is internal — no test pins it.

- [ ] **Step 4: Commit**

```bash
cd ~/reliquary-miner-priv && git add reliquary/miner/engine.py && git commit -m "feat(miner): RELIQUARY_POOL_MAX_SIZE default 16 → 200

Under v2.3 honest mining, every fire consumes up to 8 fresh entries
and the bg generator needs more headroom between fires. The 16-entry
cap was sized for the old V30 exploit recycling cached rollouts.

Memory budget at full 200-entry pool: ~26 GB CPU RAM
(200 × ~130 MB hidden_states per entry). The H200 box has 754 GB.
Disk budget same magnitude when pool persistence lands in the next
commit. Operators can lower via env var.

Implements section 'Change B' of
docs/superpowers/specs/2026-05-17-pool-prefilter-persistence-design.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: pool_persistence module

**Files:**
- Create: `reliquary/miner/pool_persistence.py`
- Create: `tests/unit/test_pool_persistence.py`

- [ ] **Step 1: Write failing tests for the module**

Create `tests/unit/test_pool_persistence.py`:

```python
"""Tests for reliquary/miner/pool_persistence.py — disk-backed entry store.

The pool persistence layer is a thin wrapper over torch.save / torch.load
plus atomic-rename for crash safety. Three functions:

  * save_entry(entry, pool_dir) → Path  (atomic .tmp + rename)
  * delete_entry(path) → None           (idempotent)
  * load_pool(pool_dir, local_checkpoint_n) → list[dict]

All synchronous, called from inside the existing pool lock — no new
concurrency.
"""

import logging
import time
from pathlib import Path

import pytest
import torch

from reliquary.miner.pool_persistence import (
    delete_entry, load_pool, save_entry,
)


def _make_entry(prompt_idx: int, checkpoint_n: int = 14) -> dict:
    """Build a minimal entry shaped like the real bake output."""
    return {
        "prompt_idx": prompt_idx,
        "problem": {"prompt": "test", "answer": "1"},
        "rollouts": [
            {
                "all_tokens": [1, 2, 3, 4],
                "prompt_length": 2,
                "completion_text": "test",
                "hidden_states_cpu": torch.zeros(2, 8),
                "token_logprobs": [-0.1, -0.2],
                "reward": 1.0,
            }
        ],
        "checkpoint_n": checkpoint_n,
    }


def test_save_entry_writes_file(tmp_path: Path):
    entry = _make_entry(prompt_idx=42)
    path = save_entry(entry, tmp_path)
    assert path.parent == tmp_path
    assert path.suffix == ".pt"
    assert path.exists()
    assert "42_" in path.name  # prompt_idx prefix


def test_save_entry_atomic_no_tmp_leftover(tmp_path: Path):
    """Atomicity check: after save_entry returns, no .tmp file remains."""
    save_entry(_make_entry(prompt_idx=7), tmp_path)
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == []


def test_save_then_load_roundtrip(tmp_path: Path):
    original = _make_entry(prompt_idx=99, checkpoint_n=14)
    save_entry(original, tmp_path)
    loaded = load_pool(tmp_path, local_checkpoint_n=14)
    assert len(loaded) == 1
    assert loaded[0]["prompt_idx"] == 99
    assert loaded[0]["checkpoint_n"] == 14
    # Tensors round-trip.
    assert loaded[0]["rollouts"][0]["hidden_states_cpu"].shape == (2, 8)


def test_load_pool_returns_mtime_sorted(tmp_path: Path):
    """Older bakes drain first — load_pool sorts by file mtime ascending."""
    p1 = save_entry(_make_entry(prompt_idx=1), tmp_path)
    time.sleep(0.01)
    p2 = save_entry(_make_entry(prompt_idx=2), tmp_path)
    time.sleep(0.01)
    p3 = save_entry(_make_entry(prompt_idx=3), tmp_path)
    loaded = load_pool(tmp_path, local_checkpoint_n=14)
    assert [e["prompt_idx"] for e in loaded] == [1, 2, 3]


def test_delete_entry_removes_file(tmp_path: Path):
    path = save_entry(_make_entry(prompt_idx=5), tmp_path)
    assert path.exists()
    delete_entry(path)
    assert not path.exists()


def test_delete_entry_idempotent_on_missing(tmp_path: Path):
    """delete_entry must not raise on a non-existent file —
    races between concurrent restarts or already-fired entries
    are benign."""
    delete_entry(tmp_path / "ghost.pt")  # must not raise


def test_load_pool_skips_corrupt_files(tmp_path: Path, caplog):
    """A .pt file that fails torch.load is logged WARNING and skipped;
    healthy files in the same dir are still returned."""
    save_entry(_make_entry(prompt_idx=11), tmp_path)
    (tmp_path / "corrupt.pt").write_bytes(b"not a torch save")
    with caplog.at_level(logging.WARNING, logger="reliquary.miner.pool_persistence"):
        loaded = load_pool(tmp_path, local_checkpoint_n=14)
    assert len(loaded) == 1
    assert loaded[0]["prompt_idx"] == 11
    assert any("corrupt" in r.message for r in caplog.records)


def test_load_pool_keeps_stale_ckpt_but_warns(tmp_path: Path, caplog):
    """Entries baked under a different checkpoint are kept (optimistic)
    but tagged with a WARNING so the operator sees the count."""
    save_entry(_make_entry(prompt_idx=21, checkpoint_n=14), tmp_path)
    save_entry(_make_entry(prompt_idx=22, checkpoint_n=20), tmp_path)
    with caplog.at_level(logging.WARNING, logger="reliquary.miner.pool_persistence"):
        loaded = load_pool(tmp_path, local_checkpoint_n=20)
    assert len(loaded) == 2
    assert {e["prompt_idx"] for e in loaded} == {21, 22}
    assert any("stale" in r.message.lower() or "ckpt" in r.message.lower()
               for r in caplog.records)


def test_load_pool_handles_local_n_sentinel(tmp_path: Path):
    """At startup local_n=-1 (sentinel before first /state). All
    persisted entries differ from -1, but we keep them silently
    (the sentinel itself is the noise-source, not a real ckpt advance)."""
    save_entry(_make_entry(prompt_idx=31, checkpoint_n=14), tmp_path)
    loaded = load_pool(tmp_path, local_checkpoint_n=-1)
    assert len(loaded) == 1


def test_load_pool_empty_dir_returns_empty(tmp_path: Path):
    """Fresh launch: empty dir → []."""
    assert load_pool(tmp_path, local_checkpoint_n=-1) == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/reliquary-miner-priv && pytest tests/unit/test_pool_persistence.py -v
```

Expected: `ModuleNotFoundError: No module named 'reliquary.miner.pool_persistence'`

- [ ] **Step 3: Create the module**

Create `reliquary/miner/pool_persistence.py`:

```python
"""Disk-backed entry store for the miner's pre-baked rollout pool.

Survives miner restarts: entries baked under one launch are reloaded
on the next, so a restart costs no forfeit window beyond cold-start
vLLM warmup.

Layout: one .pt file per entry under ``pool_dir``, named
``<prompt_idx>_<timestamp_ns>.pt``. Saves are atomic via .tmp + rename.
Loads scan the directory and sort by mtime so older bakes drain first.
Corrupt files are logged and skipped (left on disk for forensic
inspection). Entries baked under a different checkpoint than the
current local_n are kept (optimistic policy — matches the live
RELIQUARY_DROP_POOL_ON_CKPT=0 default).

All functions are synchronous and called from inside the existing
``self._pool_lock`` in engine.py — no new concurrency surface.
"""

import logging
import os
import time
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def save_entry(entry: dict, pool_dir: Path) -> Path:
    """Atomically persist a baked entry. Returns the final path.

    The path is stored back on the entry as ``entry["_persist_path"]``
    so the fire path can locate the file to delete on success.
    """
    pool_dir = Path(pool_dir)
    pool_dir.mkdir(parents=True, exist_ok=True)
    prompt_idx = int(entry["prompt_idx"])
    ts_ns = time.time_ns()
    final = pool_dir / f"{prompt_idx}_{ts_ns}.pt"
    tmp = pool_dir / f"{prompt_idx}_{ts_ns}.pt.tmp"
    torch.save(entry, tmp)
    os.rename(tmp, final)  # atomic on POSIX
    entry["_persist_path"] = final
    return final


def delete_entry(path: Path) -> None:
    """Remove a persisted entry. Idempotent: missing file is silently
    skipped (races between a concurrent restart's reload and an
    already-fired entry are benign)."""
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass


def load_pool(pool_dir: Path, local_checkpoint_n: int) -> list[dict]:
    """Scan ``pool_dir`` and torch.load every ``*.pt`` file.

    Returns entries sorted by file mtime ascending so older bakes are
    drained first by the fire path. Corrupt files are logged at WARNING
    and skipped. Entries whose ``checkpoint_n`` differs from
    ``local_checkpoint_n`` are KEPT (optimistic) but tagged with a
    single summary WARNING so operators see the count. When
    ``local_checkpoint_n == -1`` (sentinel before first /state) the
    staleness warning is suppressed — the sentinel itself is the
    mismatch source, not a real ckpt advance.

    Each loaded entry has ``entry["_persist_path"]`` set to the file
    path so the fire path can locate and delete it on success.
    """
    pool_dir = Path(pool_dir)
    if not pool_dir.exists():
        return []

    files = sorted(pool_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    loaded: list[dict] = []
    stale_count = 0
    for f in files:
        try:
            entry = torch.load(f, map_location="cpu", weights_only=False)
        except Exception as e:
            logger.warning(
                "pool_persistence: skipping corrupt file %s (%s)", f, e,
            )
            continue
        entry["_persist_path"] = f
        if local_checkpoint_n != -1 and entry.get("checkpoint_n") != local_checkpoint_n:
            stale_count += 1
        loaded.append(entry)

    if stale_count:
        logger.warning(
            "pool_persistence: %d/%d reloaded entries have stale ckpt "
            "(local=%d); keeping optimistically",
            stale_count, len(loaded), local_checkpoint_n,
        )
    return loaded
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/reliquary-miner-priv && pytest tests/unit/test_pool_persistence.py -v
```

Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
cd ~/reliquary-miner-priv && git add reliquary/miner/pool_persistence.py tests/unit/test_pool_persistence.py && git commit -m "feat(miner): add disk-backed pool persistence module

New module reliquary/miner/pool_persistence.py with three pure
functions: save_entry (atomic .tmp + rename), delete_entry
(idempotent), load_pool (mtime-sorted, corrupt-skipping, stale-ckpt-
keeping). No integration yet — that's the next commit.

10 unit tests cover the round-trip, atomicity, idempotent delete,
corrupt-file recovery, mtime ordering, stale-ckpt warning behavior,
and the local_n=-1 sentinel case.

Implements section 'Change C / module' of
docs/superpowers/specs/2026-05-17-pool-prefilter-persistence-design.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Wire persistence into engine.py

**Files:**
- Modify: `reliquary/miner/engine.py` — reload at startup, save on bake, delete on fire, wipe on ckpt drop
- Modify: `tests/unit/test_r_open_only_burst.py` — add `_pool_dir` to `_StubMiningEngine` so existing tests pass

- [ ] **Step 1: Update `_StubMiningEngine` so existing tests still compile**

In `tests/unit/test_r_open_only_burst.py`, find the `_StubMiningEngine` class. Add to the existing field block (near `self._fired_windows`):

```python
        self._inflight_fire_tasks: set = set()
        # Test stub: pool persistence is not exercised by these tests,
        # but the trigger loop code does not access self._pool_dir, so
        # we don't need to set it. The fire path will (Task 4) — but
        # those branches are gated on entries having "_persist_path",
        # which the stub never sets. Setting None here makes attribute
        # access explicit.
        self._pool_dir = None
```

(If the existing stub does NOT have `_inflight_fire_tasks` near where you're editing, just add `self._pool_dir = None` at the end of `__init__`.)

- [ ] **Step 2: Run existing tests to verify they still pass**

```bash
cd ~/reliquary-miner-priv && pytest tests/unit/test_r_open_only_burst.py -v
```

Expected: 13 passed (or whatever the existing count is, no regressions).

- [ ] **Step 3: Add reload at startup**

In `reliquary/miner/engine.py`, find the block in `mine_window` that initializes the pool. The current code looks like:

```python
        # Shared state. Mutated by both the generator (writer) and trigger
        # loops (reader/draining writer). Protected by ``_pool_lock``.
        self._pool: list[dict] = []
        self._pool_lock = asyncio.Lock()
        self._pool_max_size = int(_os.environ.get("RELIQUARY_POOL_MAX_SIZE", "200"))
```

Right **after** the `self._pool_max_size = ...` line, add:

```python
        # Disk-backed persistence for the pool. Reload on launch so
        # restarts don't lose pre-baked entries. Entries with stale
        # checkpoint_n are kept optimistically (matches the live
        # RELIQUARY_DROP_POOL_ON_CKPT=0 default).
        from pathlib import Path as _Path
        from reliquary.miner.pool_persistence import load_pool as _load_pool
        self._pool_dir = _Path(
            _os.environ.get("RELIQUARY_POOL_DIR", "/root/reliquary-state/pool"),
        )
        self._pool_dir.mkdir(parents=True, exist_ok=True)
        reloaded = _load_pool(self._pool_dir, self._local_n)
        if reloaded:
            self._pool.extend(reloaded)
            logger.info(
                "pool: reloaded %d entries from %s",
                len(reloaded), self._pool_dir,
            )
```

Note: `self._local_n` is set to `-1` just above (the sentinel). `load_pool` knows about this case and suppresses the staleness warning.

- [ ] **Step 4: Add save_entry call after bake success**

In `reliquary/miner/engine.py`, find `_generator_loop`. The code currently has a block where a successful bake appends to the pool:

```python
                async with self._pool_lock:
                    self._pool.append(new_entry)
                # ... existing log line
```

Right after `self._pool.append(new_entry)` (still inside the `async with self._pool_lock` block), add:

```python
                    # Persist to disk so restarts don't lose this entry.
                    # Failure to save (e.g. disk full) is logged but does
                    # not crash the generator — the entry stays in memory
                    # and gets another save chance on the next iteration
                    # if it survives the fire path.
                    try:
                        from reliquary.miner.pool_persistence import save_entry
                        save_entry(new_entry, self._pool_dir)
                    except OSError as e:
                        logger.error(
                            "pool_persistence: save failed for prompt=%d (%s); "
                            "entry kept in memory only",
                            new_entry["prompt_idx"], e,
                        )
```

- [ ] **Step 5: Add delete_entry calls after fire success**

In `reliquary/miner/engine.py`, find `_fire_for_window`. The asyncio.gather call collects results from `_submit_entry` coroutines. **After** the gather returns, iterate over the fired entries and delete their on-disk files. The current code looks like:

```python
        await asyncio.gather(
            *(self._submit_entry(e, state, url, client, results) for e in fire),
            return_exceptions=True,
        )
```

Right **after** that `await asyncio.gather` line, add:

```python
        # Delete persisted files for fired entries. Failure is benign
        # (load_pool would re-include them after restart and the
        # validator hash-dedupe handles the duplicate).
        from reliquary.miner.pool_persistence import delete_entry
        for e in fire:
            persist_path = e.get("_persist_path")
            if persist_path is not None:
                delete_entry(persist_path)
```

- [ ] **Step 6: Wipe on ckpt-drop path**

In `reliquary/miner/engine.py`, find the `drop_on_ckpt` branch in `_trigger_loop` — the block where `self._pool = []` clears the in-memory pool when the operator opts in to dropping on checkpoint advance. The current code is:

```python
                    if drop_on_ckpt:
                        async with self._pool_lock:
                            dropped = len(self._pool)
                            self._pool = []
```

After `self._pool = []`, add the on-disk wipe:

```python
                        # On-disk pool follows the same drop policy.
                        # rmtree + mkdir is the simplest atomic-enough
                        # operation for a directory we own outright.
                        import shutil
                        if self._pool_dir is not None and self._pool_dir.exists():
                            shutil.rmtree(self._pool_dir)
                            self._pool_dir.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 7: Run all tests to verify no regressions**

```bash
cd ~/reliquary-miner-priv && pytest tests/unit/test_pool_persistence.py tests/unit/test_pre_bake_out_of_zone.py tests/unit/test_r_open_only_burst.py tests/unit/test_clock_offset.py -v
```

Expected: all pass (10 + 7 + 13 + 1 = 31 minimum).

- [ ] **Step 8: Smoke-import to catch any wiring typo**

```bash
cd ~/reliquary-miner-priv && python -c "
from reliquary.miner.engine import MiningEngine, _skip_for_out_of_zone
from reliquary.miner.pool_persistence import save_entry, delete_entry, load_pool
print('imports ok')
"
```

Expected: `imports ok`.

- [ ] **Step 9: Commit**

```bash
cd ~/reliquary-miner-priv && git add reliquary/miner/engine.py tests/unit/test_r_open_only_burst.py && git commit -m "feat(miner): wire pool_persistence into engine — reload, save, delete

Three wire-ins:

* mine_window reload: at startup, after the model load and before the
  bg generator task, load_pool(self._pool_dir, self._local_n=-1) and
  extend self._pool. Sentinel -1 suppresses the stale-ckpt warning
  for the legitimate first /state-poll gap.

* _generator_loop save: after self._pool.append(new_entry), call
  save_entry. OSError is caught + logged (disk-full does not crash
  the generator).

* _fire_for_window delete: after asyncio.gather, iterate fired
  entries and call delete_entry on each entry['_persist_path']. The
  missing-file branch of delete_entry handles concurrent races.

* drop_on_ckpt: when the optional RELIQUARY_DROP_POOL_ON_CKPT=1 path
  fires, rmtree + mkdir the on-disk pool dir so the in-memory and
  on-disk pools stay in sync.

Test stub _StubMiningEngine gets self._pool_dir = None so the
existing test_r_open_only_burst.py tests keep passing (the fire
path's persist-delete branch is gated on _persist_path being set,
which the stub never sets).

Completes the spec at
docs/superpowers/specs/2026-05-17-pool-prefilter-persistence-design.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Final smoke + diff inspection

**Files:** none modified; this is verification.

- [ ] **Step 1: Run the full new test surface**

```bash
cd ~/reliquary-miner-priv && pytest tests/unit/test_pool_persistence.py tests/unit/test_pre_bake_out_of_zone.py tests/unit/test_r_open_only_burst.py tests/unit/test_clock_offset.py -v
```

Expected: 31+ passed, 0 failed.

- [ ] **Step 2: Repo-wide test smoke (skip pre-existing failures)**

```bash
cd ~/reliquary-miner-priv && pytest tests/unit/ -x --ignore=tests/unit/test_archive_window_content.py 2>&1 | tail -15
```

Expected: either all pass, or failures are clearly unrelated to this work (e.g. pre-existing validator-side test_state_machine, etc.).

- [ ] **Step 3: Smoke the miner CLI**

```bash
cd ~/reliquary-miner-priv && python -m reliquary.cli.main --help 2>&1 | tail -5
```

Expected: help text prints. No import errors.

- [ ] **Step 4: Inspect the diff**

```bash
cd ~/reliquary-miner-priv && git log --oneline 36307a1..HEAD
cd ~/reliquary-miner-priv && git diff 36307a1..HEAD --stat
```

Expected: 4 new commits (Tasks 1-4). engine.py grew ~60 lines, new pool_persistence.py ~75 lines, two new test files. No unrelated files touched.

- [ ] **Step 5: Stop here for review**

Do NOT deploy to the prod miner box (`86.38.238.199`) without explicit user approval. The fire path's per-fired-entry delete branch is new and runs inside the existing `asyncio.gather` flow — a deploy needs the user's restart procedure to clear the previous instance's in-memory pool before the new instance reloads from disk.

## Self-review notes

- Spec section 'Change A' → Task 1 (helper + integration + tests).
- Spec section 'Change B' → Task 2 (one-literal change).
- Spec section 'Change C / module' → Task 3 (new module + tests).
- Spec section 'Change C / integration' → Task 4 (4 wire-ins).
- Spec section 'Edge cases / Empty pool dir on first launch' → covered by `test_load_pool_empty_dir_returns_empty` in Task 3.
- Spec section 'Edge cases / Corrupt .pt file' → `test_load_pool_skips_corrupt_files` in Task 3.
- Spec section 'Edge cases / Checkpoint advance during reload' → `test_load_pool_handles_local_n_sentinel` in Task 3.
- Spec section 'Edge cases / Disk full during save_entry' → `try/except OSError` in Task 4 Step 4.
- Spec section 'Edge cases / Restart-induced reload duplicates' → spec acceptance: hash-dedup on validator side. Plan does not add miner-side dedup (out of scope).
- Spec section 'Out of scope' — no tasks for `distribution_suspicious`, `bad_termination`, compression, periodic sweep, multi-process. Each is one-function follow-up if the operator wants it later.
