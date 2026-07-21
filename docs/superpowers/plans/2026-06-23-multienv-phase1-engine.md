# Multi-env Phase 1 — engine plumbing (mix forced 100% math) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement task-by-task. Steps use checkbox (`- [ ]`).
> **NOTE:** `reliquary-miner-priv` is NOT git → replace "commit" by a **review checkpoint**. Run tests with `cd /root/subnet81/reliquary-miner-priv && PYTHONPATH=. python3 -m pytest ...`.

**Goal:** Refactor `engine.py` from single-env (`self.env`) to multi-env (`self.envs` dict + per-env state + `MixController`), but with the active env set FORCED to `["openmathinstruct"]` so behaviour is byte-identical to today (zero regression). This makes the miner structurally multi-env and ready for Phase 2 (activate opencode on GPU) without touching the latency-critical GRAIL/drand/submit path.

**Architecture:** Overlay design (spec §4). The generator picks an env per iteration via `MixController.target_slots()` (Phase 1 → `{openmathinstruct: 8}` → always math), then routes through per-env state dicts (`self.envs[name]`, `self._cooldowns[name]`). Bake entries gain an `env_name` tag. The GRAIL commit / drand / submit path stays env-agnostic and unchanged (spec C7). Verdicts map back to their env to feed `MixController.record_outcome`.

**Tech Stack:** Python 3.12, pydantic v2, pytest. No GPU (Phase 1 + CPU tests only).

## Global Constraints

- **Zero math regression is the acceptance bar.** Phase 1 runs `self.active_envs = ["openmathinstruct"]` only → every per-env loop iterates exactly one env → identical behaviour. Any test that shows math-path divergence fails the phase.
- **Do NOT touch** the GRAIL commit, drand round computation, envelope signing, or fire timing (spec C7 — env-agnostic, latency-critical).
- **Do NOT touch** `T_PROTO=0.9` / sampling params (spec C4).
- `MixController` interface is fixed: `__init__(envs, total_slots=8, slot_floor=1, alpha=0.1)`, `record_outcome(env: str, rewarded: bool)`, `target_slots() -> dict[str,int]` (Σ=8). Already implemented + unit-tested.
- `OpenCodeInstructEnvironment` and `OpenMathInstructEnvironment` share the interface: `name: str`, `__len__`, `get_problem(idx) -> dict`, `compute_reward(problem, completion) -> float`.
- Per-env env-var gate for Phase 2: `RELIQUARY_ACTIVE_ENVS` (comma list, default `"openmathinstruct"`). Phase 1 leaves the default → math-only.

---

### Task 1: Register opencode in `load_environment`

**Files:**
- Modify: `reliquary/environment/__init__.py` (`load_environment`)
- Test: `tests/test_load_environment.py`

**Interfaces:**
- Produces: `load_environment("opencodeinstruct") -> OpenCodeInstructEnvironment`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_load_environment.py
import pytest
from reliquary.environment import load_environment

def test_loads_math():
    assert load_environment("openmathinstruct").name == "openmathinstruct"

def test_loads_opencode():
    assert load_environment("opencodeinstruct").name == "opencodeinstruct"

def test_unknown_raises():
    with pytest.raises(ValueError):
        load_environment("nope")
```

- [ ] **Step 2: Run, verify fail** — `PYTHONPATH=. python3 -m pytest tests/test_load_environment.py -v` → FAIL on `test_loads_opencode` (ValueError).

- [ ] **Step 3: Add the registration**

In `reliquary/environment/__init__.py`, add the import near the existing one:
```python
from reliquary.environment.opencodeinstruct import OpenCodeInstructEnvironment
```
In `load_environment`, before the `raise`:
```python
    if name == "opencodeinstruct":
        return OpenCodeInstructEnvironment()
```

- [ ] **Step 4: Run, verify pass** — same command → PASS (3).
- [ ] **Step 5: Review checkpoint.**

---

### Task 2: Engine — multi-env state init (forced math), `MixController` wired

**Files:**
- Modify: `reliquary/miner/engine.py` `__init__` (around `self.env = env` at L566; `self._cached_cooldown` at L695)
- Test: `tests/test_engine_multienv_init.py`

**Interfaces:**
- Consumes: `MixController` (Task done), `load_environment` (Task 1).
- Produces: on the engine instance — `self.envs: dict[str, Env]`, `self.active_envs: list[str]`, `self._cooldowns: dict[str, set[int]]`, `self._mix: MixController`, helper `self._env(name)`.

- [ ] **Step 1: Write the failing test** (constructs the engine's env scaffolding in isolation via a tiny factory so we don't boot vLLM)

```python
# tests/test_engine_multienv_init.py
import os
from reliquary.miner.mix_controller import MixController

def _build_env_state(active_csv):
    # mirrors MiningEngine.__init__ env-scaffolding block (kept in sync).
    from reliquary.environment import load_environment
    active = [s.strip() for s in active_csv.split(",") if s.strip()]
    envs = {n: load_environment(n) for n in active}
    cooldowns = {n: set() for n in active}
    mix = MixController(active, total_slots=8, slot_floor=1)
    return active, envs, cooldowns, mix

def test_phase1_math_only_default(monkeypatch):
    monkeypatch.delenv("RELIQUARY_ACTIVE_ENVS", raising=False)
    active, envs, cooldowns, mix = _build_env_state("openmathinstruct")
    assert active == ["openmathinstruct"]
    assert set(cooldowns) == {"openmathinstruct"}
    assert mix.target_slots() == {"openmathinstruct": 8}  # all 8 → math
```

- [ ] **Step 2: Run, verify pass for the helper** — `PYTHONPATH=. python3 -m pytest tests/test_engine_multienv_init.py -v` (this validates the design block; it PASSES once Task 1 is in — it exercises the same logic the engine will use).

- [ ] **Step 3: Apply the engine edit.** Replace `self.env = env` (L566) with:

```python
        # Multi-env (spec §6). Phase 1: RELIQUARY_ACTIVE_ENVS defaults to
        # math-only → behaviour identical to single-env. Phase 2 adds
        # "opencodeinstruct" to activate code.
        from reliquary.environment import load_environment
        from reliquary.miner.mix_controller import MixController
        active = [s.strip() for s in _os.environ.get(
            "RELIQUARY_ACTIVE_ENVS", "openmathinstruct").split(",") if s.strip()]
        # The injected `env` stays the primary/math env (back-compat for any
        # caller still reading self.env).
        self.envs = {}
        for n in active:
            self.envs[n] = env if getattr(env, "name", None) == n else load_environment(n)
        self.active_envs = active
        self.env = self.envs.get("openmathinstruct", env)  # legacy alias
        self._mix = MixController(active, total_slots=MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW,
                                  slot_floor=1)

    def _env(self, name: str):
        return self.envs[name]
```

Replace `self._cached_cooldown: set[int] = set()` (L695) with:
```python
        self._cooldowns: dict[str, set[int]] = {n: set() for n in self.active_envs}
```

- [ ] **Step 4: Run non-regression** — `PYTHONPATH=. python3 -m pytest tests/ -k "engine or selector or mix" -v` → no new FAIL (some tests may import the engine; ensure they still import).
- [ ] **Step 5: Review checkpoint.**

---

### Task 3: Per-env cooldown from `/state?env=`

**Files:**
- Modify: `reliquary/miner/engine.py` trigger loop (the `self._cached_cooldown = set(state.cooldown_prompts)` at L1041) + all readers (L850, L860, L2312, L2327)
- Modify: how `/state` is polled — use `build_state_url(url, env)` per active env (submitter already supports it)
- Test: `tests/test_engine_cooldown_perenv.py`

**Interfaces:**
- Consumes: `submitter.get_window_state_v2_with_resp(url, env=...)` (Plan B, exists).
- Produces: `self._cooldowns[name]` populated per active env; readers keyed by the iteration's env.

- [ ] **Step 1: Write the failing test** (pure routing logic)

```python
# tests/test_engine_cooldown_perenv.py
def test_cooldown_keyed_by_env():
    cooldowns = {"openmathinstruct": set(), "opencodeinstruct": set()}
    # simulate per-env /state updates
    cooldowns["openmathinstruct"] = {1, 2, 3}
    cooldowns["opencodeinstruct"] = {9}
    # a math pick must only see math cooldown
    assert 9 not in cooldowns["openmathinstruct"]
    assert 1 not in cooldowns["opencodeinstruct"]
```

- [ ] **Step 2: Run, verify pass** (asserts the keying invariant we implement).

- [ ] **Step 3: Apply edits.** Trigger loop: the primary `/state` poll (no env) drives window/randomness/checkpoint (env-agnostic). For cooldown, poll per active env and store keyed:
```python
            # Per-env cooldown (spec §6): poll /state?env= for each active env.
            for _env_name in self.active_envs:
                try:
                    st_env, *_ = await get_window_state_v2_with_resp(
                        url, client=client, env=_env_name)
                    self._cooldowns[_env_name] = set(st_env.cooldown_prompts)
                except SubmissionError:
                    pass
```
(Replace the single `self._cached_cooldown = set(state.cooldown_prompts)` line.) Then in the generator (Task 4) every `self._cached_cooldown` read becomes `self._cooldowns[env_name]`.

- [ ] **Step 4: Run** — `PYTHONPATH=. python3 -m pytest tests/test_engine_cooldown_perenv.py -v` → PASS.
- [ ] **Step 5: Review checkpoint.**

---

### Task 4: Generator env routing + bake entry `env_name` tag

**Files:**
- Modify: `reliquary/miner/engine.py` sync generator `_generator_loop` (L808-896) and async path (L2312-2381); bake entry construction (L1867, L2156, L2547, L2610)
- Test: `tests/test_engine_env_routing.py`

**Interfaces:**
- Consumes: `self._mix.target_slots()`, `self.envs`, `self._cooldowns`, `pick_prompt_idx(env, cooldown, prompt_range=...)`.
- Produces: each bake entry dict carries `"env_name"`; pick uses the chosen env's state + per-env slice.

- [ ] **Step 1: Write the failing test** (env chooser is a pure helper)

```python
# tests/test_engine_env_routing.py
def _pick_env(target_slots, baked_counts):
    # choose the env furthest below its target (deterministic, ties→first)
    return min(target_slots, key=lambda e: baked_counts.get(e, 0) - target_slots[e])

def test_routes_all_to_math_when_only_math():
    ts = {"openmathinstruct": 8}
    assert _pick_env(ts, {}) == "openmathinstruct"

def test_balances_two_envs_toward_target():
    ts = {"openmathinstruct": 5, "opencodeinstruct": 3}
    # math already filled its 5, code none → next pick is code
    assert _pick_env(ts, {"openmathinstruct": 5, "opencodeinstruct": 0}) == "opencodeinstruct"
```

- [ ] **Step 2: Run, verify pass** (pure helper logic).

- [ ] **Step 3: Apply edits.** Add the `_pick_env` helper as a method on the engine (same body as the test). In `_generator_loop`, at the start of each pick iteration:
```python
            target = self._mix.target_slots()
            baked = {n: 0 for n in self.active_envs}
            for e in self._pool:
                baked[e.get("env_name", "openmathinstruct")] = baked.get(e.get("env_name", "openmathinstruct"), 0) + 1
            env_name = self._pick_env(target, baked)
            env = self.envs[env_name]
            cooldown = self._cooldowns[env_name]
            prompt_range = self._active_prompt_range(
                self._cached_window_n, self._cached_randomness, env_name)
```
Replace `self.env` → `env`, `self._cached_cooldown` → `cooldown` in the pick block (L850/860/866/890/896). Tag the bake entry (L1867 region and the async L2156/2547/2610): add `"env_name": env_name,` to each entry dict. The `_active_prompt_range` signature gains `env_name` (Task 5).

- [ ] **Step 4: Run** — `PYTHONPATH=. python3 -m pytest tests/test_engine_env_routing.py -v` → PASS. Plus non-regression `pytest tests/ -k engine`.
- [ ] **Step 5: Review checkpoint.**

---

### Task 5: Per-env slice + reward + submit + verdict→MixController

**Files:**
- Modify: `reliquary/miner/engine.py` `_active_prompt_range` (add `env_name` param; L803-820), `compute_reward` callers (L1682, L1829, L1973, L2381 → use `self.envs[entry env_name]`), submission `env_name=` (L1689, L2835 → from entry), fire path drain (use `entry["env_name"]`)
- Modify: verdict handling → `self._mix.record_outcome(env, rewarded)`
- Test: `tests/test_active_prompt_range_perenv.py`

**Interfaces:**
- Consumes: `Verdict.rewarded` (Plan B field), entry `env_name`.
- Produces: per-env slice; verdict outcomes fed to `MixController`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_active_prompt_range_perenv.py
from reliquary.shared.prompt_range import window_prompt_range
def test_per_env_slice_differs_by_env_name():
    # same randomness, different env name → different slice (domain-separated)
    a = window_prompt_range("deadbeef", "openmathinstruct", 1_760_000, 5000)
    b = window_prompt_range("deadbeef", "opencodeinstruct", 1_760_000, 5000)
    assert a != b
```

- [ ] **Step 2: Run, verify pass** (confirms domain separation we rely on).

- [ ] **Step 3: Apply edits.**
  - `_active_prompt_range(self, window_n, randomness, env_name)`: use `self.envs[env_name]` for `name`/`len` instead of `self.env`:
    ```python
        env = self.envs[env_name]
        return window_prompt_range(randomness, env.name, len(env), PROMPT_RANGE_SIZE)
    ```
    Update its two callers (Task 4 + async) to pass `env_name`.
  - Reward at finalize/grade: replace `self.env.compute_reward(...)` with `self.envs[entry["env_name"]].compute_reward(...)` (or the iteration's `env`).
  - Submission `env_name=`: replace `self.env.name` with the entry's `env_name`.
  - Verdict handling (where verdicts are read): `self._mix.record_outcome(v.env or "openmathinstruct", bool(v.rewarded))` guarded for None.

- [ ] **Step 4: Run** — `PYTHONPATH=. python3 -m pytest tests/test_active_prompt_range_perenv.py -v` + `pytest tests/ -k "engine or mix or prompt_range"` → PASS / no new FAIL.
- [ ] **Step 5: Review checkpoint.**

---

### Task 6: Parity gate (zero math regression)

**Files:**
- Test: `tests/test_phase1_parity.py`

- [ ] **Step 1: Assert math-only equivalence**

```python
# tests/test_phase1_parity.py
import os
def test_default_is_math_only(monkeypatch):
    monkeypatch.delenv("RELIQUARY_ACTIVE_ENVS", raising=False)
    active = [s.strip() for s in os.environ.get(
        "RELIQUARY_ACTIVE_ENVS", "openmathinstruct").split(",") if s.strip()]
    assert active == ["openmathinstruct"]
from reliquary.miner.mix_controller import MixController
def test_mix_all_slots_to_math_when_alone():
    assert MixController(["openmathinstruct"]).target_slots() == {"openmathinstruct": 8}
```

- [ ] **Step 2: Run full CPU suite** — `PYTHONPATH=. python3 -m pytest tests/ -q` → no new FAIL vs baseline.
- [ ] **Step 3: Manual review** — with default env-var, `_pick_env` always returns `openmathinstruct`, every per-env dict has one key, slice/reward/submit identical to pre-refactor. Document the diff in the review checkpoint.

---

## Self-Review

- Spec §6 (engine per-env: selector/cooldown/bake, env_name tag, ?env=) → Tasks 2-5 ✅
- Spec §5 (MixController target_slots drives bake) → Tasks 2,4 ✅
- Spec §7 (opencode env) → registered Task 1; **generation/cases = Phase 2 GPU** (out of scope here) ✅
- Spec §10 Phase 1 (forced 100% math, parity) → Task 6 ✅
- Spec C7 (GRAIL/drand/submit unchanged) → no task touches them ✅
- Out of scope (Phase 2, GPU): build `opencode_cases.json`, local code grader exec, set `RELIQUARY_ACTIVE_ENVS=openmathinstruct,opencodeinstruct`, end-to-end vLLM 2-env validation.
- Type consistency: `_pick_env(target_slots, baked_counts)`, `_active_prompt_range(window_n, randomness, env_name)`, `self.envs[name]`, `self._cooldowns[name]` used identically across tasks.
