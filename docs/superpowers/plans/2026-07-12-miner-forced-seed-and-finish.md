# Miner forced-seed sampling + finish v7/BFT — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.
> **NOTE:** `reliquary-miner-priv` is NOT git → "commit" = **review checkpoint**. Tests: `cd /root/subnet81/reliquary-miner-priv && PYTHONPATH=. python3 -m pytest <path> -v`.

**Goal:** align the miner with the validator's newest gate — **forced-seed sampling** (merged `b790e42`, `FORCED_SEED_ENFORCE=True`) — so it is not `SEED_MISMATCH`-rejected, and finish the remaining v7/BFT CPU work.

**Architecture:** port the validator's deterministic sampler verbatim (`forced_sampling.py` core + `forced_seed_sampler.py` HF `LogitsProcessor`), wire it into the miner's **HF** generation (phase-1 + phase-2 BFT) so every sampled token is the public forced inverse-CDF pick, and advertise `protocol_version=1`. Generation MUST be HF `model.generate` + a `LogitsProcessor` (the forced pick has no clean vLLM equivalent) — this is now the canonical path; vLLM stays frozen/secondary.

**Tech Stack:** Python 3.12, transformers 5.x (LogitsProcessor), torch, pydantic v2, pytest. Generation = HF. No vLLM touched.

## Global Constraints

- **Source of truth = validator `b790e42`** (`/root/subnet81/reliquary`, read via `git show b790e42:<path>` — local working tree is on `d9471f2`). Copy cited blocks **verbatim** (miner↔validator bit-parity: the validator teacher-forces the identical `warp`/`pick`/`u_at`, so any drift lowers the seed-consistency score → `SEED_MISMATCH`).
- **Do NOT touch `reliquary/miner/vllm_backend.py`** (frozen — vLLM does not load Qwen3.5; separate session).
- **Forced-seed applies to ALL envs** (math + code): it gates the general σ/reward sampling, not just BFT. It is orthogonal to BFT — BFT `force_span` positions are **excluded** from seed-consistency validator-side (`1518663`/`5d18f47`), so no miner action there.
- **Sampler = `do_sample=False`, NO `temperature`/`top_k`/`top_p` on `generate()`** — the processor applies the protocol warp (`T_PROTO=0.6`/`TOP_K_PROTO=20`/`TOP_P_PROTO=0.95`) itself; HF's own warpers must NOT run (double-warp = honest false mismatch). `generation_config` logit processors (repetition_penalty, …) must be neutralized.
- **`checkpoint_hash`** is already computed + sent by the miner (`engine.py:1600` `ckpt_hash`) — reuse it as the `u_at` seed input; do not recompute.

## File Structure

- `reliquary/environment/forced_sampling.py` — CREATE, byte-exact core (`warp`/`pick`/`u_at`/`seed_consistency`). Shared primitive.
- `reliquary/miner/forced_seed_sampler.py` — CREATE, byte-exact HF `LogitsProcessor` glue.
- `reliquary/constants.py` — MODIFY, add `FORCED_SEED_*` block.
- `reliquary/protocol/submission.py` — MODIFY, add `protocol_version` field + `SEED_MISMATCH` reason.
- `reliquary/miner/engine.py` — MODIFY, wire processor into `_generate_m_rollouts` (phase-1) + `_bft_from_seqs` (phase-2), emit `protocol_version`.
- `scripts/difficulty_probe.py` — MODIFY, wire processor into `stage_generate_code_hf` (predictor data must match the forced stream).

---

## PART A — Forced-seed sampling (BLOCKING)

### Task 1: Forced-seed constants

**Files:** Modify `reliquary/constants.py` · Test `tests/test_forced_seed_constants.py`

**Produces:** `FORCED_SEED_DOMAIN`, `FORCED_SEED_STOCHASTIC_MAXPROB`, `FORCED_SEED_CONSISTENCY_FLOOR`, `FORCED_SEED_MIN_STOCH_POSITIONS`, `FORCED_SEED_ROLLOUT_FLOOR`, `FORCED_SEED_ROLLOUT_MIN_STOCH`, `FORCED_SEED_ENFORCE`, `FORCED_SEED_PROTOCOL_VERSION`.

- [ ] **Step 1: test** (`tests/test_forced_seed_constants.py`)
```python
import reliquary.constants as c
def test_forced_seed_constants():
    assert c.FORCED_SEED_DOMAIN == "reliquary-forced-seed-v1"
    assert c.FORCED_SEED_STOCHASTIC_MAXPROB == 0.99
    assert c.FORCED_SEED_CONSISTENCY_FLOOR == 0.80
    assert c.FORCED_SEED_MIN_STOCH_POSITIONS == 30
    assert c.FORCED_SEED_ROLLOUT_FLOOR == 0.75
    assert c.FORCED_SEED_ROLLOUT_MIN_STOCH == 20
    assert c.FORCED_SEED_PROTOCOL_VERSION == 1
    assert isinstance(c.FORCED_SEED_ENFORCE, bool)
```
- [ ] **Step 2: run → FAIL** (`AttributeError`).
- [ ] **Step 3:** append to `reliquary/constants.py` (values verbatim from validator `b790e42:reliquary/constants.py:660-695`; use `import os as _os` if not already imported):
```python
# ──────────────── FORCED-SEED SAMPLING (validator b790e42) ────────────────
FORCED_SEED_DOMAIN = "reliquary-forced-seed-v1"
FORCED_SEED_STOCHASTIC_MAXPROB = 0.99
FORCED_SEED_CONSISTENCY_FLOOR = 0.80
FORCED_SEED_MIN_STOCH_POSITIONS = 30
FORCED_SEED_ROLLOUT_FLOOR = 0.75
FORCED_SEED_ROLLOUT_MIN_STOCH = 20
FORCED_SEED_ENFORCE = _os.environ.get(
    "FORCED_SEED_ENFORCE", "true"
).strip().lower() in ("1", "true", "yes", "on")
FORCED_SEED_PROTOCOL_VERSION = 1
```
- [ ] **Step 4: run → PASS.**
- [ ] **Step 5: checkpoint.**

### Task 2: Port `forced_sampling.py` (byte-exact core)

**Files:** Create `reliquary/environment/forced_sampling.py` · Test `tests/test_forced_sampling.py`

**Consumes:** `FORCED_SEED_DOMAIN` (Task 1).
**Produces:** `warp(logits, t, top_k, top_p)`, `pick(probs, u)->int`, `u_at(randomness, hotkey, prompt_idx, checkpoint_hash, rollout_index, t)->float`, `seed_consistency(logits, token_ids, u_values, *, t, top_k, top_p, stochastic_threshold)->(n_stoch, n_match)`.

- [ ] **Step 1: test** (`tests/test_forced_sampling.py`) — the honest-sampler round-trip: a token produced by `pick(warp(logits), u_at(...))` must match the forced pick that `seed_consistency` recomputes.
```python
import torch
from reliquary.environment.forced_sampling import warp, pick, u_at, seed_consistency

def test_pick_matches_seed_consistency_roundtrip():
    torch.manual_seed(0)
    logits = torch.randn(5, 100)  # 5 positions, vocab 100
    us = [u_at("rand", "hk", 3, "abc", 0, t) for t in range(5)]
    toks = [pick(warp(logits[t], t=0.6, top_k=20, top_p=0.95), us[t]) for t in range(5)]
    n_stoch, n_match = seed_consistency(logits, toks, us, t=0.6, top_k=20, top_p=0.95,
                                        stochastic_threshold=0.99)
    assert n_match == n_stoch  # honest forced picks are 100% consistent

def test_u_at_is_deterministic_and_in_unit_interval():
    a = u_at("r", "h", 1, "c", 0, 7)
    assert a == u_at("r", "h", 1, "c", 0, 7) and 0.0 <= a < 1.0
    assert a != u_at("r", "h", 1, "c", 1, 7)  # different rollout → different u
```
- [ ] **Step 2: run → FAIL** (module missing).
- [ ] **Step 3:** create `reliquary/environment/forced_sampling.py` — **copy VERBATIM** from `git show b790e42:reliquary/environment/forced_sampling.py` (96 lines: module docstring, `warp`, `pick`, `_lp`, `u_at`, `_warp_batch`, `seed_consistency`). Full text:
```python
"""Protocol-fixed sampler shared by miner (generation) and validator (verification).

The per-position draw is a public deterministic function of window randomness, so
there is exactly one legal generation per (miner, prompt, rollout, window). A rollout
not generated from this draw is detectable by teacher-forced consistency.
"""
from __future__ import annotations

import hashlib

import torch

from reliquary.constants import FORCED_SEED_DOMAIN


def warp(logits: torch.Tensor, t: float, top_k: int, top_p: float) -> torch.Tensor:
    """Temperature -> top-k -> top-p, returned in canonical (token-id ascending) order."""
    lg = logits.float() / float(t)
    if top_k and top_k > 0:
        k = min(top_k, lg.numel())
        kth = torch.topk(lg, k).values[-1]
        lg = torch.where(lg < kth, torch.full_like(lg, float("-inf")), lg)
    probs = torch.softmax(lg, dim=-1)
    if top_p and top_p < 1.0:
        sp, si = torch.sort(probs, descending=True)
        cum = torch.cumsum(sp, dim=-1)
        sp = torch.where((cum - sp) < top_p, sp, torch.zeros_like(sp))  # include crossing token
        probs = torch.zeros_like(probs).scatter(-1, si, sp)
    return probs / probs.sum()


def pick(probs: torch.Tensor, u: float) -> int:
    """First token id whose cumulative probability exceeds u (inverse-CDF)."""
    cdf = torch.cumsum(probs, dim=-1)
    u_tensor = torch.tensor(float(u), device=cdf.device, dtype=cdf.dtype)
    idx = int(torch.searchsorted(cdf, u_tensor, right=True))
    return min(idx, probs.numel() - 1)


def _lp(b: bytes) -> bytes:
    return len(b).to_bytes(2, "big") + b


def u_at(randomness: str, hotkey: str, prompt_idx: int, checkpoint_hash: str,
         rollout_index: int, t: int) -> float:
    """Public uniform in [0, 1) for rollout `rollout_index`, completion position `t`."""
    msg = (FORCED_SEED_DOMAIN.encode()
           + _lp(randomness.encode()) + _lp(hotkey.encode())
           + int(prompt_idx).to_bytes(8, "big")
           + _lp(checkpoint_hash.encode())
           + int(rollout_index).to_bytes(4, "big")
           + int(t).to_bytes(4, "big"))
    return int.from_bytes(hashlib.sha256(msg).digest()[:8], "big") / 2.0**64


def _warp_batch(logits: torch.Tensor, t: float, top_k: int, top_p: float) -> torch.Tensor:
    """Row-batched ``warp``: logits [n, vocab] -> probs [n, vocab], bit-identical
    per row to the 1-D ``warp`` (each op is independent along dim=-1) but with no
    per-row Python loop."""
    lg = logits.float() / float(t)
    if top_k and top_k > 0:
        k = min(top_k, lg.shape[-1])
        kth = torch.topk(lg, k, dim=-1).values[..., -1:]
        lg = torch.where(lg < kth, torch.full_like(lg, float("-inf")), lg)
    probs = torch.softmax(lg, dim=-1)
    if top_p and top_p < 1.0:
        sp, si = torch.sort(probs, descending=True, dim=-1)
        cum = torch.cumsum(sp, dim=-1)
        sp = torch.where((cum - sp) < top_p, sp, torch.zeros_like(sp))
        probs = torch.zeros_like(probs).scatter(-1, si, sp)
    return probs / probs.sum(dim=-1, keepdim=True)


def seed_consistency(logits: torch.Tensor, token_ids: list[int], u_values: list[float], *,
                     t: float, top_k: int, top_p: float,
                     stochastic_threshold: float) -> tuple[int, int]:
    """Teacher-forced check. logits is [n, vocab] predicting token_ids[i] at u_values[i].
    Counts stochastic positions (max_prob < threshold) and how many match the forced pick."""
    n = min(len(token_ids), len(u_values), int(logits.shape[0]))
    if n == 0:
        return 0, 0
    probs = _warp_batch(logits[:n], t=t, top_k=top_k, top_p=top_p)
    stochastic = probs.max(dim=-1).values < stochastic_threshold
    cdf = torch.cumsum(probs, dim=-1)
    u = torch.tensor([float(x) for x in u_values[:n]],
                     device=cdf.device, dtype=cdf.dtype).unsqueeze(-1)
    picks = torch.searchsorted(cdf, u, right=True).squeeze(-1).clamp(max=probs.shape[-1] - 1)
    toks = torch.tensor([int(x) for x in token_ids[:n]],
                        device=picks.device, dtype=picks.dtype)
    matched = stochastic & (picks == toks)
    return int(stochastic.sum().item()), int(matched.sum().item())
```
> **Parity gate:** after creating, run `diff <(git -C /root/subnet81/reliquary show b790e42:reliquary/environment/forced_sampling.py) reliquary/environment/forced_sampling.py` → **must be empty**.
- [ ] **Step 4: run → PASS.**
- [ ] **Step 5: byte-parity diff empty (above). Checkpoint.**

### Task 3: Port `forced_seed_sampler.py` (byte-exact HF glue)

**Files:** Create `reliquary/miner/forced_seed_sampler.py` · Test `tests/test_forced_seed_sampler.py`

**Consumes:** `warp`/`pick`/`u_at` (Task 2); `T_PROTO`/`TOP_K_PROTO`/`TOP_P_PROTO` (constants).
**Produces:** `ForcedSeedLogitsProcessor(*, randomness, hotkey, prompt_idx, checkpoint_hash, rollout_indices, base_offsets, start_len, temperature=T_PROTO, top_k=TOP_K_PROTO, top_p=TOP_P_PROTO)`, `forced_seed_generate_kwargs(base_kwargs, processor)->dict`, `phase2_base_offsets(primed_lengths, prompt_length)->list[int]`.

- [ ] **Step 1: test** (`tests/test_forced_seed_sampler.py`) — the processor returns a one-hot at the forced pick; `forced_seed_generate_kwargs` strips warpers + sets `do_sample=False`.
```python
import torch
from reliquary.miner.forced_seed_sampler import (
    ForcedSeedLogitsProcessor, forced_seed_generate_kwargs, phase2_base_offsets)
from reliquary.environment.forced_sampling import warp, pick, u_at

def test_processor_forces_the_inverse_cdf_pick():
    proc = ForcedSeedLogitsProcessor(randomness="r", hotkey="h", prompt_idx=1,
        checkpoint_hash="c", rollout_indices=[0], base_offsets=[0], start_len=3)
    scores = torch.randn(1, 50)
    input_ids = torch.zeros(1, 3, dtype=torch.long)  # s = 3 - 3 = 0
    out = proc(input_ids, scores)
    expect = pick(warp(scores[0], t=0.6, top_k=20, top_p=0.95), u_at("r","h",1,"c",0,0))
    assert int(out[0].argmax()) == expect and out[0].max() == 0.0

def test_generate_kwargs_strip_warpers_and_greedy():
    kw = forced_seed_generate_kwargs({"temperature": 0.6, "top_p": 0.95, "top_k": 20}, object())
    assert "temperature" not in kw and kw["do_sample"] is False
    assert kw["repetition_penalty"] == 1.0

def test_phase2_base_offsets():
    assert phase2_base_offsets([5, 3], prompt_length=3) == [2, 0]
```
- [ ] **Step 2: run → FAIL** (module missing).
- [ ] **Step 3:** create `reliquary/miner/forced_seed_sampler.py` — **copy VERBATIM** from `git show b790e42:reliquary/miner/forced_seed_sampler.py` (112 lines: docstring, `_WARPER_KWARGS`, `_NEUTRAL_PROCESSOR_KWARGS`, `ForcedSeedLogitsProcessor`, `forced_seed_generate_kwargs`, `phase2_base_offsets`). It imports `from reliquary.environment.forced_sampling import pick, u_at, warp` and `from reliquary.constants import T_PROTO, TOP_K_PROTO, TOP_P_PROTO`.
> **Parity gate:** `diff <(git -C /root/subnet81/reliquary show b790e42:reliquary/miner/forced_seed_sampler.py) reliquary/miner/forced_seed_sampler.py` → **must be empty**.
- [ ] **Step 4: run → PASS.**
- [ ] **Step 5: byte-parity diff empty. Checkpoint.**

### Task 4: Schema — `protocol_version` + `SEED_MISMATCH`

**Files:** Modify `reliquary/protocol/submission.py` · Test `tests/test_forced_seed_schema.py`

**Produces:** `BatchSubmissionRequest.protocol_version: int` (default 0), `RejectReason.SEED_MISMATCH`.

- [ ] **Step 1: test** (`tests/test_forced_seed_schema.py`)
```python
from reliquary.protocol.submission import BatchSubmissionRequest, RejectReason

def test_protocol_version_defaults_zero_and_accepts_one():
    # minimal construction path used elsewhere in the suite; adapt field names to
    # the existing BatchSubmissionRequest test fixture in tests/unit/test_submitter.py
    assert RejectReason.SEED_MISMATCH.value == "seed_mismatch"
    f = BatchSubmissionRequest.model_fields["protocol_version"]
    assert f.default == 0
```
- [ ] **Step 2: run → FAIL.**
- [ ] **Step 3:** in `reliquary/protocol/submission.py`: add `SEED_MISMATCH = "seed_mismatch"` to `RejectReason`; add to `BatchSubmissionRequest` (verbatim from `b790e42:reliquary/protocol/submission.py`):
```python
    protocol_version: int = Field(default=0, ge=0)
```
and widen `checkpoint_hash` if present: `checkpoint_hash: str = Field(..., min_length=0, max_length=256)`.
- [ ] **Step 4: run → PASS.**
- [ ] **Step 5: checkpoint.**

### Task 5: Wire forced-seed into the miner engine (HF generation)

**Files:** Modify `reliquary/miner/engine.py` (`_generate_m_rollouts` phase-1, `_bft_from_seqs` phase-2, `BatchSubmissionRequest` build) · Test `tests/test_engine_forced_seed_wire.py`

**Consumes:** Tasks 2/3 (`ForcedSeedLogitsProcessor`, `forced_seed_generate_kwargs`, `phase2_base_offsets`), `FORCED_SEED_PROTOCOL_VERSION`.

**Reference (mirror exactly):** `b790e42:reliquary/miner/engine.py` — phase-1 build at ~L684 (`rollout_indices=list(range(M_ROLLOUTS))`, `base_offsets=[0]*M_ROLLOUTS`, `start_len=prompt_length`), phase-2 build at ~L273 (`rollout_indices=list(unfinished_idx)`, `base_offsets=phase2_base_offsets(primed_lengths, prompt_length)`, `start_len=` left-padded width), `protocol_version=FORCED_SEED_PROTOCOL_VERSION` at ~L529.

- [ ] **Step 1:** thread `hotkey` + `prompt_idx` + `checkpoint_hash` into `_generate_m_rollouts(self, problem, randomness, env=None)`. `hotkey` = `self.wallet.hotkey.ss58_address` (confirm the attribute the miner already uses for `hotkey` in the submit path, `engine.py:1600` area); `prompt_idx` = `problem["index"]` (the field `get_problem` sets); `checkpoint_hash` = the same `ckpt_hash` the submit path computes (hoist it or recompute via the existing helper).
- [ ] **Step 2:** in `_generate_m_rollouts`, HF branch (the `self.vllm_model.generate` path), construct the phase-1 processor and route the generate call through `forced_seed_generate_kwargs`:
```python
from reliquary.miner.forced_seed_sampler import (
    ForcedSeedLogitsProcessor, forced_seed_generate_kwargs)
phase1_proc = ForcedSeedLogitsProcessor(
    randomness=randomness, hotkey=hotkey, prompt_idx=prompt_idx,
    checkpoint_hash=checkpoint_hash, rollout_indices=list(range(M_ROLLOUTS)),
    base_offsets=[0] * M_ROLLOUTS, start_len=prompt_length)
base_kwargs = {"max_new_tokens": max_new, "num_return_sequences": 1,
               "pad_token_id": self.tokenizer.pad_token_id}
outputs = self.vllm_model.generate(
    input_tensor, **forced_seed_generate_kwargs(base_kwargs, phase1_proc))
```
(Drop the old `do_sample=True, temperature=T_PROTO, top_p, top_k` — the processor warps. Keep the first-EOS truncation.)
- [ ] **Step 3:** in `_bft_from_seqs(self, seqs, prompt_tokens, *, randomness, hotkey, prompt_idx, checkpoint_hash)` (add these params; pass them from `_generate_m_rollouts`), build the phase-2 processor for the forced rows and merge into `phase2_kwargs` before `bft_rollouts_from_completions`. Because `bft_rollouts_from_completions` calls `model.generate` internally, extend its signature to accept `gen_kwargs` already wired via `forced_seed_generate_kwargs` (the reference passes the processor through `gen_kwargs`). Use `rollout_indices` = the unfinished row indices and `base_offsets = phase2_base_offsets(primed_lengths, len(prompt_tokens))`, `start_len` = the left-padded batch width.
- [ ] **Step 4:** at the `BatchSubmissionRequest(...)` build (`engine.py:1605`), add `protocol_version=FORCED_SEED_PROTOCOL_VERSION`.
- [ ] **Step 5: test** (`tests/test_engine_forced_seed_wire.py`) — a fake `model.generate` that records the kwargs it was called with asserts `do_sample is False`, no `temperature`, and a `logits_processor` containing a `ForcedSeedLogitsProcessor`. (Reuse the fake-model pattern from `tests/test_bft_assemble.py`.)
- [ ] **Step 6:** `PYTHONPATH=. python3 -c "import reliquary.miner.engine"` + full suite → **no NEW fails** vs the 159 baseline (single-env non-forced path must stay byte-identical when `FORCED_SEED_ENFORCE` off is NOT the point — the miner always forces now; assert the fake-model wiring test passes).
- [ ] **Step 7: checkpoint.**

### Task 6: Wire forced-seed into the probe HF generation

**Files:** Modify `scripts/difficulty_probe.py` (`stage_generate_code_hf`) · Test `tests/test_probe_forced_seed.py`

**Why:** predictor labels must reflect the on-subnet distribution — the model samples from the forced stream, so the probe must too, else in-zone labels drift.

- [ ] **Step 1:** in `stage_generate_code_hf`, replace the `hf_model.generate(..., do_sample=True, temperature, top_p, top_k, num_return_sequences=m, ...)` call: build one `ForcedSeedLogitsProcessor` per prompt with `rollout_indices=list(range(m))`, `base_offsets=[0]*m`, `start_len=prompt_len`, `randomness=<a fixed probe randomness string>`, `hotkey="probe"`, `prompt_idx=r["dataset_index"]`, `checkpoint_hash=<model rev or "probe">`, then `hf_model.generate(input_ids, num_return_sequences=m, **forced_seed_generate_kwargs({"max_new_tokens": max_tokens, "eos_token_id": eos_list, "pad_token_id": pad_id}, proc))`.
- [ ] **Step 2: test** (`tests/test_probe_forced_seed.py`) — import-only smoke that `stage_generate_code_hf` references `ForcedSeedLogitsProcessor` (grep-style: `assert "ForcedSeedLogitsProcessor" in inspect.getsource(stage_generate_code_hf)`).
- [ ] **Step 3:** `python3 -m py_compile scripts/difficulty_probe.py`.
- [ ] **Step 4: checkpoint.** (Runtime validation on GPU = Part D.)

### Task 7: CPU end-to-end seed-consistency test

**Files:** Test `tests/test_forced_seed_e2e.py`

- [ ] **Step 1: test** — a tiny fake LM (a fixed logits matrix per position) generated through `ForcedSeedLogitsProcessor` yields tokens that score `n_match == n_stoch` under `seed_consistency` with the SAME `u_at` inputs — proving an honest forced miner passes the 0.80 floor.
```python
import torch
from reliquary.environment.forced_sampling import u_at, seed_consistency, warp, pick
def test_forced_generation_scores_full_consistency():
    torch.manual_seed(1)
    logits = torch.randn(40, 200)  # 40 positions
    us = [u_at("w", "hk", 9, "ck", 0, t) for t in range(40)]
    toks = [pick(warp(logits[t], t=0.6, top_k=20, top_p=0.95), us[t]) for t in range(40)]
    n_stoch, n_match = seed_consistency(logits, toks, us, t=0.6, top_k=20, top_p=0.95,
                                        stochastic_threshold=0.99)
    assert n_stoch >= 30 and n_match == n_stoch  # ≥ FORCED_SEED_MIN_STOCH_POSITIONS, 100% match
```
- [ ] **Step 2: run → PASS. Checkpoint.**

---

## PART B — Finish v7/BFT (CPU)

### Task 8: Math BFT branch in the probe (was v7-plan Task 6)

**Files:** Modify `scripts/difficulty_probe.py` · Test `tests/test_probe_bft_wire.py`

- [ ] **Step 1:** add a math generation path (or `--env math` in `stage_generate_code_hf`'s sibling) that routes through `bft_assemble_rollouts` when `bft_applicable("openmathinstruct")`, code path stays single-pass. **Combine with Task 6's forced-seed processor** (both apply).
- [ ] **Step 2: test** (fake model + fake tokenizer, reuse `tests/test_bft_assemble.py::_Model`) that the math branch calls `bft_assemble_rollouts` and the code branch does not.
- [ ] **Step 3: run → PASS. Checkpoint.**

### Task 9: Engine BFT-routing lock test (was v7-plan Task 7 hardening)

**Files:** Test `tests/test_engine_bft_routing.py`

- [ ] **Step 1: test** — with a fake model, assert `_generate_m_rollouts` for env `openmathinstruct` uses `phase1_max_new_tokens == BFT_THINKING_BUDGET` and calls `_bft_from_seqs`; for env `opencodeinstruct` it uses `self.max_new_tokens` and does NOT call `_bft_from_seqs`. Locks the math→2-phase / code→mono-phase routing so a future edit can't silently break the carve-out.
- [ ] **Step 2: run → PASS. Checkpoint.**

---

## PART C — Verify low-risk upstream is no-op (CPU)

### Task 10: Confirm all-token-auth + grader-cgroup do not reject an honest miner

**Files:** none (verification) · Doc: append findings to `CLAUDE.md`

- [ ] **Step 1:** read `git show b790e42:reliquary/validator/batcher.py` around `evaluate_all_token_auth_shadow` / `ALL_TOKEN_AUTH_ENFORCE` — confirm the argmax gate rejects only tokens whose argmax-confidence exceeds threshold AND differ from the submitted token (edited tokens). An honest forced-seed miner submits the model's own picks → passes. Record the exact reject condition + margin in `CLAUDE.md`.
- [ ] **Step 2:** read `git show b790e42:reliquary/environment/grader/server.py` diff — confirm `#104` only disables runsc cgroups (sandbox runtime), not the grading result. Our `code_grader_driver.py` uses a lightweight subprocess (no runsc) → grading result unaffected. Record in `CLAUDE.md`.
- [ ] **Step 3: checkpoint.**

---

## PART D — GPU validation (DEFERRED — needs GPU)

### Task 11: Runtime validation (GPU-gated)

- [ ] Load the real 2B (`ReliquaryForge/qwen3.5-2b-reliquary`@`6a8c5637`) via HF; generate M rollouts through the forced-seed processor; compute `seed_consistency` on the generated tokens with the model's own logits → confirm ≥ `FORCED_SEED_CONSISTENCY_FLOOR` (target ~1.0). This is the real proof the port matches.
- [ ] Re-run predictor CODE data generation (now forced-seed) → `analyze` for the first 2B AUC.
- [ ] v7/BFT runtime parity (math-only forced spans, ACCEPTED vs rejects).
- [ ] GRAIL parity: compute a proof on our stack + run validator `verify_commitment_proofs` (now includes seed-consistency) → confirm ACCEPTED.

---

## Self-Review

- **Forced-seed core** (`warp`/`pick`/`u_at`/`seed_consistency`) → Task 2 (byte-exact) ✅
- **Constants** (`FORCED_SEED_*` + protocol version) → Task 1 ✅
- **HF LogitsProcessor glue** → Task 3 (byte-exact) ✅
- **Schema** (`protocol_version`, `SEED_MISMATCH`) → Task 4 ✅
- **Engine wiring** (phase-1 + phase-2 + emit protocol_version) → Task 5 ✅
- **Probe wiring** (predictor data on forced stream) → Task 6 ✅
- **Honest-consistency proof** → Task 7 (CPU) + Task 11 (GPU) ✅
- **Finish v7/BFT** → Tasks 8, 9 ✅
- **Low-risk upstream no-op** → Task 10 ✅
- **GRAIL/runtime** → Task 11 (GPU) ✅

**Ordering note:** Part A is blocking (mining is rejected without it) → do first. Parts B/C are CPU cleanups. Part D waits for GPU (and the vLLM fix is a separate session; forced-seed is HF so it does not depend on vLLM).
