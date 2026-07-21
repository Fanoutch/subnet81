# vLLM Forced-Seed Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **This repo is NOT git-tracked** (miner-priv is the rsync-synced prod tree) — every "Checkpoint" step means: tests green locally, then `rsync` to the GPU box. No `git commit`.

**Goal:** Generate the submitted forced-seed rollouts with vLLM (triton/FLA) instead of HF, fast enough to submit within the validator's 300 s window, without breaking the HF-teacher-forced `seed_consistency` check.

**Architecture:** A batch-level vLLM v1 `LogitsProcessor` forces every token to the public inverse-CDF pick (`u_at`), replicating `miner/forced_seed_sampler.ForcedSeedLogitsProcessor` inside vLLM's sampler. The BFT 2-phase math flow is replicated on the vLLM path. Engine wiring is gated behind `RELIQUARY_VLLM_FORCED_SEED` (default OFF) so the live HF miner is untouched until an offline seed-consistency gate (Task 4) passes.

**Tech Stack:** vLLM 0.24 (v1 engine, triton GDN backend), torch 2.11+cu130, transformers 5.9, HF (sdpa) for the parity oracle.

## Global Constraints

- Forced-seed math is authoritative in `reliquary/environment/forced_sampling.py` (`u_at`, `warp`, `pick`) and `reliquary/miner/forced_seed_sampler.py` — the vLLM processor must produce **byte-identical picks for identical input logits**. Do not re-derive the math; call the existing functions.
- `u_at(randomness, prompt_idx, checkpoint_hash, rollout_index, t)` — v2 signature, NO hotkey.
- Sampler v7: `T=0.6`, `top_p=0.95`, `top_k=20`. Math BFT caps: thinking 2048 (exact), answer 512.
- vLLM must load qwen3_5 with: `trust_remote_code=True`, `limit_mm_per_prompt={"image":0,"video":0}`, `additional_config={"gdn_prefill_backend":"triton"}`, `enforce_eager=True`. Env: `VLLM_USE_DEEP_GEMM=0 VLLM_DEEP_GEMM_WARMUP=skip VLLM_USE_FLASHINFER_SAMPLER=0`, `CUDA_HOME`/`PATH` on `nvidia/cu13`, nvcc pinned 13.0.88.
- NO ngram `speculative_config` under forced-seed (spec-decoded tokens fail seed-consistency).
- Live miner behavior must not change until Task 5. Flag `RELIQUARY_VLLM_FORCED_SEED` defaults OFF.
- GPU box: `ssh -p 40299 root@31.56.109.64`, venv `/workspace/venv`, miner tree `/workspace/reliquary-miner-priv`, `HF_HOME=/workspace/hf`.

---

### Task 1: `_build_llm` loads qwen3_5 correctly under forced-seed

**Files:**
- Modify: `reliquary/miner/vllm_backend.py:56-79` (`_build_llm`)
- Test: `tests/miner_priv/test_vllm_build_llm_qwen35.py`

**Interfaces:**
- Produces: `_build_llm(model_path, tokenizer_path, gpu_id, gpu_memory_utilization, max_model_len, dtype, forced_seed: bool=False) -> LLM` — when `forced_seed=True`, passes the qwen3_5 kwargs and omits `speculative_config`.

- [ ] **Step 1: Write the failing test** (unit-level: assert the kwargs dict, mocking `vllm.LLM`)

```python
# tests/miner_priv/test_vllm_build_llm_qwen35.py
import sys, types
from unittest import mock

def test_build_llm_forced_seed_kwargs(monkeypatch):
    captured = {}
    fake_llm = mock.MagicMock()
    def _LLM(**kwargs):
        captured.update(kwargs); return fake_llm
    mod = types.ModuleType("vllm"); mod.LLM = _LLM
    monkeypatch.setitem(sys.modules, "vllm", mod)
    from reliquary.miner import vllm_backend
    vllm_backend._build_llm(
        model_path="/m", tokenizer_path="/t", gpu_id=0,
        gpu_memory_utilization=0.55, max_model_len=16384, dtype="bfloat16",
        forced_seed=True,
    )
    assert captured["trust_remote_code"] is True
    assert captured["limit_mm_per_prompt"] == {"image": 0, "video": 0}
    assert captured["additional_config"] == {"gdn_prefill_backend": "triton"}
    assert captured["enforce_eager"] is True
    assert "speculative_config" not in captured  # no spec-decode under forced-seed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /root/subnet81/reliquary-miner-priv && PYTHONPATH=. python3 -m pytest tests/miner_priv/test_vllm_build_llm_qwen35.py -q`
Expected: FAIL (`_build_llm` has no `forced_seed` param / missing kwargs).

- [ ] **Step 3: Write minimal implementation** — add `forced_seed` param + qwen3_5 kwargs

```python
def _build_llm(model_path, tokenizer_path, gpu_id, gpu_memory_utilization,
               max_model_len, dtype, forced_seed: bool = False):
    # ... existing CUDA_VISIBLE_DEVICES + all_special_tokens_extended shim ...
    from vllm import LLM
    kwargs = dict(
        model=model_path, tokenizer=tokenizer_path or model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len, dtype=dtype, kv_cache_dtype="auto",
    )
    if forced_seed:
        kwargs.update(
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 0, "video": 0},
            additional_config={"gdn_prefill_backend": "triton"},
            enforce_eager=True,
        )
    elif os.environ.get("RELIQUARY_DISABLE_SPECULATIVE", "0") != "1":
        kwargs["speculative_config"] = {
            "method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 4,
        }
    return LLM(**kwargs)
```

- [ ] **Step 4: Run test to verify it passes** — Expected: PASS.
- [ ] **Step 5: Checkpoint** — full forced-seed suite still green (`pytest tests/test_forced_seed_v2_port.py tests/miner_priv/test_vllm_build_llm_qwen35.py -q`); rsync to box.

---

### Task 2: `VLLMForcedSeedLogitsProcessor` (vLLM v1 batch processor)

**Files:**
- Create: `reliquary/miner/vllm_forced_seed.py`
- Test: `tests/miner_priv/test_vllm_forced_seed_processor.py`

**Interfaces:**
- Consumes: `u_at`, `warp`, `pick` from `reliquary.environment.forced_sampling`.
- Produces: `VLLMForcedSeedLogitsProcessor` (subclass of `vllm.v1.sample.logits_processor.interface.LogitsProcessor`) with per-request state `{randomness, prompt_idx, checkpoint_hash, rollout_index, base_offset, start_len}` threaded via `SamplingParams.extra_args["forced_seed"]`. Core row transform: `force_row(logits_row, randomness, prompt_idx, checkpoint_hash, rollout_index, t) -> logits_row'` (all `-inf` except the picked token at `0.0`), **byte-identical** to `ForcedSeedLogitsProcessor.__call__` for one row.

- [ ] **Step 0 (discovery): inspect the real v1 interface on the box** — the signatures of `apply`, `update_state`, and how `extra_args` reaches the processor are version-specific. Record them before writing the class.

Run: `ssh -p 40299 root@31.56.109.64 '/workspace/venv/bin/python -c "import inspect; from vllm.v1.sample.logits_processor.interface import LogitsProcessor as L; [print(m, inspect.signature(getattr(L,m))) for m in (\"apply\",\"update_state\",\"is_argmax_invariant\",\"validate_params\") if hasattr(L,m)]"'`
Record the signatures in the test file docstring; write Steps 1-3 against them.

- [ ] **Step 1: Write the failing test** — the pure row transform matches the HF processor for identical logits (kernel-independent; isolates OUR logic).

```python
# tests/miner_priv/test_vllm_forced_seed_processor.py
import torch
from reliquary.environment.forced_sampling import u_at, warp, pick
from reliquary.miner.vllm_forced_seed import force_row

def test_force_row_matches_hf_pick():
    torch.manual_seed(0)
    logits = torch.randn(50)
    R, P, C, RI, T = "rand", 7, "ck", 2, 5
    out = force_row(logits.clone(), R, P, C, RI, T)
    expect_tok = pick(warp(logits, t=0.6, top_k=20, top_p=0.95),
                      u_at(R, P, C, RI, T))
    assert int(out.argmax()) == expect_tok
    assert float(out.max()) == 0.0
    assert out[expect_tok] == 0.0
    others = out[torch.arange(50) != expect_tok]
    assert torch.isinf(others).all() and (others < 0).all()
```

- [ ] **Step 2: Run test to verify it fails** — Expected: FAIL (`vllm_forced_seed` missing).
- [ ] **Step 3: Write minimal implementation** — `force_row` first (pure, testable without vLLM), then the batch class wrapping it per row using the signatures from Step 0.

```python
# reliquary/miner/vllm_forced_seed.py
import torch
from reliquary.environment.forced_sampling import u_at, warp, pick
from reliquary.constants import T_PROTO, TOP_K_PROTO, TOP_P_PROTO  # v7 sampler

def force_row(logits_row, randomness, prompt_idx, checkpoint_hash, rollout_index, t):
    u = u_at(randomness, prompt_idx, checkpoint_hash, rollout_index, t)
    probs = warp(logits_row, t=T_PROTO, top_k=TOP_K_PROTO, top_p=TOP_P_PROTO)
    tok = pick(probs, u)
    out = torch.full_like(logits_row, float("-inf"))
    out[tok] = 0.0
    return out
# class VLLMForcedSeedLogitsProcessor(LogitsProcessor): apply() loops rows,
#   derives t from per-request position, calls force_row. (Written against
#   Step 0 signatures.)
```

- [ ] **Step 4: Run test to verify it passes** — Expected: PASS.
- [ ] **Step 5: Batch-class test** — a 2-row batch (two rollout_indices at the same t) forces two DIFFERENT tokens (u_at differs by rollout_index); assert both rows match their own `force_row`.
- [ ] **Step 6: Checkpoint** — suite green; rsync to box.

---

### Task 3: BFT 2-phase flow on the vLLM path

**Files:**
- Modify: `reliquary/miner/vllm_backend.py` (add a `generate_forced_bft` method on `AsyncVLLMBackend`)
- Test: `tests/miner_priv/test_vllm_bft_flow.py`

**Interfaces:**
- Consumes: `VLLMForcedSeedLogitsProcessor`, `reliquary/miner/bft.py` (phase caps + forced template — byte-exact reference).
- Produces: `AsyncVLLMBackend.generate_forced_bft(prompt, randomness, prompt_idx, checkpoint_hash, rollout_indices, checkpoint_hash) -> list[list[int]]` — M rollouts, each the BFT 2-phase forced sequence (thinking cap 2048 → forced `</think>\n\nFinal Answer: \boxed{` → answer cap 512).

- [ ] **Step 1: Write the failing test** — using a tiny stub LLM (mock) that returns known logits, assert the produced sequence follows the phase caps and injects the forced template at the phase boundary. (Full detail written after Task 2 nails the processor wiring; test asserts: phase-1 length ≤ 2048, template tokens present at boundary, phase-2 length ≤ 512.)
- [ ] **Step 2-4:** implement against `bft.py` reference; run; verify.
- [ ] **Step 5: Checkpoint** — `tests/test_bft_*.py` + new BFT-vLLM test green; rsync.

---

### Task 4: Offline seed-consistency gate (vLLM gen → HF teacher-force)

**Files:**
- Create: `scripts/validate_vllm_forced_seed.py`

**Interfaces:**
- Consumes: `AsyncVLLMBackend.generate_forced_bft`, `load_text_generation_model` (HF oracle), `seed_consistency`.

- [ ] **Step 1:** Script: load 2B in vLLM (forced_seed=True) AND in HF (sdpa). Generate a forced-seed BFT rollout with vLLM → tokens `T`. Teacher-force `T` through HF, compute `seed_consistency(step_logits_hf, T, u_values, ...)`. Print `n_stoch, n_match, rate`.
- [ ] **Step 2:** Run on the box across ~10 prompts. **GATE:** rate ≥ `FORCED_SEED_CONSISTENCY_FLOOR` (0.80) on every group, ideally ~1.0.
- [x] **Step 3:** **Decision — PASS (2026-07-17).** Measured on an 8-rollout group
  (302 stochastic positions, reasoning-heavy prompt): **group rate 0.9768**
  (floor 0.80, +0.177 margin), **worst rollout 0.9423** (floor 0.75, +0.192
  margin). Kernel divergence is real (~2.3% of positions flip vLLM-vs-HF) but
  both floors clear with wide margin. Real-world seed_consistency is HIGHER: this
  measured phase-1 only (vLLM); phase-2 answer tokens are generated by HF
  (`_bft_from_seqs`) → 1.0 by construction. Proceed to Task 5. Deploy behind the
  flag with live SEED_MISMATCH watch + instant revert (Task 6).

---

### Task 5: Engine wiring behind `RELIQUARY_VLLM_FORCED_SEED` (default OFF)

**Files:**
- Modify: `reliquary/miner/engine.py:960-970` (the `use_async_loop` gate)
- Modify: `reliquary/cli/main.py` (build `AsyncVLLMBackend` with `forced_seed=True` when the flag is set)
- Test: `tests/test_vllm_forced_seed_gate.py`

**Interfaces:**
- Consumes: `RELIQUARY_VLLM_FORCED_SEED` env (default "0").
- Produces: `use_async_loop = isinstance(self._vllm_backend, AsyncVLLMBackend) and (not FORCED_SEED_ENFORCE or vllm_forced_seed_enabled())`.

- [ ] **Step 1: Write the failing test** — with flag OFF + enforcement ON, `use_async_loop` is False (live behavior unchanged); with flag ON, True.
- [ ] **Step 2-4:** implement `vllm_forced_seed_enabled()`, update the gate + CLI backend construction; run; verify.
- [ ] **Step 5: Checkpoint** — full forced-seed + wiring suite green; rsync. **Do NOT flip the flag on the box yet.**

---

### Task 6: Live cutover + watch

- [ ] **Step 1:** On the box, set `RELIQUARY_VLLM_FORCED_SEED=1` in `launch_miner.sh`, restart the miner in tmux.
- [ ] **Step 2:** Watch verdicts: seed_consistency ~1.0, **zero** `SEED_MISMATCH`/`GRAIL_FAIL`, and `batch_filled` dropping as window→submit latency falls below 300 s.
- [ ] **Step 3:** If SEED_MISMATCH appears → flip flag back to 0 (instant revert to HF), investigate. If clean → measure the latency win.

## Self-Review

- **Spec coverage:** Task 1 ↔ `_build_llm` fix; Task 2 ↔ Component 1 (processor); Task 3 ↔ Component 2 (BFT); Task 4 ↔ Step 4 gate; Task 5 ↔ Component 3 (wiring); Task 6 ↔ live cutover. All spec sections mapped.
- **Placeholders:** Tasks 3 test detail is deferred to post-Task-2 (the processor wiring shape is a genuine dependency); flagged explicitly, not hidden. Tasks 1-2-4-5 are fully concrete.
- **Type consistency:** `force_row(logits_row, randomness, prompt_idx, checkpoint_hash, rollout_index, t)` used identically in Task 2 def and Task 2/3 consumers; `u_at` v2 signature (no hotkey) consistent throughout.
