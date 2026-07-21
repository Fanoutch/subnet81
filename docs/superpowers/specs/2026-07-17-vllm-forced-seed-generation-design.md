# vLLM forced-seed generation — design

**Date**: 2026-07-17
**Status**: approved (build-direct; feasibility gate folded into Step 4 validation)

## Problem

Under `FORCED_SEED_ENFORCE` the miner generates rollouts on the **HF sync loop**
(`engine._generator_loop`), because vLLM's fast path cannot apply the per-token
forced-seed pick. HF generation is slow: producing one valid (non-unanimous,
in-range) rollout group takes **~6.5 min**, but the validator's collection
window is **300 s**. Result: submissions arrive after the batch seals →
`batch_filled` (the current dominant reject). We are systematically too slow.

Faster generation lets us (a) hit the 300 s window and (b) scan more prompts to
find the rare **hard** ones the auction pays for (`std·(1-mean)` of the reward
vector; the 2B solves most prompts unanimously → `out_of_zone` skips).

## Goal

Generate the submitted forced-seed rollouts with **vLLM** instead of HF, fast
enough to submit within 300 s, **without** breaking the validator's
`seed_consistency` check (floor 0.80 group / 0.75 rollout), which teacher-forces
in **HF**.

## Constraint that shapes everything

The submitted tokens must match the inverse-CDF pick the validator recomputes in
HF. vLLM (triton/FLA kernels) and the validator's HF (sdpa) produce numerically
close but non-identical logits. At "stochastic" positions (`max_prob < 0.99`) a
pick can flip if the logit delta moves the warped CDF across `u`. Whether this
stays above the 0.80 floor is **unknown until measured** — hence the Step 4 gate
before any live cutover.

## Architecture

### vLLM API reality (0.24, v1)

`SamplingParams` has **no** per-request `logits_processors` field. vLLM v1 uses a
**batch-level** `vllm.v1.sample.logits_processor.interface.LogitsProcessor`
(`apply(logits)`, `update_state(...)`, `is_argmax_invariant`, `validate_params`),
registered with the engine, applied across the whole decode batch in the sampler.

### Component 1 — `VLLMForcedSeedLogitsProcessor` (batch v1)

A custom v1 batch LogitsProcessor. Per request in the batch it must know:
`randomness`, `prompt_idx`, `checkpoint_hash`, `rollout_index`, `base_offset`,
`start_len`. It derives position `t` per row from the row's current sequence
length. In `apply(logits)`, for each row: `u = u_at(randomness, prompt_idx,
checkpoint_hash, rollout_index, t)`, `probs = warp(row_logits, T, top_k, top_p)`,
forced token = `pick(probs, u)`; set the row's logits to `-inf` except the forced
token (mirrors `miner/forced_seed_sampler.ForcedSeedLogitsProcessor`, which stays
the source of truth for the math). `is_argmax_invariant = False`.
Per-request params are threaded via `SamplingParams.extra_args` and tracked in
`update_state` as requests join/leave the batch.

### Component 2 — BFT 2-phase flow in vLLM

The math BFT flow (thinking cap 2048 → force `</think>\n\nFinal Answer:
\boxed{` → answer cap 512, sampler v7 T=0.6/top_p=0.95/top_k=20) must be
replicated on the vLLM path. Phase boundaries and the forced template injection
are the delicate part. `reliquary/miner/bft.py` is the byte-exact reference.

### Component 3 — engine wiring

Today `use_async_loop = isinstance(self._vllm_backend, AsyncVLLMBackend) and not
FORCED_SEED_ENFORCE`. New gate: allow the vLLM async loop under enforcement
**when** the forced-seed processor is wired and a flag
(`RELIQUARY_VLLM_FORCED_SEED=1`) is set. Default OFF → live miner unchanged.
`vllm_backend._build_llm` must also load qwen3_5 correctly (`trust_remote_code`,
`limit_mm_per_prompt={"image":0,"video":0}`, `additional_config={"gdn_prefill_
backend":"triton"}`, `enforce_eager`; drop the ngram `speculative_config` under
forced-seed — spec-decode tokens would fail seed-consistency).

## Steps (build order)

1. `_build_llm` fixes so vLLM loads qwen3_5 on this box (proven recipe from the
   smoke test).
2. `VLLMForcedSeedLogitsProcessor` (v1 batch) + unit tests vs the HF
   `ForcedSeedLogitsProcessor` on identical logits (must be byte-identical picks
   for identical logits input — isolates our logic from the kernel-divergence
   question).
3. BFT 2-phase replication on the vLLM path + tests.
4. **Validation gate (offline, before any live cutover)**: generate a forced-seed
   rollout with vLLM, teacher-force it through HF, compute `seed_consistency`.
   PASS (≥ floor, ideally ~1.0) → proceed to wiring. FAIL → approach is dead;
   fall back to the difficulty predictor (no numerical risk).
5. Engine wiring behind `RELIQUARY_VLLM_FORCED_SEED` (default OFF).
6. Live cutover: flip the flag on the GPU box, watch verdicts (seed_consistency
   ~1.0, zero SEED_MISMATCH, batch_filled drops as latency falls).

## Safety property

The HF miner keeps running throughout. The live hotkey switches to the vLLM path
only after Step 4 passes and Step 5 is tested. Nothing touches live behavior
until then (flag default OFF).

## Risks / open questions

- **Kernel divergence** (central): vLLM triton/FLA vs validator HF sdpa → picks
  may flip at stochastic positions. Quantified only at Step 4.
- **Prefill vs decode**: Step 4 teacher-forces (prefill); real generation decodes.
  Slight numeric difference possible; Step 6 live watch is the true confirmation.
- **BFT phase transitions** in a batched continuous-generation setting are the
  hardest engineering part; may need per-request phase state in the processor.
- **GPU memory**: vLLM (0.55 util) + HF proof model co-resident on the 80 GB
  H100 — fits (2B), but verify no OOM at reload.
