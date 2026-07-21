"""vLLM generation backend for the private miner.

Wraps vllm.LLM with a thin synchronous API:
    backend = VLLMBackend(model_path, gpu_id=0)
    completions = backend.generate(prompt_tokens=[1,2,3], n=8, temperature=0.9, ...)
    backend.reload(new_model_path)

The engine calls `generate` from an asyncio coroutine via
`await asyncio.to_thread(backend.generate, ...)` to avoid blocking the loop.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from reliquary.constants import MAX_NEW_TOKENS_PROTOCOL_CAP

logger = logging.getLogger(__name__)


def _build_llm(
    model_path: str,
    gpu_id: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    dtype: str,
    tokenizer_path: Optional[str] = None,
    forced_seed: bool = False,
):
    """Construct a vllm.LLM. Wrapped in a function for mock-ability in tests.

    ``tokenizer_path`` lets us load weights from a trained checkpoint that
    doesn't ship a tokenizer (we strip tokenizer files from the snapshot to
    decouple the miner's transformers version from the validator's). When
    provided, vLLM reads the tokenizer from there instead of ``model_path``.

    ``forced_seed=True`` loads the qwen3.5-2b hybrid-GDN checkpoint the way it
    actually boots on vLLM 0.24 (proven by scripts/vllm_smoke_test.py): text-only
    multimodal, Triton/FLA GDN prefill (the validator's own path — flashinfer's
    GDN JIT crashes on ptxas), eager, remote code. It also DROPS ngram
    speculative decoding: under forced-seed every token must equal the public
    inverse-CDF pick, and spec-decoded tokens would fail the validator's
    seed-consistency check.
    """
    # Pin which CUDA device vLLM uses BEFORE importing it. vLLM picks up
    # CUDA_VISIBLE_DEVICES at import / engine init time.
    # Only set if the shell hasn't already pinned a device: with the dual
    # miner launcher, ``CUDA_VISIBLE_DEVICES=1`` is exported per-process
    # before launching, and overwriting it here would force both miners
    # back onto GPU 0.
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # transformers 5.x removed `Tokenizer.all_special_tokens_extended` but
    # vLLM 0.7.3 still calls it during LLM init. Restore the attribute as
    # a thin alias of `all_special_tokens` before vLLM imports the
    # tokenizer wrapper. Idempotent: only patches if missing.
    from transformers import PreTrainedTokenizerBase
    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        PreTrainedTokenizerBase.all_special_tokens_extended = property(
            lambda self: self.all_special_tokens
        )

    from vllm import LLM
    # vLLM 0.10+ accepts `speculative_config` as a dict; ngram speculation
    # gives ~1.5-2x throughput on math reasoning workloads. Compatible with
    # transformers 5.x as long as the all_special_tokens_extended shim
    # below has been applied before vLLM imports the tokenizer wrapper.
    # Disable via RELIQUARY_DISABLE_SPECULATIVE=1 if the validator's
    # distribution check (q10 of chosen-token probabilities) flags our
    # submissions — spec-decoded sequences can drift slightly off the
    # validator's HF-computed distribution and tank q10.
    kwargs = dict(
        model=model_path,
        tokenizer=tokenizer_path or model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype=dtype,
        kv_cache_dtype="auto",
    )
    if forced_seed:
        # qwen3.5-2b hybrid-GDN bring-up recipe (see reference_vllm_qwen35_2b
        # bringup): text-only, Triton/FLA GDN prefill, eager, remote code. NO
        # speculative_config — spec-decode breaks forced-seed consistency.
        kwargs.update(
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 0, "video": 0},
            additional_config={"gdn_prefill_backend": "triton"},
            enforce_eager=True,
        )
    elif os.environ.get("RELIQUARY_DISABLE_SPECULATIVE", "0") != "1":
        kwargs["speculative_config"] = {
            "method": "ngram",
            "num_speculative_tokens": 5,
            "prompt_lookup_max": 4,
        }
    return LLM(**kwargs)


def _build_sampling_params(
    n: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    stop_token_ids: Optional[list[int]] = None,
):
    """Construct a vllm.SamplingParams. Wrapped for mock-ability in tests.

    The vllm import is local so the module imports cleanly on machines
    where vllm is not installed.
    """
    from vllm import SamplingParams
    # CRITICAL: pass EOS tokens explicitly. vLLM falls back to
    # ``tokenizer.eos_token_id`` only (e.g. Qwen3-4B reports just 151645
    # ``<|im_end|>``; 151643 ``<|endoftext|>`` is the pad_token and absent from
    # the R0mAI fine-tune's (missing) generation_config). The validator's
    # ``has_eos_padding`` check rejects any rollout with an EOS NOT at the last
    # position, so if the model samples an EOS that vLLM doesn't stop on, the
    # rollout becomes EOS-in-middle → ``bad_termination``.
    #
    # ``stop_token_ids`` is now resolved dynamically by the caller (the engine,
    # via ``resolve_eos_token_ids(hf_model, tokenizer)`` — covers Qwen3.5's
    # generation_config + nested text_config EOS set). Falls back to the
    # Qwen3-4B pair {151643, 151645} when the caller passes nothing, so older
    # call paths keep their previous behaviour.
    if not stop_token_ids:
        stop_token_ids = [151643, 151645]
    return SamplingParams(
        n=n,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        stop_token_ids=list(stop_token_ids),
        include_stop_str_in_output=True,
    )


class VLLMBackend:
    """Lazy-initialized vLLM engine on a specific GPU."""

    def __init__(
        self,
        model_path: str,
        gpu_id: int = 0,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 4096,
        dtype: str = "bfloat16",
        tokenizer_path: Optional[str] = None,
        forced_seed: bool = False,
    ) -> None:
        self._model_path = model_path
        self._gpu_id = gpu_id
        self._gpu_memory_utilization = gpu_memory_utilization
        self._max_model_len = max_model_len
        self._dtype = dtype
        self._tokenizer_path = tokenizer_path
        self._forced_seed = forced_seed
        self._llm: Optional[object] = None

    def _ensure_loaded(self) -> None:
        """Lazy-build the vLLM LLM, with retry on ckpt-advance OOM.

        H100 80GB hardening: at ckpt-advance the old vLLM's EngineCore
        subprocess takes 3-10 s to actually release its KV cache + weights
        back to the device. The first attempt can therefore see less free
        VRAM than ``gpu_memory_utilization * total`` requests and raise::

            ValueError: Free memory on device (N/total GiB) on startup is
            less than desired GPU memory utilization ...

        We catch that specific class of init failure (ValueError +
        RuntimeError from vLLM's launch_core_engines wrapper), force gc +
        empty_cache, sleep, and retry. Without this the generator loop
        re-enters _ensure_loaded() in a tight loop, each attempt racing
        the same stale memory, and the miner stays in a permanent
        zero-vLLM state for the rest of the process lifetime.
        """
        if self._llm is not None:
            return
        last_exc: Optional[BaseException] = None
        for attempt in range(5):
            try:
                self._llm = _build_llm(
                    model_path=self._model_path,
                    gpu_id=self._gpu_id,
                    gpu_memory_utilization=self._gpu_memory_utilization,
                    max_model_len=self._max_model_len,
                    dtype=self._dtype,
                    tokenizer_path=self._tokenizer_path,
                    forced_seed=self._forced_seed,
                )
                if attempt > 0:
                    logger.info(
                        "vLLM _build_llm succeeded on retry attempt %d",
                        attempt + 1,
                    )
                return
            except (ValueError, RuntimeError) as exc:
                last_exc = exc
                msg = str(exc)
                # Only retry on the OOM-during-init signature; let other
                # build errors (bad path, bad config, etc.) fail fast.
                is_oom = (
                    "Free memory on device" in msg
                    or "Engine core initialization failed" in msg
                )
                if not is_oom:
                    raise
                logger.warning(
                    "vLLM _build_llm attempt %d/5 hit init-time OOM "
                    "(%s); gc+sleep+retry",
                    attempt + 1,
                    msg.split("\n", 1)[0][:200],
                )
                try:
                    import gc as _gc
                    import time as _time
                    _gc.collect()
                    try:
                        import torch as _torch
                        if _torch.cuda.is_available():
                            _torch.cuda.empty_cache()
                    except Exception:
                        pass
                    # Backoff: 3s, 6s, 12s, 24s. Caps total at ~45s — past
                    # which the stale vLLM is either dead or genuinely stuck.
                    _time.sleep(3.0 * (2 ** attempt))
                except Exception:
                    pass
        # All retries exhausted; surface the original error so the engine
        # loop can fall back to a no-op iteration rather than spin.
        assert last_exc is not None
        raise last_exc

    def generate(
        self,
        prompt_token_ids: list[int],
        n: int = 8,
        temperature: float = 0.9,
        top_p: float = 1.0,
        top_k: int = -1,
        max_tokens: int = MAX_NEW_TOKENS_PROTOCOL_CAP,
        stop_token_ids: Optional[list[int]] = None,
    ) -> list[list[int]]:
        """Sample `n` completions for the given prompt; return token-id lists.

        Returns a list of length `n`; each element is the list of generated
        token IDs (NOT including the prompt). ``stop_token_ids`` should be the
        model's full EOS set (resolved by the engine); falls back to the
        Qwen3-4B pair inside ``_build_sampling_params`` when omitted.
        """
        self._ensure_loaded()
        sampling_params = _build_sampling_params(
            n=n,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            stop_token_ids=stop_token_ids,
        )
        # vLLM 0.10+ no longer accepts ``prompt_token_ids=`` as a kwarg on
        # ``LLM.generate``; pass a TokensPrompt instead. Backwards-compat
        # with older vLLM versions is left to the operator.
        from vllm.inputs import TokensPrompt
        outputs = self._llm.generate(
            [TokensPrompt(prompt_token_ids=prompt_token_ids)],
            sampling_params=sampling_params,
        )
        # outputs is a list of length 1 (we passed one prompt).
        # outputs[0].outputs is a list of n CompletionOutput objects.
        return [list(co.token_ids) for co in outputs[0].outputs]

    def generate_forced_phase1(
        self,
        prompt_token_ids: list[int],
        *,
        randomness: str,
        prompt_idx: int,
        checkpoint_hash: str,
        m_rollouts: int,
        max_tokens: int,
        stop_token_ids: Optional[list[int]] = None,
    ) -> list[list[int]]:
        """BFT phase-1 under forced-seed: ``m_rollouts`` completions, each forced
        onto its own ``rollout_index`` stream by the engine-registered
        VLLMForcedSeedLogitsProcessor. Greedy (``temperature=0``) so the argmax is
        the forced token. One batched ``LLM.generate`` call (continuous batching
        across the M rollouts). Returns generated token lists (prompt excluded);
        phase-2 continues on HF via ``engine._bft_from_seqs``.
        """
        self._ensure_loaded()
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        from reliquary.miner.vllm_forced_seed import (
            FORCED_SEED_EXTRA_KEY, forced_seed_extra_args,
        )
        start_len = len(prompt_token_ids)
        prompts = [TokensPrompt(prompt_token_ids=prompt_token_ids)
                   for _ in range(m_rollouts)]
        sps = [
            SamplingParams(
                n=1, temperature=0.0, max_tokens=max_tokens,
                stop_token_ids=list(stop_token_ids) if stop_token_ids else None,
                extra_args={FORCED_SEED_EXTRA_KEY: forced_seed_extra_args(
                    randomness=randomness, prompt_idx=prompt_idx,
                    checkpoint_hash=checkpoint_hash, rollout_index=r,
                    base_offset=0, start_len=start_len)},
            )
            for r in range(m_rollouts)
        ]
        outputs = self._llm.generate(prompts, sampling_params=sps)
        return [list(out.outputs[0].token_ids) for out in outputs]

    def generate_forced_phase1_multi(
        self,
        prompts_token_ids: list[list[int]],
        *,
        prompt_indices: list[int],
        randomness: str,
        checkpoint_hash: str,
        m_rollouts: int,
        max_tokens: int,
        stop_token_ids: Optional[list[int]] = None,
    ) -> list[list[list[int]]]:
        """Forced-seed phase-1 for MANY prompts in ONE batched ``generate`` call.

        Same contract as ``generate_forced_phase1`` but across prompts: every
        (prompt, rollout) pair becomes its own sequence carrying its own
        ``prompt_idx``/``start_len`` in ``extra_args``, so the engine-registered
        processor forces each stream independently.

        Why this exists: under forced-seed the bake loop used to call
        ``generate_forced_phase1`` once per prompt, so N prompts cost N x ~44s.
        With a 100s collection window that meant a 6-prompt bake (~264s) was
        always flushed stale at the randomness flip and never submitted.
        Batching turns it into roughly one 44s call.

        Returns ``[[rollout_tokens] * m_rollouts] * len(prompts)`` in input order.
        """
        if len(prompts_token_ids) != len(prompt_indices):
            raise ValueError(
                f"prompts_token_ids ({len(prompts_token_ids)}) and prompt_indices "
                f"({len(prompt_indices)}) must have the same length; zip would "
                f"silently mislabel forced-seed streams"
            )
        self._ensure_loaded()
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        from reliquary.miner.vllm_forced_seed import (
            FORCED_SEED_EXTRA_KEY, forced_seed_extra_args,
        )

        prompts = []
        sps = []
        for tokens, prompt_idx in zip(prompts_token_ids, prompt_indices):
            start_len = len(tokens)
            for r in range(m_rollouts):
                prompts.append(TokensPrompt(prompt_token_ids=tokens))
                sps.append(SamplingParams(
                    n=1, temperature=0.0, max_tokens=max_tokens,
                    stop_token_ids=list(stop_token_ids) if stop_token_ids else None,
                    extra_args={FORCED_SEED_EXTRA_KEY: forced_seed_extra_args(
                        randomness=randomness, prompt_idx=prompt_idx,
                        checkpoint_hash=checkpoint_hash, rollout_index=r,
                        base_offset=0, start_len=start_len)},
                ))

        outputs = self._llm.generate(prompts, sampling_params=sps)
        flat = [list(out.outputs[0].token_ids) for out in outputs]
        # regroup: sequences were emitted prompt-major, m_rollouts per prompt
        return [
            flat[i * m_rollouts:(i + 1) * m_rollouts]
            for i in range(len(prompts_token_ids))
        ]

    def generate_multi(
        self,
        prompts_token_ids: list[list[int]],
        n: int = 8,
        temperature: float = 0.9,
        top_p: float = 1.0,
        top_k: int = -1,
        max_tokens: int = MAX_NEW_TOKENS_PROTOCOL_CAP,
        stop_token_ids: Optional[list[int]] = None,
    ) -> list[list[list[int]]]:
        """Generate ``n`` completions for each prompt in ``prompts_token_ids``.

        Issued as a single ``LLM.generate`` call so vLLM's continuous
        batching can interleave the rollouts of all prompts on the same
        GPU step — keeps the SM busy while individual rollouts trickle in
        and out. This is the win on a single big GPU (H200) where one
        prompt × 8 rollouts leaves the device under-utilised between
        cycles.

        Returns a list parallel to ``prompts_token_ids``: index ``i``
        holds the ``n`` token-id lists generated for prompt ``i``.
        """
        self._ensure_loaded()
        sampling_params = _build_sampling_params(
            n=n,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            stop_token_ids=stop_token_ids,
        )
        from vllm.inputs import TokensPrompt
        outputs = self._llm.generate(
            [TokensPrompt(prompt_token_ids=p) for p in prompts_token_ids],
            sampling_params=sampling_params,
        )
        # outputs is parallel to prompts. outputs[i].outputs is the list
        # of n CompletionOutput objects for prompt i.
        return [
            [list(co.token_ids) for co in out.outputs]
            for out in outputs
        ]

    def reload(self, new_model_path: str) -> None:
        """Swap to a new checkpoint. Drops the current LLM and BLOCKS
        until its VRAM is actually released.

        The next call to `generate()` will rebuild against `new_model_path`.

        H100 80GB hardening: a naive ``del self._llm; empty_cache()`` is
        racy — vLLM's EngineCore subprocess shutdown is async (Popen
        signal, OS reaps, CUDA driver returns memory) and takes 3-10 s.
        If the next ``_ensure_loaded()`` fires before the old subprocess
        releases, the new ``LLM()`` init OOMs (only ~10 GiB free while
        it wants ~32 GiB) and the miner enters a crash loop. We poll
        ``torch.cuda.mem_get_info`` and wait until the device has enough
        free memory for the new vLLM to fit (``gpu_memory_utilization``
        × total + 5 GiB safety), or 30 s max — whichever comes first.

        Cost on the happy path: an extra ~3-8 s of synchronous wait at
        ckpt-advance. Worth it vs the indefinite zombie state we get
        without the wait.
        """
        if self._llm is not None:
            # Drop the Python reference so refcount can hit zero and
            # vLLM's __del__ chain (which signals the EngineCore subprocess
            # to terminate) starts.
            self._llm = None
            try:
                import gc as _gc
                import time as _time
                import torch as _torch

                _gc.collect()
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
                    try:
                        _, total_b = _torch.cuda.mem_get_info()
                        total_gib = total_b / (1024 ** 3)
                    except Exception:
                        total_gib = 80.0  # H100 fallback
                    target_free_gib = self._gpu_memory_utilization * total_gib + 5.0

                    waited = 0.0
                    deadline = 30.0
                    while waited < deadline:
                        try:
                            free_b, _ = _torch.cuda.mem_get_info()
                            free_gib = free_b / (1024 ** 3)
                            if free_gib >= target_free_gib:
                                logger.info(
                                    "vllm_backend.reload: %.1f/%.1f GiB free "
                                    "(target %.1f) after %.1fs — proceeding",
                                    free_gib, total_gib, target_free_gib, waited,
                                )
                                break
                        except Exception:
                            break
                        _time.sleep(1.0)
                        waited += 1.0
                        _gc.collect()
                        try:
                            _torch.cuda.empty_cache()
                        except Exception:
                            pass
                    else:
                        # Loop exhausted without break — log and let the
                        # next _ensure_loaded() retry-loop handle it.
                        try:
                            free_b, _ = _torch.cuda.mem_get_info()
                            free_gib = free_b / (1024 ** 3)
                        except Exception:
                            free_gib = -1
                        logger.warning(
                            "vllm_backend.reload: timed out waiting for "
                            "VRAM release (free=%.1f, target=%.1f); "
                            "_ensure_loaded will retry",
                            free_gib, target_free_gib,
                        )
            except Exception:
                # Best-effort cleanup; never raise from reload.
                pass
        self._model_path = new_model_path


# ---------------------------------------------------------------------------
# AsyncLLM backend — feature-flagged, opt-in via RELIQUARY_ASYNC_LLM=1.
#
# vLLM's ``AsyncLLMEngine`` lets us push prompt requests at any time and
# stream completed rollouts as they finish. This unlocks pipeline overlap:
# while one prompt's rollouts are running through HF proof + submit, the
# next prompt is already generating on the GPU. On a single H200 (where
# the sync miner sits idle ~25-35% of the cycle waiting on HF/HTTP), this
# closes most of that gap. Sync ``VLLMBackend`` above is unchanged so
# Targon (vllm 0.7.3 — different async API) keeps working as-is.
# ---------------------------------------------------------------------------


class AsyncVLLMBackend:
    """Async vLLM engine for pipeline-overlapped mining.

    API mirrors ``VLLMBackend`` but ``generate`` is an async coroutine,
    and a streaming variant ``add_request`` returns an async iterator over
    completed ``RequestOutput`` objects.

    Validated against vllm 0.10.2 only. Older vllm (0.7.x) has a
    different ``AsyncLLMEngine`` import path / signature; if you flip the
    feature flag on a 0.7.x box, expect import errors.
    """

    def __init__(
        self,
        model_path: str,
        gpu_id: int = 0,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 16384,
        dtype: str = "bfloat16",
        tokenizer_path: Optional[str] = None,
        forced_seed: bool = False,
    ) -> None:
        self._model_path = model_path
        self._gpu_id = gpu_id
        self._gpu_memory_utilization = gpu_memory_utilization
        self._max_model_len = max_model_len
        self._dtype = dtype
        self._tokenizer_path = tokenizer_path
        self._forced_seed = forced_seed
        self._engine = None
        self._lock = None  # asyncio.Lock, lazy-init on first call
        self._req_counter = 0

    async def _ensure_loaded(self) -> None:
        import asyncio
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if self._engine is not None:
                return

            # Same shim + CUDA pinning logic as the sync path. The shim
            # patches transformers 5.x for vllm 0.10.2 compat and must run
            # before any vllm submodule import.
            if "CUDA_VISIBLE_DEVICES" not in os.environ:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self._gpu_id)

            from transformers import PreTrainedTokenizerBase
            if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
                PreTrainedTokenizerBase.all_special_tokens_extended = property(
                    lambda self: self.all_special_tokens
                )

            from vllm import AsyncEngineArgs, AsyncLLMEngine
            engine_kwargs = dict(
                model=self._model_path,
                tokenizer=self._tokenizer_path or self._model_path,
                gpu_memory_utilization=self._gpu_memory_utilization,
                max_model_len=self._max_model_len,
                dtype=self._dtype,
                kv_cache_dtype="auto",
                disable_log_stats=True,
            )
            if self._forced_seed:
                # qwen3.5-2b hybrid-GDN bring-up recipe + register the forced-seed
                # batch logits processor (per-request payload via extra_args). NO
                # speculative_config — spec-decode breaks seed-consistency.
                from reliquary.miner.vllm_forced_seed import (
                    build_forced_seed_logitsproc_class,
                )
                engine_kwargs.update(
                    trust_remote_code=True,
                    limit_mm_per_prompt={"image": 0, "video": 0},
                    additional_config={"gdn_prefill_backend": "triton"},
                    enforce_eager=True,
                    logits_processors=[build_forced_seed_logitsproc_class()],
                )
            elif os.environ.get("RELIQUARY_DISABLE_SPECULATIVE", "0") != "1":
                engine_kwargs["speculative_config"] = {
                    "method": "ngram",
                    "num_speculative_tokens": 5,
                    "prompt_lookup_max": 4,
                }
            args = AsyncEngineArgs(**engine_kwargs)
            # Mirror sync _VLLMBackend._ensure_loaded: at ckpt-advance the
            # previous EngineCore subprocess takes 3-10 s to release its KV
            # cache. If reload()'s VRAM-poll didn't catch the full release
            # (or wasn't called — first build path), retry on init-time OOM
            # rather than die and leave the miner generator-less.
            last_exc: Optional[BaseException] = None
            for attempt in range(5):
                try:
                    self._engine = AsyncLLMEngine.from_engine_args(args)
                    if attempt > 0:
                        logger.info(
                            "AsyncLLMEngine.from_engine_args succeeded on "
                            "retry attempt %d", attempt + 1,
                        )
                    return
                except (ValueError, RuntimeError) as exc:
                    last_exc = exc
                    msg = str(exc)
                    is_oom = (
                        "Free memory on device" in msg
                        or "Engine core initialization failed" in msg
                    )
                    if not is_oom:
                        raise
                    logger.warning(
                        "AsyncLLMEngine.from_engine_args attempt %d/5 hit "
                        "init-time OOM (%s); gc+sleep+retry",
                        attempt + 1,
                        msg.split("\n", 1)[0][:200],
                    )
                    try:
                        import asyncio as _asyncio
                        import gc as _gc
                        _gc.collect()
                        try:
                            import torch as _torch
                            if _torch.cuda.is_available():
                                _torch.cuda.empty_cache()
                        except Exception:
                            pass
                        await _asyncio.sleep(3.0 * (2 ** attempt))
                    except Exception:
                        pass
            assert last_exc is not None
            raise last_exc

    def _next_request_id(self) -> str:
        self._req_counter += 1
        return f"reliquary-{os.getpid()}-{self._req_counter}"

    async def generate(
        self,
        prompt_token_ids: list[int],
        n: int = 8,
        temperature: float = 0.9,
        top_p: float = 1.0,
        top_k: int = -1,
        max_tokens: int = MAX_NEW_TOKENS_PROTOCOL_CAP,
        stop_token_ids: Optional[list[int]] = None,
    ) -> list[list[int]]:
        """Sample ``n`` completions, await them all, return token lists.

        Workaround for vLLM 0.10.2 AsyncLLMEngine bug: with ``n>1`` and
        concurrent requests, the engine silently drops samples (returns
        partial ``output.outputs`` under KV cache preemption). We submit
        ``n`` independent ``n=1`` requests instead and gather their
        results. vLLM internally still continuous-batches them all.
        """
        await self._ensure_loaded()
        from vllm.inputs import TokensPrompt

        async def _one_sample() -> list[int]:
            sp = _build_sampling_params(
                n=1,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                stop_token_ids=stop_token_ids,
            )
            request_id = self._next_request_id()
            results_iter = self._engine.generate(
                TokensPrompt(prompt_token_ids=prompt_token_ids),
                sampling_params=sp,
                request_id=request_id,
            )
            final_output = None
            async for output in results_iter:
                final_output = output
            if final_output is None or not final_output.outputs:
                raise RuntimeError(
                    f"AsyncLLMEngine yielded no output for {request_id}"
                )
            return list(final_output.outputs[0].token_ids)

        import asyncio
        results = await asyncio.gather(*[_one_sample() for _ in range(n)])
        return results

    async def generate_forced_phase1(
        self,
        prompt_token_ids: list[int],
        *,
        randomness: str,
        prompt_idx: int,
        checkpoint_hash: str,
        m_rollouts: int,
        max_tokens: int,
        stop_token_ids: Optional[list[int]] = None,
    ) -> list[list[int]]:
        """BFT phase-1 on vLLM under forced-seed: ``m_rollouts`` completions,
        each forced onto its own ``rollout_index`` stream by the engine-registered
        ``VLLMForcedSeedLogitsProcessor`` (payload threaded via extra_args).

        Sampling is greedy (``temperature=0``): the processor masks every logit
        to ``-inf`` except the inverse-CDF pick (set to ``0.0``), so the greedy
        argmax IS the forced token. Returns generated token lists (prompt
        excluded); phase-2 (answer) continues on the HF path via
        ``engine._bft_from_seqs``.
        """
        await self._ensure_loaded()
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        from reliquary.miner.vllm_forced_seed import (
            FORCED_SEED_EXTRA_KEY, forced_seed_extra_args,
        )

        start_len = len(prompt_token_ids)

        async def _one_rollout(rollout_index: int) -> list[int]:
            sp = SamplingParams(
                n=1,
                temperature=0.0,  # greedy → picks the forced (0.0) token
                max_tokens=max_tokens,
                stop_token_ids=list(stop_token_ids) if stop_token_ids else None,
                extra_args={
                    FORCED_SEED_EXTRA_KEY: forced_seed_extra_args(
                        randomness=randomness, prompt_idx=prompt_idx,
                        checkpoint_hash=checkpoint_hash,
                        rollout_index=rollout_index,
                        base_offset=0, start_len=start_len,
                    )
                },
            )
            request_id = self._next_request_id()
            results_iter = self._engine.generate(
                TokensPrompt(prompt_token_ids=prompt_token_ids),
                sampling_params=sp,
                request_id=request_id,
            )
            final_output = None
            async for output in results_iter:
                final_output = output
            if final_output is None or not final_output.outputs:
                raise RuntimeError(
                    f"AsyncLLMEngine yielded no output for {request_id}"
                )
            return list(final_output.outputs[0].token_ids)

        import asyncio
        return await asyncio.gather(
            *[_one_rollout(r) for r in range(m_rollouts)]
        )

    async def reload(self, new_model_path: str) -> None:
        """Swap checkpoint. Drains the engine, rebuilds on next ``generate``.

        Mirrors the sync ``_VLLMBackend.reload`` hardening: after dropping
        the engine reference we poll ``torch.cuda.mem_get_info`` until the
        device actually has enough free VRAM for the next ``_ensure_loaded``
        to fit (``gpu_memory_utilization × total + 5 GiB``), or 30 s max.
        Without this, the next ``_ensure_loaded`` races the still-alive
        EngineCore subprocess and OOMs at init.
        """
        if self._engine is not None:
            try:
                await self._engine.shutdown_background_loop()
            except Exception:
                pass
            self._engine = None
            try:
                import asyncio
                import gc as _gc
                import torch as _torch

                _gc.collect()
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
                    try:
                        _, total_b = _torch.cuda.mem_get_info()
                        total_gib = total_b / (1024 ** 3)
                    except Exception:
                        total_gib = 80.0
                    target_free_gib = self._gpu_memory_utilization * total_gib + 5.0

                    waited = 0.0
                    deadline = 30.0
                    while waited < deadline:
                        try:
                            free_b, _ = _torch.cuda.mem_get_info()
                            free_gib = free_b / (1024 ** 3)
                            if free_gib >= target_free_gib:
                                logger.info(
                                    "AsyncVLLMBackend.reload: %.1f/%.1f GiB free "
                                    "(target %.1f) after %.1fs — proceeding",
                                    free_gib, total_gib, target_free_gib, waited,
                                )
                                break
                        except Exception:
                            break
                        await asyncio.sleep(1.0)
                        waited += 1.0
                        _gc.collect()
                        try:
                            _torch.cuda.empty_cache()
                        except Exception:
                            pass
                    else:
                        try:
                            free_b, _ = _torch.cuda.mem_get_info()
                            free_gib = free_b / (1024 ** 3)
                        except Exception:
                            free_gib = -1
                        logger.warning(
                            "AsyncVLLMBackend.reload: timed out waiting for "
                            "VRAM release (%.1f GiB free, target %.1f); "
                            "_ensure_loaded will retry",
                            free_gib, target_free_gib,
                        )
            except Exception:
                pass
        self._model_path = new_model_path
