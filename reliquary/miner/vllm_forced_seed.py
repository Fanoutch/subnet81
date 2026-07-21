"""Forced-seed logits processing for the vLLM generation path.

Under FORCED_SEED_ENFORCE every generated token must equal the public inverse-CDF
pick derived from ``u_at`` (v2, hotkey-free). On the HF path this is
``miner/forced_seed_sampler.ForcedSeedLogitsProcessor``. vLLM 0.24 exposes a
per-request logits processor as a ``Callable[[output_token_ids, logits], logits]``
wrapped by ``AdapterLogitsProcessor`` (which owns all batch bookkeeping), so we
only need:

* ``force_row`` — the pure per-position transform (torch-only, no vLLM), the
  single source of the forcing math, byte-parity with the HF processor for
  identical input logits.
* ``make_forced_seed_request_proc`` — build the per-request closure that derives
  the completion position ``t`` from the tokens generated so far and applies
  ``force_row`` with that request's ``rollout_index``.
* ``build_forced_seed_logitsproc_class`` — factory that imports vLLM lazily and
  returns the ``AdapterLogitsProcessor`` subclass (kept out of module import so
  the pure functions stay testable on a box without vLLM installed).

Per-request params travel on ``SamplingParams.extra_args["forced_seed"]`` as a
dict: ``{randomness, prompt_idx, checkpoint_hash, rollout_index, base_offset,
start_len}``.
"""
from __future__ import annotations

import torch

from reliquary.constants import T_PROTO, TOP_K_PROTO, TOP_P_PROTO
from reliquary.environment.forced_sampling import u_at, warp, pick

FORCED_SEED_EXTRA_KEY = "forced_seed"


def forced_seed_extra_args(*, randomness: str, prompt_idx: int,
                           checkpoint_hash: str, rollout_index: int,
                           base_offset: int, start_len: int) -> dict:
    """The per-request forced-seed payload the backend nests under
    ``SamplingParams.extra_args[FORCED_SEED_EXTRA_KEY]``. Its keys are exactly
    what ``make_forced_seed_request_proc`` / ``new_req_logits_processor`` read
    back — single source of the producer/consumer contract."""
    return {
        "randomness": randomness,
        "prompt_idx": prompt_idx,
        "checkpoint_hash": checkpoint_hash,
        "rollout_index": rollout_index,
        "base_offset": base_offset,
        "start_len": start_len,
    }


def force_row(logits_row: torch.Tensor, randomness: str, prompt_idx: int,
              checkpoint_hash: str, rollout_index: int, t: int) -> torch.Tensor:
    """Return ``logits_row`` masked to force the inverse-CDF pick at position ``t``.

    All entries become ``-inf`` except the forced token, set to ``0.0`` — exactly
    what ``ForcedSeedLogitsProcessor.__call__`` writes for one row, so a greedy
    sampler downstream emits the forced token.
    """
    u = u_at(randomness, prompt_idx, checkpoint_hash, rollout_index, t)
    probs = warp(logits_row, t=T_PROTO, top_k=TOP_K_PROTO, top_p=TOP_P_PROTO)
    tok = pick(probs, u)
    out = torch.full_like(logits_row, float("-inf"))
    out[tok] = 0.0
    return out


def make_forced_seed_request_proc(*, randomness: str, prompt_idx: int,
                                  checkpoint_hash: str, rollout_index: int,
                                  base_offset: int, start_len: int):
    """Build a per-request vLLM logits processor closure.

    Signature matches vLLM's 2-arg ``RequestLogitsProcessor``:
    ``(output_token_ids, logits) -> logits``. The completion position is
    ``t = base_offset + len(output_token_ids)`` — mirrors the HF processor's
    ``t = base_offsets[r] + (input_ids.shape[1] - start_len)`` where the number
    of already-generated tokens equals ``len(output_token_ids)``. ``start_len``
    is accepted for parity/validation with the HF sampler but the count comes
    from vLLM's output-token bookkeeping directly.
    """
    def _proc(output_token_ids, logits: torch.Tensor) -> torch.Tensor:
        t = base_offset + len(output_token_ids)
        return force_row(logits, randomness, prompt_idx, checkpoint_hash,
                         rollout_index, t)

    return _proc


def build_forced_seed_logitsproc_class():
    """Return the ``AdapterLogitsProcessor`` subclass (imports vLLM lazily)."""
    from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

    class VLLMForcedSeedLogitsProcessor(AdapterLogitsProcessor):
        """Batch adapter: one forced-seed closure per request that advertises a
        ``forced_seed`` extra_args payload; requests without it are untouched."""

        def is_argmax_invariant(self) -> bool:
            # Forcing changes which token wins → NOT argmax-invariant.
            return False

        def new_req_logits_processor(self, params):
            extra = getattr(params, "extra_args", None) or {}
            fs = extra.get(FORCED_SEED_EXTRA_KEY)
            if not fs:
                return None
            return make_forced_seed_request_proc(
                randomness=fs["randomness"], prompt_idx=fs["prompt_idx"],
                checkpoint_hash=fs["checkpoint_hash"],
                rollout_index=fs["rollout_index"],
                base_offset=fs.get("base_offset", 0),
                start_len=fs.get("start_len", 0),
            )

    return VLLMForcedSeedLogitsProcessor
