"""Wiring helpers that connect the engine's generation paths to the (already
tested) bft_assemble_rollouts core: env-gate, phase-1 token cap, and the
variable-length-completions → padded-tensor adapter."""
import torch

from reliquary.constants import BFT_THINKING_BUDGET
from reliquary.miner.bft import (
    bft_applicable,
    bft_rollouts_from_completions,
    phase1_max_new_tokens,
    rollout_metadata,
)

EOS = {2}
CLOSE = {7}
FORCE = [7, 8]


class _Model:
    device = "cpu"

    def generate(self, rows, attention_mask=None, max_new_tokens=0, **kw):
        # append a boxed answer then EOS to every primed row
        return torch.tensor([r.tolist() + [50, 2] for r in rows])


def test_bft_applicable_math_only():
    # BFT carve-out is scoped to the math env (validator ca3ac67); single-env
    # fallback (None) is treated as math, code is excluded.
    assert bft_applicable("openmathinstruct") is True
    assert bft_applicable(None) is True
    assert bft_applicable("opencodeinstruct") is False


def test_phase1_budget_is_exactly_thinking_budget_for_math():
    # The validator pins the FORCE span at prompt_len + BFT_THINKING_BUDGET
    # EXACTLY (TOKEN_TAMPERED otherwise), so phase-1 must be 2048 regardless of
    # the miner's configured max_new_tokens — even a smaller value (e.g. the
    # vllm_backend 1500 default) must NOT shorten it.
    assert phase1_max_new_tokens(8192, "openmathinstruct") == BFT_THINKING_BUDGET
    assert phase1_max_new_tokens(1000, "openmathinstruct") == BFT_THINKING_BUDGET
    assert phase1_max_new_tokens(8192, "opencodeinstruct") == 8192  # uncapped (non-BFT)


def test_bft_rollouts_from_completions_pads_variable_lengths():
    # Variable-length phase-1 completions (each = prompt + gen), as vLLM emits.
    # INVARIANT: non-EOS completions are at the batch max length (they hit the
    # cap), so only the EOS-finished row carries trailing pad — which must be
    # trimmed at its real first EOS, not corrupted by the padding.
    prompt = [1, 1, 1]
    completions = [
        [1, 1, 1, 9, 2],        # row0: EOS(2), shorter → will be padded
        [1, 1, 1, 7, 9, 9],     # row1: </think>(7) no EOS, full width
        [1, 1, 1, 9, 9, 9],     # row2: neither, full width
    ]
    out = bft_rollouts_from_completions(
        completions, prompt, model=_Model(),
        think_close_ids=CLOSE, force_ids=FORCE, eos_ids=EOS,
        answer_budget=4, randomness="rand", hotkey="hk", prompt_idx=0,
        checkpoint_hash="abc", gen_kwargs={},
    )
    # row0: EOS-finished, trailing pad trimmed at real EOS, not forced
    assert out[0]["forced"] is False
    assert out[0]["tokens"] == [1, 1, 1, 9, 2]
    assert "force_span" not in out[0]
    # row1: natural </think> close → phase-2 answer, NO force injected
    assert out[1]["forced"] is False and "force_span" not in out[1]
    # row2: neither → force injected, span length == len(FORCE)
    assert out[2]["forced"] is True
    assert out[2]["force_span"][1] - out[2]["force_span"][0] == len(FORCE)


def test_rollout_metadata_carries_forced_span():
    # the commit metadata the validator reads must surface forced + span
    gen = {"tokens": [1, 1, 1, 7, 8, 50, 2], "prompt_length": 3,
           "forced": True, "force_span": (3, 5)}
    md = rollout_metadata(gen, [0.1, 0.2, 0.3, 0.4])
    assert md["forced"] is True
    assert md["force_span"] == [3, 5]
    assert md["completion_length"] == 4
    assert md["token_logprobs"] == [0.1, 0.2, 0.3, 0.4]


def test_rollout_metadata_defaults_unforced():
    gen = {"tokens": [1, 1, 1, 9, 2], "prompt_length": 3}
    md = rollout_metadata(gen, [0.1, 0.2])
    assert md["forced"] is False and md["force_span"] is None
