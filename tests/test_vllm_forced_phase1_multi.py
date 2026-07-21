"""Batch phase-1 across PROMPTS, not just across rollouts.

Today `_pre_bake_batch` loops prompt-by-prompt under forced-seed, so a bake of
N prompts costs N x ~44s. Measured 2026-07-21: 6 prompts = ~264s against a 100s
collection window, so the pool is ALWAYS flushed stale at the randomness flip
and nothing is ever submitted.

vLLM already carries forced-seed state per sequence via SamplingParams.extra_args
(that is how `generate_forced_phase1` forces M rollouts in ONE call). Nothing
prevents putting N prompts x M rollouts in that same call — each sequence just
needs its own prompt_idx and start_len. That turns N x 44s into roughly one 44s
batch.

These tests pin the CONSTRUCTION contract (one call, correct per-sequence args),
not vLLM's behaviour.
"""

from __future__ import annotations

import pytest


class _FakeOutput:
    def __init__(self, token_ids):
        self.outputs = [type("CO", (), {"token_ids": token_ids})()]


class _FakeLLM:
    """Records how generate() was called and returns one output per sequence."""

    def __init__(self):
        self.calls = []

    def generate(self, prompts, sampling_params=None, **kw):
        self.calls.append({"prompts": prompts, "sampling_params": sampling_params})
        # echo a distinct token list per sequence so ordering is checkable
        return [_FakeOutput([100 + i]) for i in range(len(prompts))]


def _backend(fake):
    from reliquary.miner.vllm_backend import VLLMBackend

    b = VLLMBackend.__new__(VLLMBackend)   # bypass __init__ (no GPU here)
    b._llm = fake
    b._ensure_loaded = lambda: None
    return b


PROMPTS = [[1, 2, 3], [4, 5, 6, 7], [8, 9]]   # deliberately different lengths
IDXS = [11, 22, 33]
M = 4


def _run(fake):
    return _backend(fake).generate_forced_phase1_multi(
        prompts_token_ids=PROMPTS,
        prompt_indices=IDXS,
        randomness="ab" * 32,
        checkpoint_hash="ckpt-hash",
        m_rollouts=M,
        max_tokens=2048,
    )


def test_all_prompts_and_rollouts_go_in_a_single_generate_call():
    """The whole point: one batched call instead of one call per prompt."""
    fake = _FakeLLM()
    _run(fake)
    assert len(fake.calls) == 1
    assert len(fake.calls[0]["prompts"]) == len(PROMPTS) * M


def test_each_sequence_carries_its_own_prompt_idx_and_rollout_index():
    """A shared prompt_idx would make every sequence force the WRONG stream and
    fail the validator's seed-consistency check."""
    from reliquary.miner.vllm_forced_seed import FORCED_SEED_EXTRA_KEY

    fake = _FakeLLM()
    _run(fake)
    seen = []
    for sp in fake.calls[0]["sampling_params"]:
        args = sp.extra_args[FORCED_SEED_EXTRA_KEY]
        seen.append((args["prompt_idx"], args["rollout_index"]))
    assert sorted(seen) == sorted(
        [(idx, r) for idx in IDXS for r in range(M)]
    )


def test_start_len_matches_each_prompts_own_length():
    """start_len pins where the forced stream begins; using one prompt's length
    for another shifts every u_at position -> TOKEN_TAMPERED."""
    from reliquary.miner.vllm_forced_seed import FORCED_SEED_EXTRA_KEY

    fake = _FakeLLM()
    _run(fake)
    by_idx = {}
    for sp in fake.calls[0]["sampling_params"]:
        args = sp.extra_args[FORCED_SEED_EXTRA_KEY]
        by_idx.setdefault(args["prompt_idx"], set()).add(args["start_len"])
    for idx, prompt in zip(IDXS, PROMPTS):
        assert by_idx[idx] == {len(prompt)}


def test_results_are_grouped_back_per_prompt_in_input_order():
    """Caller expects [[rollouts of prompt0], [rollouts of prompt1], ...]."""
    fake = _FakeLLM()
    out = _run(fake)
    assert len(out) == len(PROMPTS)
    assert all(len(rolls) == M for rolls in out)
    # sequences were emitted in order, so prompt0 owns the first M
    assert out[0] == [[100], [101], [102], [103]]


def test_generation_is_greedy_so_the_forced_token_is_the_argmax():
    """temperature must stay 0 — sampling would break the forced pick."""
    fake = _FakeLLM()
    _run(fake)
    assert all(sp.temperature == 0.0 for sp in fake.calls[0]["sampling_params"])


def test_mismatched_prompts_and_indices_is_rejected():
    """Silent zip truncation would mislabel every stream."""
    fake = _FakeLLM()
    with pytest.raises(ValueError):
        _backend(fake).generate_forced_phase1_multi(
            prompts_token_ids=PROMPTS,
            prompt_indices=IDXS[:2],
            randomness="ab" * 32,
            checkpoint_hash="h",
            m_rollouts=M,
            max_tokens=128,
        )
