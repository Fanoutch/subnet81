"""Bit-exact parity gate for the top_k-restricted warp optimisation.

`warp()` is SHARED with the validator and determines every forced token — any
deviation = SEED_MISMATCH. The optimisation restricts the O(V log V) sort to the
top_k survivors (V=vocab ~151k, only ~20 non-zero after top_k) while keeping the
softmax denominator and final renorm sum over the FULL vocab (reordering a
reduction changes it at the ULP → the likely cause of the 3 prior parity
failures). This test asserts warp_fast == warp BIT-FOR-BIT.

⚠️ CPU parity here is necessary but NOT sufficient: GPU reduction kernels differ,
so this MUST be re-verified on the actual H100/H200 before shipping.
"""
from __future__ import annotations

import torch

from reliquary.environment import forced_sampling as fs

T, TOP_K, TOP_P = 0.6, 20, 0.95  # protocol v7 sampler


def _logits(vocab: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(vocab, generator=g)


def test_warp_fast_is_bit_exact_vs_reference():
    for seed in range(300):
        logits = _logits(32000, seed)
        ref = fs.warp(logits, T, TOP_K, TOP_P)
        fast = fs.warp_fast(logits, T, TOP_K, TOP_P)
        assert torch.equal(ref, fast), (
            f"seed {seed}: probs differ, max |Δ|={float((ref - fast).abs().max()):.3e}"
        )


def test_forced_pick_is_identical_across_u():
    # The thing that actually matters: the picked token id must match for every u.
    us = [i / 64.0 for i in range(64)]  # sweep the whole [0,1) CDF
    for seed in range(100):
        logits = _logits(32000, seed)
        ref = fs.warp(logits, T, TOP_K, TOP_P)
        fast = fs.warp_fast(logits, T, TOP_K, TOP_P)
        for u in us:
            assert fs.pick(ref, u) == fs.pick(fast, u), f"seed {seed} u={u}"


def test_realistic_vocab_size_stays_bit_exact():
    # 151936 = Qwen3.5 vocab — the size that makes the full-vocab sort expensive.
    logits = _logits(151936, 0)
    ref = fs.warp(logits, T, TOP_K, TOP_P)
    fast = fs.warp_fast(logits, T, TOP_K, TOP_P)
    assert torch.equal(ref, fast)
