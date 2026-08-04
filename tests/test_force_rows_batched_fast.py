"""Lever 1 (étude): restrict the per-step batched sort/cumsum to the top_k
survivors — bit-exact vs force_rows_batched (the parity oracle).

After the batched processor, the remaining forced/free gap (0.67 vs the study's
0.98) is dominated by full-vocab work per step: a [n, 151k] descending sort and
a [n, 151k] cdf cumsum, for ~20 non-zero entries per row. The fast path keeps
every REDUCTION bit-identical (full-vocab softmax denominator, full-vocab
renorm sum — the warp_fast lesson) and reuses torch.topk's fixed-size output
(no nonzero() → no CPU-GPU sync). Rows with ties at the top_k threshold or
inside the top_k values fall back to the reference implementation wholesale —
correctness first, speed on the (overwhelmingly common) tie-free path.
"""
from __future__ import annotations

import torch

from reliquary.environment.forced_sampling import (
    force_rows_batched, force_rows_batched_fast,
)

T, K, P = 0.6, 20, 0.95


def _us(n: int, seed: int) -> list[float]:
    g = torch.Generator().manual_seed(1000 + seed)
    return torch.rand(n, generator=g).tolist()


def test_fast_picks_are_bit_identical_across_seeds():
    for seed in range(60):
        g = torch.Generator().manual_seed(seed)
        n = 4 + seed % 13
        logits = torch.randn(n, 32000, generator=g)
        us = _us(n, seed)
        ref = force_rows_batched(logits, us, t=T, top_k=K, top_p=P)
        fast = force_rows_batched_fast(logits, us, t=T, top_k=K, top_p=P)
        assert torch.equal(ref, fast), f"seed {seed}"


def test_fast_realistic_vocab_and_batch():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(64, 151936, generator=g)
    us = _us(64, 0)
    assert torch.equal(
        force_rows_batched(logits, us, t=T, top_k=K, top_p=P),
        force_rows_batched_fast(logits, us, t=T, top_k=K, top_p=P),
    )


def test_ties_at_the_threshold_fall_back_and_stay_exact():
    # bf16-style collisions: duplicate the kth value across many positions so
    # the kept-set exceeds k — the fast path MUST detect it and fall back.
    g = torch.Generator().manual_seed(3)
    logits = torch.randn(6, 4096, generator=g)
    logits[2, :40] = 1.2345  # 40-way tie spanning the top_k boundary on row 2
    logits[4, 10] = logits[4, 11] = logits[4, 12]  # in-top-k duplicates row 4
    us = _us(6, 3)
    assert torch.equal(
        force_rows_batched(logits, us, t=T, top_k=K, top_p=P),
        force_rows_batched_fast(logits, us, t=T, top_k=K, top_p=P),
    )


def test_extreme_u_values_match_reference_edges():
    g = torch.Generator().manual_seed(9)
    logits = torch.randn(3, 8192, generator=g)
    for u in (0.0, 1e-12, 0.5, 1.0 - 1e-12):
        assert torch.equal(
            force_rows_batched(logits, [u] * 3, t=T, top_k=K, top_p=P),
            force_rows_batched_fast(logits, [u] * 3, t=T, top_k=K, top_p=P),
        )
