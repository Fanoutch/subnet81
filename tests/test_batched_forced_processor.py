"""Batched forced-seed masking — parity gate vs the per-row reference.

Measured on the 4B (H100, phase 1): the per-row Python loop in vLLM's
AdapterLogitsProcessor.apply() (one force_row call per sequence per step) is the
dominant forced-seed cost — fs_on 1452 vs fs_off 3487 tok/s (2.4x), and a faster
per-row warp did NOT help (warp_fast -9%: the loop/syncs dominate, not the sort).

The fix: ONE vectorized pass per step over all forced rows, using
force_rows_batched (already parity-proven bit-exact vs warp+pick — it powers the
HF path). This test locks the batched mask against the per-row force_row
reference: identical forced token per row, identical mask shape.
"""
from __future__ import annotations

import torch

from reliquary.environment.forced_sampling import u_at
from reliquary.miner.vllm_forced_seed import batched_force_mask, force_row

RND, CKPT = "feedface" * 8, "batched-gate"


def test_batched_mask_is_bit_identical_to_per_row_force_row():
    g = torch.Generator().manual_seed(0)
    vocab = 32000
    # 6-row batch: rows 1,3,4 forced (different rollout/t), rows 0,2,5 untouched.
    logits = torch.randn(6, vocab, generator=g)
    original = logits.clone()
    entries = [
        (1, u_at(RND, 11, CKPT, 0, 5)),
        (3, u_at(RND, 11, CKPT, 3, 17)),
        (4, u_at(RND, 42, CKPT, 7, 0)),
    ]

    # reference: the current per-row path
    expected = original.clone()
    expected[1] = force_row(original[1], RND, 11, CKPT, 0, 5)
    expected[3] = force_row(original[3], RND, 11, CKPT, 3, 17)
    expected[4] = force_row(original[4], RND, 42, CKPT, 7, 0)

    got = batched_force_mask(logits, entries)
    assert torch.equal(got, expected)
    # untouched rows really untouched
    for r in (0, 2, 5):
        assert torch.equal(got[r], original[r])


def test_batched_mask_empty_entries_is_a_no_op():
    logits = torch.randn(3, 1000)
    original = logits.clone()
    assert torch.equal(batched_force_mask(logits, []), original)


def test_batched_mask_many_rows_all_match_reference():
    g = torch.Generator().manual_seed(7)
    vocab = 32000
    n = 24
    logits = torch.randn(n, vocab, generator=g)
    original = logits.clone()
    entries = [(i, u_at(RND, 100 + i, CKPT, i % 8, i * 3)) for i in range(n)]
    got = batched_force_mask(logits, entries)
    for i in range(n):
        ref = force_row(original[i], RND, 100 + i, CKPT, i % 8, i * 3)
        assert torch.equal(got[i], ref), f"row {i} diverges"
