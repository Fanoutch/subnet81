"""Wiring gate: force_row must pick the IDENTICAL token via warp_fast.

force_row is the per-token hot path of the vLLM forced-seed processor — the
measured 2.4x throughput tax on the 4B (fs_on 1452 vs fs_off 3487 tok/s, H100).
Swapping warp -> warp_fast shrinks the O(V log V) full-vocab sort to the ~top_k
survivors. warp_fast is proven bit-exact vs warp (test_warp_fast_parity), so the
forced token CANNOT change — this test locks the wiring end-to-end against the
plain warp+pick reference.
"""
from __future__ import annotations

import torch

from reliquary.constants import T_PROTO, TOP_K_PROTO, TOP_P_PROTO
from reliquary.environment.forced_sampling import pick, u_at, warp
from reliquary.miner.vllm_forced_seed import force_row


def test_force_row_matches_plain_warp_pick_reference():
    rnd, ckpt = "deadbeef" * 8, "test-hash"
    for seed in range(50):
        g = torch.Generator().manual_seed(seed)
        logits = torch.randn(32000, generator=g)
        # reference: the validator-side computation, plain warp
        u = u_at(rnd, 7, ckpt, seed % 8, seed)
        ref_tok = pick(warp(logits, t=T_PROTO, top_k=TOP_K_PROTO,
                            top_p=TOP_P_PROTO), u)
        out = force_row(logits, rnd, 7, ckpt, seed % 8, seed)
        forced_tok = int(torch.argmax(out).item())
        assert forced_tok == ref_tok, f"seed {seed}: {forced_tok} != {ref_tok}"
        # masked-row contract: exactly one 0.0, everything else -inf
        assert float(out[forced_tok]) == 0.0
        assert torch.isinf(out).sum().item() == out.numel() - 1
