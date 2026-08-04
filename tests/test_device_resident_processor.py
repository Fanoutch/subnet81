"""Study §3.2 — device-resident forced-seed processor, parity gate.

The remaining forced/free gap (0.61 vs their 0.98) maps to our apply() rebuilding
the row-index tensor and the u tensor EVERY step with blocking H2D copies. The
device-resident rework: u values written into a reusable (pinned when CUDA)
staging buffer + non_blocking copy; row indices kept as a persistent device
tensor rebuilt only when the batch composition changes; no per-step allocation.

Parity contract: the new path must produce BIT-IDENTICAL masks to
batched_force_mask (which is itself locked to the per-row reference).
"""
from __future__ import annotations

import torch

from reliquary.environment.forced_sampling import u_at
from reliquary.miner.vllm_forced_seed import (
    ForcedRowsState, batched_force_mask,
)

RND, CKPT = "0badf00d" * 8, "resident-gate"


def _fs(prompt_idx: int, rollout: int, base: int = 0) -> dict:
    return {"randomness": RND, "prompt_idx": prompt_idx,
            "checkpoint_hash": CKPT, "rollout_index": rollout,
            "base_offset": base, "start_len": 0}


def test_state_apply_matches_batched_force_mask():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(6, 32000, generator=g)
    out_a, out_b, out_c = [1, 2], [], [7, 8, 9]
    st = ForcedRowsState()
    st.rebuild({1: (_fs(11, 0), out_a), 3: (_fs(11, 3), out_b),
                4: (_fs(42, 7, base=5), out_c)}, device=logits.device)

    expected = batched_force_mask(logits.clone(), [
        (1, u_at(RND, 11, CKPT, 0, 0 + len(out_a))),
        (3, u_at(RND, 11, CKPT, 3, 0 + len(out_b))),
        (4, u_at(RND, 42, CKPT, 7, 5 + len(out_c))),
    ])
    got = st.apply(logits.clone())
    assert torch.equal(got, expected)


def test_state_tracks_growing_output_between_steps_without_rebuild():
    g = torch.Generator().manual_seed(1)
    out = [5]
    st = ForcedRowsState()
    st.rebuild({0: (_fs(7, 2), out)}, device="cpu")
    l1 = torch.randn(2, 8000, generator=g)
    e1 = batched_force_mask(l1.clone(), [(0, u_at(RND, 7, CKPT, 2, 1))])
    assert torch.equal(st.apply(l1.clone()), e1)
    out.append(6)  # a token was generated; NO rebuild — live list reference
    l2 = torch.randn(2, 8000, generator=g)
    e2 = batched_force_mask(l2.clone(), [(0, u_at(RND, 7, CKPT, 2, 2))])
    assert torch.equal(st.apply(l2.clone()), e2)


def test_state_empty_is_noop_and_buffers_are_reused():
    st = ForcedRowsState()
    st.rebuild({}, device="cpu")
    logits = torch.randn(3, 1000)
    ref = logits.clone()
    assert torch.equal(st.apply(logits), ref)
    # rebuild to a smaller set after a larger one → buffers shrink logically
    out = [1, 2, 3]
    st.rebuild({2: (_fs(1, 1), out)}, device="cpu")
    st.rebuild({1: (_fs(2, 0), out)}, device="cpu")
    l = torch.randn(3, 1000)
    e = batched_force_mask(l.clone(), [(1, u_at(RND, 2, CKPT, 0, 3))])
    assert torch.equal(st.apply(l.clone()), e)
