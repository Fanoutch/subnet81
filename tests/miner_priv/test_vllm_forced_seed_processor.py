"""Task 2: the vLLM forced-seed row transform must be byte-identical to the HF
ForcedSeedLogitsProcessor for identical input logits. This isolates OUR logic
from the kernel-divergence question (measured separately in Task 4): given the
same logits, vLLM and HF must force the same token.

vLLM v1 discovery (box, 0.24): a per-request logits processor is
`Callable[[output_token_ids, logits], logits]`, wrapped by AdapterLogitsProcessor
which handles all batch bookkeeping. We only implement `force_row` (the pure
transform) + a per-request closure factory + is_argmax_invariant/new_req.
"""
import torch

from reliquary.environment.forced_sampling import u_at, warp, pick
from reliquary.constants import T_PROTO, TOP_K_PROTO, TOP_P_PROTO
from reliquary.miner.vllm_forced_seed import force_row, make_forced_seed_request_proc


def test_force_row_matches_hf_pick():
    torch.manual_seed(0)
    logits = torch.randn(50)
    R, P, C, RI, T = "rand", 7, "ck", 2, 5
    out = force_row(logits.clone(), R, P, C, RI, T)
    expect_tok = pick(warp(logits, t=T_PROTO, top_k=TOP_K_PROTO, top_p=TOP_P_PROTO),
                      u_at(R, P, C, RI, T))
    assert int(out.argmax()) == expect_tok
    assert float(out.max()) == 0.0
    assert out[expect_tok] == 0.0
    others = out[torch.arange(50) != expect_tok]
    assert torch.isinf(others).all() and (others < 0).all()


def test_request_proc_uses_position_and_rollout():
    # The per-request closure derives t = base_offset + len(output_token_ids)
    # and carries its own rollout_index → two rollouts at the same position
    # force DIFFERENT tokens (u_at differs by rollout_index).
    torch.manual_seed(1)
    logits = torch.randn(50)
    common = dict(randomness="w", prompt_idx=9, checkpoint_hash="ck",
                  base_offset=0, start_len=3)
    proc0 = make_forced_seed_request_proc(rollout_index=0, **common)
    proc1 = make_forced_seed_request_proc(rollout_index=1, **common)
    out0 = proc0([10, 20], logits.clone())   # 2 tokens generated → t=2
    out1 = proc1([10, 20], logits.clone())
    t = 2
    exp0 = pick(warp(logits, t=T_PROTO, top_k=TOP_K_PROTO, top_p=TOP_P_PROTO),
                u_at("w", 9, "ck", 0, t))
    exp1 = pick(warp(logits, t=T_PROTO, top_k=TOP_K_PROTO, top_p=TOP_P_PROTO),
                u_at("w", 9, "ck", 1, t))
    assert int(out0.argmax()) == exp0
    assert int(out1.argmax()) == exp1
    assert exp0 != exp1  # rollout_index changes the forced stream


def test_extra_args_payload_roundtrips_through_new_req_proc():
    # Producer/consumer contract: the dict the backend puts on
    # SamplingParams.extra_args["forced_seed"] must be exactly what the
    # processor's new_req path reads back. Guards against key drift between
    # generate_forced_phase1 and VLLMForcedSeedLogitsProcessor.
    import torch
    from reliquary.miner.vllm_forced_seed import (
        forced_seed_extra_args, make_forced_seed_request_proc,
        FORCED_SEED_EXTRA_KEY,
    )
    payload = forced_seed_extra_args(
        randomness="w", prompt_idx=9, checkpoint_hash="ck",
        rollout_index=3, base_offset=0, start_len=5,
    )
    assert set(payload) == {"randomness", "prompt_idx", "checkpoint_hash",
                            "rollout_index", "base_offset", "start_len"}
    # a proc built from the payload must equal one built from explicit kwargs
    proc_from_payload = make_forced_seed_request_proc(**payload)
    proc_explicit = make_forced_seed_request_proc(
        randomness="w", prompt_idx=9, checkpoint_hash="ck",
        rollout_index=3, base_offset=0, start_len=5)
    logits = torch.randn(40)
    assert int(proc_from_payload([1, 2], logits.clone()).argmax()) == \
           int(proc_explicit([1, 2], logits.clone()).argmax())
    # and the key the processor looks up matches what the producer would nest under
    assert FORCED_SEED_EXTRA_KEY == "forced_seed"
