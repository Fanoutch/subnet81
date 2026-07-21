import torch
from reliquary.miner.forced_seed_sampler import (
    ForcedSeedLogitsProcessor, forced_seed_generate_kwargs, phase2_base_offsets)
from reliquary.environment.forced_sampling import warp, pick, u_at


def test_processor_forces_the_inverse_cdf_pick():
    proc = ForcedSeedLogitsProcessor(randomness="r", hotkey="h", prompt_idx=1,
        checkpoint_hash="c", rollout_indices=[0], base_offsets=[0], start_len=3)
    scores = torch.randn(1, 50)
    input_ids = torch.zeros(1, 3, dtype=torch.long)  # s = 3 - 3 = 0
    out = proc(input_ids, scores)
    expect = pick(warp(scores[0], t=0.6, top_k=20, top_p=0.95), u_at("r",1,"c",0,0))
    assert int(out[0].argmax()) == expect and out[0].max() == 0.0


def test_generate_kwargs_strip_warpers_and_greedy():
    kw = forced_seed_generate_kwargs({"temperature": 0.6, "top_p": 0.95, "top_k": 20}, object())
    assert "temperature" not in kw and kw["do_sample"] is False
    assert kw["repetition_penalty"] == 1.0


def test_phase2_base_offsets():
    assert phase2_base_offsets([5, 3], prompt_length=3) == [2, 0]
