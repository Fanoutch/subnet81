import torch
from reliquary.environment.forced_sampling import warp, pick, u_at, seed_consistency

def test_pick_matches_seed_consistency_roundtrip():
    torch.manual_seed(0)
    logits = torch.randn(5, 100)
    us = [u_at("rand", 3, "abc", 0, t) for t in range(5)]
    toks = [pick(warp(logits[t], t=0.6, top_k=20, top_p=0.95), us[t]) for t in range(5)]
    n_stoch, n_match = seed_consistency(logits, toks, us, t=0.6, top_k=20, top_p=0.95,
                                        stochastic_threshold=0.99)
    assert n_match == n_stoch

def test_u_at_is_deterministic_and_in_unit_interval():
    a = u_at("r", 1, "c", 0, 7)
    assert a == u_at("r", 1, "c", 0, 7) and 0.0 <= a < 1.0
    assert a != u_at("r", 1, "c", 1, 7)
