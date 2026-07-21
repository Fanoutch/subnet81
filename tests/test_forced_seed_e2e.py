"""End-to-end: an honest miner that samples every token through the protocol
forced-seed pick scores FULL seed-consistency AND carries enough stochastic
positions to clear the validator's abstain floor (FORCED_SEED_MIN_STOCH_POSITIONS)
— i.e. it actively PASSES the gate, not merely abstains on thin signal."""
import torch

from reliquary.constants import (
    FORCED_SEED_CONSISTENCY_FLOOR, FORCED_SEED_MIN_STOCH_POSITIONS,
    FORCED_SEED_STOCHASTIC_MAXPROB, T_PROTO, TOP_K_PROTO, TOP_P_PROTO,
)
from reliquary.environment.forced_sampling import pick, seed_consistency, u_at, warp


def test_forced_generation_scores_full_consistency():
    torch.manual_seed(1)
    logits = torch.randn(40, 200)  # 40 completion positions, vocab 200
    us = [u_at("w", 9, "ck", 0, t) for t in range(40)]
    toks = [pick(warp(logits[t], t=T_PROTO, top_k=TOP_K_PROTO, top_p=TOP_P_PROTO), us[t])
            for t in range(40)]
    n_stoch, n_match = seed_consistency(
        logits, toks, us, t=T_PROTO, top_k=TOP_K_PROTO, top_p=TOP_P_PROTO,
        stochastic_threshold=FORCED_SEED_STOCHASTIC_MAXPROB)
    # enough stochastic positions to clear the abstain floor, and every one matches
    assert n_stoch >= FORCED_SEED_MIN_STOCH_POSITIONS
    assert n_match == n_stoch
    # → consistency rate 1.0 ≥ the reject floor
    assert n_match / n_stoch >= FORCED_SEED_CONSISTENCY_FLOOR


def test_non_forced_sampling_fails_the_floor():
    """A miner that ignores the forced pick (samples its own tokens) scores far
    below the floor — the gate's discriminating power."""
    torch.manual_seed(2)
    logits = torch.randn(40, 200)
    us = [u_at("w", 9, "ck", 0, t) for t in range(40)]
    # "cheating": pick argmax every step instead of the forced inverse-CDF token
    toks = [int(warp(logits[t], t=T_PROTO, top_k=TOP_K_PROTO, top_p=TOP_P_PROTO).argmax())
            for t in range(40)]
    n_stoch, n_match = seed_consistency(
        logits, toks, us, t=T_PROTO, top_k=TOP_K_PROTO, top_p=TOP_P_PROTO,
        stochastic_threshold=FORCED_SEED_STOCHASTIC_MAXPROB)
    if n_stoch >= FORCED_SEED_MIN_STOCH_POSITIONS:
        assert n_match / n_stoch < FORCED_SEED_CONSISTENCY_FLOOR
