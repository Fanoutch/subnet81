"""Parity tests for the forced-seed v2 port (upstream difficulty-auction-v2,
validator 6f32673 / PR #140).

v2 drops the hotkey from the forced-seed derivation `u_at` (kills multi-hotkey
variance farming: the forced group is identical for every miner in the window),
keys the derivation on a new domain so v1/v2 streams never collide, and bumps
FORCED_SEED_PROTOCOL_VERSION to 2 (the validator hard-rejects any other version
pre-queue under FORCED_SEED_ENFORCE + pinned checkpoint -> SEED_MISMATCH).

Golden values were computed from the upstream 6f32673 `u_at` implementation.
"""
from reliquary.environment.forced_sampling import u_at
from reliquary import constants


def test_u_at_v2_matches_upstream_golden():
    # u_at(randomness, prompt_idx, checkpoint_hash, rollout_index, t) — NO hotkey.
    assert u_at("r", 1, "c", 0, 7) == 0.5149404366242722


def test_u_at_v2_second_golden():
    assert u_at("w", 9, "ck", 0, 0) == 0.05883567744682221


def test_u_at_v2_signature_has_no_hotkey():
    # Exactly 5 positional params: randomness, prompt_idx, checkpoint_hash,
    # rollout_index, t. A stray 6th arg (a leftover hotkey) must be rejected.
    import inspect

    params = list(inspect.signature(u_at).parameters)
    assert params == ["randomness", "prompt_idx", "checkpoint_hash",
                      "rollout_index", "t"]


def test_forced_seed_domain_is_v2():
    assert constants.FORCED_SEED_DOMAIN == "reliquary-forced-seed-v2"


def test_forced_seed_protocol_version_is_2():
    assert constants.FORCED_SEED_PROTOCOL_VERSION == 2
