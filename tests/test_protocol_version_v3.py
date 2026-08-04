"""4B/v3 compliance: bump forced-seed protocol version 2 -> 3.

The live validator runs protocol v3 (4B/auction-v3). The forced-seed domain is
``reliquary-forced-seed-v{PROTOCOL_VERSION}`` — it seeds u_at, so a v2 miner
against a v3 validator produces a DIFFERENT forced token at every position =
100% SEED_MISMATCH (and PROTOCOL_VERSION_MISMATCH on the wire). The forced-seed
ALGORITHM (u_at, warp, pick) and sampling (8/0.6/0.95/20) are byte-identical in
v3 — only the version/domain change. WIRE_V2 stays OFF (v3 enforces the LEGACY
merkle root, which is our WIRE_V2-off path).

Env-overridable (default = live 3) so a rollback is a launch flag, not a code
change.
"""
from __future__ import annotations

from reliquary import constants


def test_protocol_version_defaults_to_live_v3():
    assert constants.protocol_version({}) == 3


def test_protocol_version_env_override_allows_rollback():
    assert constants.protocol_version({"RELIQUARY_PROTOCOL_VERSION": "2"}) == 2


def test_protocol_version_malformed_falls_back_to_live():
    assert constants.protocol_version({"RELIQUARY_PROTOCOL_VERSION": "x"}) == 3


def test_forced_seed_domain_tracks_the_version():
    assert constants.forced_seed_domain(3) == "reliquary-forced-seed-v3"
    assert constants.forced_seed_domain(2) == "reliquary-forced-seed-v2"


def test_shipped_constants_are_v3_by_default():
    # what the miner actually emits with no override → must be v3 (live).
    assert constants.FORCED_SEED_PROTOCOL_VERSION == 3
    assert constants.FORCED_SEED_DOMAIN == "reliquary-forced-seed-v3"
