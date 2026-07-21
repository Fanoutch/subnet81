"""Wire-v2 port (upstream agent/wire-v2-cutover), GATED by RELIQUARY_WIRE_V2.

Default OFF = today's live wire v1, byte-identical behaviour. At the validator
cutover, flip RELIQUARY_WIRE_V2=1: protocol_version=2, canonical Merkle root
(reliquary.protocol.merkle), and the version bound into the v2 envelope domain.
"""
import hashlib
import json
import subprocess

import pytest

from reliquary.protocol.merkle import compute_rollouts_merkle_root
from reliquary.protocol.signatures import build_envelope_binding
from reliquary.protocol.submission import RejectReason

UPSTREAM_REPO = "/root/subnet81/reliquary"
UPSTREAM_REF = "origin/agent/wire-v2-cutover"

_ROLLOUTS = [
    {"tokens": [1, 2, 3], "reward": 1.0, "env_name": "openmathinstruct",
     "commit": {"proof_version": "v7", "b": 1, "a": [2, 3]}},
    {"tokens": [4, 5], "reward": 0.0, "env_name": "openmathinstruct",
     "commit": {"proof_version": "v7", "z": None}},
    {"tokens": [6], "reward": 1.0, "env_name": "openmathinstruct",
     "commit": {"nested": {"y": 2, "x": 1}}},
]


# ---- canonical merkle: executable byte-parity vs upstream -------------------

def test_canonical_merkle_matches_upstream_execution():
    src = subprocess.run(
        ["git", "-C", UPSTREAM_REPO, "show",
         f"{UPSTREAM_REF}:reliquary/protocol/merkle.py"],
        capture_output=True, text=True, check=True,
    ).stdout
    ns: dict = {}
    exec(compile(src, "upstream_merkle.py", "exec"), ns)
    upstream_root = ns["compute_rollouts_merkle_root"](_ROLLOUTS)
    assert compute_rollouts_merkle_root(_ROLLOUTS) == upstream_root
    assert len(upstream_root) == 64


def test_canonical_merkle_binds_env_name_and_order():
    base = compute_rollouts_merkle_root(_ROLLOUTS)
    swapped = compute_rollouts_merkle_root([_ROLLOUTS[1], _ROLLOUTS[0], _ROLLOUTS[2]])
    assert base != swapped
    other_env = [dict(_ROLLOUTS[0], env_name="opencodeinstruct")] + _ROLLOUTS[1:]
    assert compute_rollouts_merkle_root(other_env) != base


# ---- gate --------------------------------------------------------------------

def test_wire_v2_disabled_by_default(monkeypatch):
    from reliquary.miner.engine import wire_v2_enabled
    monkeypatch.delenv("RELIQUARY_WIRE_V2", raising=False)
    assert wire_v2_enabled() is False
    monkeypatch.setenv("RELIQUARY_WIRE_V2", "1")
    assert wire_v2_enabled() is True


def test_submission_root_and_version_follow_gate(monkeypatch):
    from reliquary.miner import engine as engine_mod

    class _R:
        def __init__(self, d):
            self.tokens, self.reward = d["tokens"], d["reward"]
            self.env_name, self.commit = d["env_name"], d["commit"]

    subs = [_R(d) for d in _ROLLOUTS]
    # forced-seed v2 port DECOUPLES protocol_version from the wire-v2 flag: the
    # live validator (6f32673) wants protocol_version=2 WITH the legacy merkle
    # root, so wire_protocol_version() == 2 even with the flag OFF (it reflects
    # FORCED_SEED_PROTOCOL_VERSION). The flag now only switches merkle legacy
    # -> canonical (the still-unmerged wire-v2 cutover), which we keep OFF.
    monkeypatch.delenv("RELIQUARY_WIRE_V2", raising=False)
    assert engine_mod.submission_merkle_root(subs) == engine_mod._compute_merkle_root(subs)
    assert engine_mod.wire_protocol_version() == 2
    monkeypatch.setenv("RELIQUARY_WIRE_V2", "1")
    assert engine_mod.submission_merkle_root(subs) == compute_rollouts_merkle_root(subs)
    assert engine_mod.wire_protocol_version() == 2


# ---- envelope binding ---------------------------------------------------------

_KW = dict(miner_hotkey="hk", window_start=7, prompt_idx=42,
           merkle_root="ab" * 32, checkpoint_hash="ckpt", drand_round=999,
           randomness="cd" * 32, nonce="n1")


def _manual_binding(domain: bytes, with_version: int | None) -> bytes:
    def lp(b):
        return len(b).to_bytes(4, "big") + b
    parts = [
        b"hk",
        (7).to_bytes(8, "big"), (42).to_bytes(8, "big"),
        bytes.fromhex("ab" * 32), b"ckpt", (999).to_bytes(8, "big"),
    ]
    if with_version is not None:
        parts.append(int(with_version).to_bytes(8, "big"))
    parts += [bytes.fromhex("cd" * 32), b"n1"]
    h = hashlib.sha256()
    h.update(domain)
    for p in parts:
        h.update(lp(p))
    return h.digest()


def test_envelope_v1_preimage_unchanged():
    # No protocol_version → the EXACT pre-port v1 bytes (live wire today).
    assert build_envelope_binding(**_KW) == _manual_binding(
        b"reliquary-envelope-v1", None)


def test_envelope_v2_binds_version_between_round_and_randomness():
    assert build_envelope_binding(**_KW, protocol_version=2) == _manual_binding(
        b"reliquary-envelope-v2", 2)


# ---- schema safety ------------------------------------------------------------

def test_protocol_version_mismatch_reject_present():
    assert RejectReason("protocol_version_mismatch")


def test_rollouts_must_share_one_env():
    from reliquary.protocol.submission import (
        BatchSubmissionRequest, RolloutSubmission,
    )
    common = dict(reward=1.0, commit={"c": 1}, tokens=[1, 2])
    rollouts = [RolloutSubmission(**common, env_name="openmathinstruct")
                for _ in range(7)]
    rollouts.append(RolloutSubmission(**common, env_name="opencodeinstruct"))
    with pytest.raises(Exception, match="share one env_name"):
        BatchSubmissionRequest(
            miner_hotkey="hk", prompt_idx=0, window_start=0,
            merkle_root="ab" * 32, rollouts=rollouts, checkpoint_hash="",
        )
