"""v3 wire compliance — the 4 blockers found by the port verification.

The live v3 validator (admission.py) verifies the envelope signature WITH
``protocol_version`` + ``generation_profile_id`` bound under the v3 domain, and
``server.py::_protocol_contract_reject_reason`` rejects submissions whose
``generation_profile_id`` != "qwen35-4b-auction-v3" (GENERATION_CONTRACT_MISMATCH).
Missing enum members would crash our strict response parse on exactly the rejects
the live validator now emits.

Parity target = origin/main ``signatures.py::build_envelope_binding``:
  profile present -> SHA256(ENVELOPE_DOMAIN_V3 || len||x for x in
    [hotkey, window8, prompt8, merkle, ckpt, round8, rand, nonce,
     protocol8, profile])
  profile empty  -> exact legacy v1 preimage (unchanged).
"""
from __future__ import annotations

import hashlib

from reliquary import constants
from reliquary.protocol import signatures as sig
from reliquary.protocol.submission import (
    BatchSubmissionRequest, RejectReason, SubmissionPrecommitRequest,
)

_ARGS = dict(
    miner_hotkey="5FakeHotkey",
    window_start=123,
    prompt_idx=456,
    merkle_root="ab" * 32,
    checkpoint_hash="deadbeef",
    drand_round=789,
    randomness="cd" * 32,
    nonce="test-nonce",
)


def _expected_v3_digest(protocol_version: int, profile: str) -> bytes:
    def lp(b: bytes) -> bytes:
        return len(b).to_bytes(4, "big") + b

    parts = [
        _ARGS["miner_hotkey"].encode(),
        int(_ARGS["window_start"]).to_bytes(8, "big"),
        int(_ARGS["prompt_idx"]).to_bytes(8, "big"),
        bytes.fromhex(_ARGS["merkle_root"]),
        _ARGS["checkpoint_hash"].encode(),
        int(_ARGS["drand_round"]).to_bytes(8, "big"),
        bytes.fromhex(_ARGS["randomness"]),
        _ARGS["nonce"].encode(),
        int(protocol_version).to_bytes(8, "big"),
        profile.encode(),
    ]
    h = hashlib.sha256()
    h.update(b"reliquary-envelope-v3")
    for p in parts:
        h.update(lp(p))
    return h.digest()


def test_envelope_binding_v3_matches_upstream_layout():
    got = sig.build_envelope_binding(
        **_ARGS, protocol_version=3,
        generation_profile_id="qwen35-4b-auction-v3",
    )
    assert got == _expected_v3_digest(3, "qwen35-4b-auction-v3")


def test_envelope_binding_without_profile_keeps_exact_legacy_preimage():
    # Empty profile -> byte-identical to the historical legacy call.
    assert sig.build_envelope_binding(**_ARGS) == sig.build_envelope_binding(
        **_ARGS, generation_profile_id="",
    )


def test_batch_submission_request_carries_profile_field():
    # max_length=64 mirror of upstream; default "" so old flows still validate.
    field = BatchSubmissionRequest.model_fields["generation_profile_id"]
    assert field.default == ""


def test_precommit_request_carries_profile_field():
    field = SubmissionPrecommitRequest.model_fields["generation_profile_id"]
    assert field.default == ""


def test_reject_reason_has_v3_members():
    assert RejectReason("protocol_mismatch")
    assert RejectReason("generation_contract_mismatch")
    assert RejectReason("proof_capacity_abort")


def test_v3_binding_with_profile_binds_the_advertised_protocol_version():
    # Regression lock: with the profile present, protocol_version=None must NOT
    # silently bind 0 while the request advertises 3 — the engine passes the
    # advertised version explicitly. None-as-0 and 3 must therefore differ.
    with_v3 = sig.build_envelope_binding(
        **_ARGS, protocol_version=3, generation_profile_id="p",
    )
    with_none = sig.build_envelope_binding(
        **_ARGS, protocol_version=None, generation_profile_id="p",
    )
    assert with_v3 != with_none


def _precommit_args() -> dict:
    return dict(
        miner_hotkey="5FakeHotkey", window_start=1, prompt_idx=2,
        merkle_root="ab" * 32, checkpoint_hash="ck", environment="opencodeinstruct",
        payload_bytes=10, payload_sha256="cd" * 32, drand_round=3,
        randomness="ef" * 32, protocol_version=3, nonce="n",
    )


def test_precommit_binding_v3_appends_profile_under_v3_domain():
    a = _precommit_args()

    def lp(b: bytes) -> bytes:
        return len(b).to_bytes(4, "big") + b

    fields = [
        a["miner_hotkey"].encode(), (1).to_bytes(8, "big"), (2).to_bytes(8, "big"),
        bytes.fromhex(a["merkle_root"]), b"ck", b"opencodeinstruct",
        (10).to_bytes(8, "big"), bytes.fromhex(a["payload_sha256"]),
        (3).to_bytes(8, "big"), bytes.fromhex(a["randomness"]),
        (3).to_bytes(8, "big"), b"n", b"qwen35-4b-auction-v3",
    ]
    h = hashlib.sha256()
    h.update(b"reliquary-upload-precommit-v3")
    for f in fields:
        h.update(lp(f))
    got = sig.build_precommit_binding(
        **a, generation_profile_id="qwen35-4b-auction-v3",
    )
    assert got == h.digest()
    # and the legacy (no-profile) binding is unchanged
    assert sig.build_precommit_binding(**a) == sig.build_precommit_binding(
        **a, generation_profile_id="",
    )


def test_generation_profile_id_defaults_to_live_and_is_env_overridable():
    assert constants.generation_profile_id({}) == "qwen35-4b-auction-v3"
    assert constants.generation_profile_id(
        {"RELIQUARY_GENERATION_PROFILE_ID": "other"}
    ) == "other"
    assert constants.GENERATION_PROFILE_ID == "qwen35-4b-auction-v3"
