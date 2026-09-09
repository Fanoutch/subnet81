"""Invariance du WIRE v5 pendant le port v6 (rapport A §5).

Les octets figés ci-dessous ont été générés sur le worktree AVANT toute
modification de code du port v6 (`scratchpad/v6port/w/snap_before.txt`,
08/09/2026) : sous ``RELIQUARY_PROTOCOL_VERSION=5`` le corps de
``BatchSubmissionRequest`` / ``SubmissionPrecommitRequest`` et les
préimages signées doivent rester byte-identiques après le port.
"""
import hashlib
import importlib

import pytest

HK = "5DvpFN3QEa9iimQiA5jQaRmx8dbW2uxonM53j51Cw3kBva7q"
CKPT = "7ea1150e81ca229519b1d137ef5f8ec2c63ce361"
RAND = "a42a203fd7f0" + "0" * 52
NONCE = "0011223344556677"
ROUND = 30123456

# --- snapshot figé (worktree tel quel, avant le port) ---------------------
BODY_LEN_V5 = 2510
BODY_SHA_V5 = "89e2749f2a2dbfa4a9c086c690610b0ae4d6d55433750b117892ad2213380e01"
PRECOMMIT_JSON_V5 = (
    '{"miner_hotkey":"5DvpFN3QEa9iimQiA5jQaRmx8dbW2uxonM53j51Cw3kBva7q",'
    '"prompt_idx":4174,"window_start":44000,'
    '"merkle_root":"abababababababababababababababababababababababababababababababab",'
    '"checkpoint_hash":"7ea1150e81ca229519b1d137ef5f8ec2c63ce361",'
    '"environment":"opencodeinstruct","payload_bytes":2510,'
    '"payload_sha256":"89e2749f2a2dbfa4a9c086c690610b0ae4d6d55433750b117892ad2213380e01",'
    '"drand_round":30123456,"protocol_version":5,'
    '"generation_profile_id":"qwen3-4b-base-dapo-reasoning-v5",'
    '"nonce":"0011223344556677",'
    '"precommit_signature":"' + "ef" * 64 + '"}'
)
ENVELOPE_BINDING_V5 = "6d9fdf994e8ac1a7f6eb4038f48183f11261d9e324c28444b44b0b80f844a516"
PRECOMMIT_BINDING_V5 = "bec10c175145335cd467c4ef6d4a4895a26d4046d5555bf115cb0de02afc0baf"


@pytest.fixture
def constants_v5(monkeypatch):
    monkeypatch.setenv("RELIQUARY_PROTOCOL_VERSION", "5")
    monkeypatch.delenv("RELIQUARY_GENERATION_PROFILE_ID", raising=False)
    import reliquary.constants as c
    import reliquary.protocol.submission as sub
    c = importlib.reload(c)
    # ``submission.M_ROLLOUTS`` est lié à l'import (env du process) ; le
    # validateur le lit à l'appel → on l'aligne sur v5 (16), restauré par
    # monkeypatch, sans recharger le module (les classes resteraient les
    # mêmes objets pour engine.py & co).
    monkeypatch.setattr(sub, "M_ROLLOUTS", c.M_ROLLOUTS)
    yield c
    monkeypatch.undo()
    importlib.reload(c)


def _build_v5(c):
    from reliquary.protocol.submission import (
        BatchSubmissionRequest,
        RolloutSubmission,
        SubmissionPrecommitRequest,
    )

    req = BatchSubmissionRequest(
        miner_hotkey=HK, prompt_idx=4174, window_start=44000,
        merkle_root="ab" * 32,
        rollouts=[
            RolloutSubmission(
                tokens=[101, 202 + i, 303],
                reward=1.0 if i < 4 else 0.0,
                commit={"tokens": [101, 202 + i, 303], "proof_version": "v7"},
                env_name="opencodeinstruct",
            )
            for i in range(c.M_ROLLOUTS)
        ],
        checkpoint_hash=CKPT, drand_round=ROUND, nonce=NONCE,
        envelope_signature="cd" * 64,
        protocol_version=c.PROTOCOL_VERSION,
        generation_profile_id=c.GENERATION_PROFILE_ID,
    )
    body = req.model_dump_json().encode("utf-8")
    pre = SubmissionPrecommitRequest(
        miner_hotkey=HK, prompt_idx=4174, window_start=44000,
        merkle_root="ab" * 32, checkpoint_hash=CKPT,
        environment="opencodeinstruct", payload_bytes=len(body),
        payload_sha256=hashlib.sha256(body).hexdigest(), drand_round=ROUND,
        protocol_version=c.PROTOCOL_VERSION,
        generation_profile_id=c.GENERATION_PROFILE_ID,
        nonce=NONCE, precommit_signature="ef" * 64,
    )
    return body, pre.model_dump_json()


def test_v5_wire_snapshot(constants_v5):
    """Sous v5, le corps /submit et le precommit sont byte-identiques au
    snapshot pris avant le port : le port ne touche pas aux octets envoyés."""
    c = constants_v5
    assert (c.PROTOCOL_VERSION, c.M_ROLLOUTS) == (5, 16)
    assert c.GENERATION_PROFILE_ID == "qwen3-4b-base-dapo-reasoning-v5"
    body, pre_json = _build_v5(c)
    assert len(body) == BODY_LEN_V5
    assert hashlib.sha256(body).hexdigest() == BODY_SHA_V5
    assert pre_json == PRECOMMIT_JSON_V5


def test_v5_bindings_snapshot(constants_v5):
    """Les préimages signées (enveloppe v3 + precommit) sous v5 = snapshot."""
    from reliquary.protocol.signatures import (
        build_envelope_binding,
        build_precommit_binding,
    )

    c = constants_v5
    body, _ = _build_v5(c)
    env_b = build_envelope_binding(
        miner_hotkey=HK, window_start=44000, prompt_idx=4174,
        merkle_root="ab" * 32, checkpoint_hash=CKPT, drand_round=ROUND,
        randomness=RAND, nonce=NONCE, protocol_version=c.PROTOCOL_VERSION,
        generation_profile_id=c.GENERATION_PROFILE_ID,
    )
    pc_b = build_precommit_binding(
        miner_hotkey=HK, window_start=44000, prompt_idx=4174,
        merkle_root="ab" * 32, checkpoint_hash=CKPT,
        environment="opencodeinstruct", payload_bytes=len(body),
        payload_sha256=hashlib.sha256(body).hexdigest(), drand_round=ROUND,
        randomness=RAND, protocol_version=c.PROTOCOL_VERSION, nonce=NONCE,
        generation_profile_id=c.GENERATION_PROFILE_ID,
    )
    assert env_b.hex() == ENVELOPE_BINDING_V5
    assert pc_b.hex() == PRECOMMIT_BINDING_V5


def _envelope_v3_preimage(protocol_version: int, profile: str) -> bytes:
    """Reconstruction indépendante de la préimage v3 documentée dans
    ``build_envelope_binding`` (domaine v3, 8 parts legacy + protocole BE-8
    + profil), pour vérifier ce qui change entre v5 et v6."""
    from reliquary.protocol.signatures import ENVELOPE_DOMAIN_V3

    parts = (
        HK.encode(), (44000).to_bytes(8, "big"), (4174).to_bytes(8, "big"),
        bytes.fromhex("ab" * 32), CKPT.encode(), ROUND.to_bytes(8, "big"),
        bytes.fromhex(RAND), NONCE.encode(),
        protocol_version.to_bytes(8, "big"), profile.encode(),
    )
    h = hashlib.sha256(ENVELOPE_DOMAIN_V3)
    for p in parts:
        h.update(len(p).to_bytes(4, "big"))
        h.update(p)
    return h.digest()


def test_envelope_v6_preimage_identique_v5_sauf_champs():
    """Même préimage v3 : seuls ``protocol_version`` (6) et le profil v6
    changent dans les octets signés ; la reconstruction v5 = snapshot."""
    from reliquary.protocol.signatures import build_envelope_binding

    v5 = _envelope_v3_preimage(5, "qwen3-4b-base-dapo-reasoning-v5")
    assert v5.hex() == ENVELOPE_BINDING_V5
    v6 = _envelope_v3_preimage(6, "qwen3-4b-base-dapo-fill-closed-v6")
    assert build_envelope_binding(
        miner_hotkey=HK, window_start=44000, prompt_idx=4174,
        merkle_root="ab" * 32, checkpoint_hash=CKPT, drand_round=ROUND,
        randomness=RAND, nonce=NONCE, protocol_version=6,
        generation_profile_id="qwen3-4b-base-dapo-fill-closed-v6",
    ) == v6
    assert v6 != v5
