"""Port of the mandatory upload-precommit handshake (upstream 8835a95).

The live validator gates EVERY ``/submit`` behind a signed precommit
(``server.py:2413`` -> ``RejectReason.PRECOMMIT_REQUIRED``) whenever the
difficulty auction is enforced, and ``RELIQUARY_DIFFICULTY_AUCTION_ENFORCE``
defaults to "1". Without this handshake we submit nothing at all.

The binding golden below was computed by running upstream's own
``build_precommit_binding`` from ``git show 8835a95:reliquary/protocol/signatures.py``
against the fixture inputs, so these tests pin BYTE PARITY with the validator
rather than merely re-testing our own re-implementation.
"""

from __future__ import annotations

import hashlib
import json

import httpx
import pytest

from reliquary.protocol.submission import (
    BatchSubmissionResponse,
    RejectReason,
)

# --- fixture inputs shared with the upstream golden computation --------------
BINDING_FIXTURE = dict(
    miner_hotkey="5DvpFN3QEa9iimQiA5jQaRmx8dbW2uxonM53j51Cw3kBva7q",
    window_start=24243,
    prompt_idx=987654,
    merkle_root="a" * 64,
    checkpoint_hash="deadbeef" * 8,
    environment="openmathinstruct",
    payload_bytes=131072,
    payload_sha256="b" * 64,
    drand_round=1234567,
    randomness="e421aa6374ada3f68d1e22f5af6778ec34abe5c53f53945eed2385f3c98ee3dc",
    protocol_version=2,
    nonce="0123456789abcdef0123456789abcdef",
)
UPSTREAM_GOLDEN = "d75dfa7fb383cd10b7f568e6e5a0a5290d33924e3da026c73b345520071bee7c"


# --- 1. signature layer ------------------------------------------------------


def test_precommit_domain_is_the_v2_upload_tag():
    from reliquary.protocol.signatures import PRECOMMIT_DOMAIN

    assert PRECOMMIT_DOMAIN == b"reliquary-upload-precommit-v2"


def test_build_precommit_binding_matches_upstream_golden():
    """Byte parity with the validator's own binding, or the signature fails."""
    from reliquary.protocol.signatures import build_precommit_binding

    assert build_precommit_binding(**BINDING_FIXTURE).hex() == UPSTREAM_GOLDEN


@pytest.mark.parametrize("field", sorted(BINDING_FIXTURE))
def test_every_field_is_bound_into_the_precommit_digest(field):
    """A field the validator signs but we ignore = forgeable commitment."""
    from reliquary.protocol.signatures import build_precommit_binding

    mutated = dict(BINDING_FIXTURE)
    value = mutated[field]
    if isinstance(value, int):
        mutated[field] = value + 1
    elif field in {"merkle_root", "payload_sha256", "randomness"}:
        # hex-decoded fields: mutate a digit, appending would break fromhex
        mutated[field] = "f" + str(value)[1:]
    else:
        mutated[field] = str(value) + "x"
    assert build_precommit_binding(**mutated) != build_precommit_binding(
        **BINDING_FIXTURE
    )


def test_sign_precommit_is_verifiable_by_the_hotkey():
    import bittensor as bt

    from reliquary.protocol.signatures import build_precommit_binding, sign_precommit

    keypair = bt.Keypair.create_from_uri("//Alice")
    fields = dict(BINDING_FIXTURE, miner_hotkey=keypair.ss58_address)

    class _Wallet:
        hotkey = keypair

    signature = sign_precommit(wallet=_Wallet(), **fields)
    assert keypair.verify(
        data=build_precommit_binding(**fields), signature=signature
    )


# --- 2. enum / wire models ---------------------------------------------------


@pytest.mark.parametrize(
    "value", ["precommit_required", "precommit_invalid", "precommit_expired"]
)
def test_reject_reason_has_the_new_precommit_members(value):
    assert RejectReason(value).value == value


def test_precommit_required_verdict_parses_instead_of_raising():
    """The validator returns this at HTTP 200 — a strict enum would explode."""
    resp = BatchSubmissionResponse.model_validate(
        {"accepted": False, "reason": "precommit_required"}
    )
    assert resp.reason is RejectReason.PRECOMMIT_REQUIRED


def test_precommit_request_never_carries_randomness_on_the_wire():
    """Upstream signs randomness but does NOT transmit it (server substitutes
    its own); an extra field trips ``extra="forbid"`` at the validator."""
    from reliquary.protocol.submission import SubmissionPrecommitRequest

    payload = SubmissionPrecommitRequest(
        **{k: v for k, v in BINDING_FIXTURE.items() if k != "randomness"},
        precommit_signature="ab" * 32,
    ).model_dump()
    assert "randomness" not in payload


def test_precommit_response_carries_receipt_and_deadline():
    from reliquary.protocol.submission import SubmissionPrecommitResponse

    resp = SubmissionPrecommitResponse.model_validate(
        {
            "accepted": True,
            "reason": "accepted",
            "receipt_id": "r-123",
            "upload_deadline_ts": 1234.5,
        }
    )
    assert resp.receipt_id == "r-123" and resp.upload_deadline_ts == 1234.5


# --- 3. the two-phase submit flow -------------------------------------------


def _request():
    """A minimal valid BatchSubmissionRequest for the submitter."""
    from reliquary.protocol.submission import (
        BatchSubmissionRequest,
        RolloutSubmission,
    )

    return BatchSubmissionRequest(
        miner_hotkey=BINDING_FIXTURE["miner_hotkey"],
        prompt_idx=42,
        window_start=100,
        merkle_root="00" * 32,
        rollouts=[
            RolloutSubmission(
                tokens=[1, 2, 3],
                reward=1.0 if i < 4 else 0.0,
                commit={"tokens": [1, 2, 3], "proof_version": "v7"},
                env_name="openmathinstruct",
            )
            for i in range(8)
        ],
        checkpoint_hash="sha256:test",
        # the engine always stamps these before submitting
        drand_round=BINDING_FIXTURE["drand_round"],
        nonce=BINDING_FIXTURE["nonce"],
        envelope_signature="ab" * 32,
    )


class _Recorder:
    """Captures the two POSTs so we can assert the handshake contract."""

    def __init__(self, precommit_body=None, submit_status=200):
        self.calls = []
        self._precommit_body = precommit_body or {
            "accepted": True,
            "reason": "accepted",
            "receipt_id": "receipt-abc",
            "upload_deadline_ts": 9e9,
        }
        self._submit_status = submit_status

    async def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        if request.url.path.endswith("/submit/precommit"):
            return httpx.Response(200, json=self._precommit_body)
        return httpx.Response(
            self._submit_status, json={"accepted": True, "reason": "accepted"}
        )

    @property
    def client(self):
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))


def _wallet():
    import bittensor as bt

    keypair = bt.Keypair.create_from_uri("//Alice")

    class _Wallet:
        hotkey = keypair

    return _Wallet()


@pytest.mark.asyncio
async def test_submit_sends_precommit_before_body():
    from reliquary.miner.submitter import submit_batch_v2

    rec = _Recorder()
    async with rec.client as client:
        resp = await submit_batch_v2(
            "http://v", _request(), client=client,
            wallet=_wallet(), randomness=BINDING_FIXTURE["randomness"],
        )

    assert resp.accepted
    assert [c.url.path for c in rec.calls] == ["/submit/precommit", "/submit"]


@pytest.mark.asyncio
async def test_body_bytes_match_the_sha256_we_precommitted():
    """The whole point of the handshake: the reveal must be the committed bytes."""
    from reliquary.miner.submitter import submit_batch_v2

    rec = _Recorder()
    async with rec.client as client:
        await submit_batch_v2(
            "http://v", _request(), client=client,
            wallet=_wallet(), randomness=BINDING_FIXTURE["randomness"],
        )

    precommit, submit = rec.calls
    committed = json.loads(precommit.content)
    assert committed["payload_sha256"] == hashlib.sha256(submit.content).hexdigest()
    assert committed["payload_bytes"] == len(submit.content)


@pytest.mark.asyncio
async def test_reveal_carries_the_receipt_header():
    from reliquary.miner.submitter import submit_batch_v2

    rec = _Recorder()
    async with rec.client as client:
        await submit_batch_v2(
            "http://v", _request(), client=client,
            wallet=_wallet(), randomness=BINDING_FIXTURE["randomness"],
        )

    assert rec.calls[1].headers["X-Reliquary-Precommit"] == "receipt-abc"


@pytest.mark.asyncio
async def test_rejected_precommit_short_circuits_without_uploading_body():
    """Don't waste the window uploading a body the validator already refused."""
    from reliquary.miner.submitter import submit_batch_v2

    rec = _Recorder(
        precommit_body={"accepted": False, "reason": "prompt_full"}
    )
    async with rec.client as client:
        resp = await submit_batch_v2(
            "http://v", _request(), client=client,
            wallet=_wallet(), randomness=BINDING_FIXTURE["randomness"],
        )

    assert not resp.accepted
    assert resp.reason is RejectReason.PROMPT_FULL
    assert [c.url.path for c in rec.calls] == ["/submit/precommit"]


@pytest.mark.asyncio
async def test_legacy_direct_submit_when_no_wallet_is_passed():
    """Single-env / test callers must keep the old one-shot behaviour."""
    from reliquary.miner.submitter import submit_batch_v2

    rec = _Recorder()
    async with rec.client as client:
        resp = await submit_batch_v2("http://v", _request(), client=client)

    assert resp.accepted
    assert [c.url.path for c in rec.calls] == ["/submit"]
