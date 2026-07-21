"""The engine must hand the submitter what the precommit needs.

``submit_batch_v2`` silently falls back to the legacy bare ``/submit`` when
``wallet``/``randomness`` are missing — which the live validator answers with
PRECOMMIT_REQUIRED. So the wiring itself is the thing worth pinning: a missing
kwarg is invisible in unit tests but costs 100% of submissions in production.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def test_submit_entry_passes_wallet_and_randomness_to_submitter(monkeypatch):
    from reliquary.miner import submitter as submitter_mod
    from reliquary.miner.engine import MiningEngine
    from reliquary.protocol.submission import BatchSubmissionResponse, RejectReason

    captured = {}

    async def _fake_submit(url, request, *, client=None, timeout=60.0, **kwargs):
        captured.update(kwargs)
        return BatchSubmissionResponse(
            accepted=True, reason=RejectReason.ACCEPTED
        )

    monkeypatch.setattr(submitter_mod, "submit_batch_v2", _fake_submit)

    engine = MiningEngine.__new__(MiningEngine)
    wallet = SimpleNamespace(
        hotkey=SimpleNamespace(ss58_address="5Dvp_test_hotkey")
    )
    engine.wallet = wallet
    engine._submitted_env = {}
    engine._entry_env_name = MagicMock(return_value="openmathinstruct")
    engine._finalize_pool_entry = MagicMock(return_value=(["r"], "ab" * 32))
    engine._build_signed_request_sync = MagicMock(
        return_value=(1234567, MagicMock())
    )

    state = SimpleNamespace(
        window_n=24243,
        randomness="e4" * 32,
    )
    results: list = []

    asyncio.run(
        engine._submit_entry(
            {"prompt_idx": 42}, state, "http://v", None, results,
        )
    )

    assert captured.get("wallet") is wallet, (
        "engine dropped `wallet` -> submitter takes the legacy path -> "
        "validator answers PRECOMMIT_REQUIRED"
    )
    assert captured.get("randomness") == state.randomness
