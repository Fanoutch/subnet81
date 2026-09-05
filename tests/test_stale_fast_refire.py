"""Re-tir immédiat sur stale_round (RELIQUARY_STALE_FAST_REFIRE, 01/09).

Mesure du 31/08 (agent chrono) : 21,6 % des têtes meurent en stale_round et le
re-tir par la file de retry coûte +6,5 s — condamné d'avance (le round
re-vieillit). Le fix : UN re-tir immédiat dans _submit_entry, round et nonce
frais, sans repasser par la file. Flag OFF par défaut = comportement identique.
"""
import asyncio
import types

import pytest

from reliquary.miner.engine import MiningEngine


def _resp(accepted, reason):
    return types.SimpleNamespace(accepted=accepted, reason=reason)


def _engine():
    e = MiningEngine.__new__(MiningEngine)
    e.wallet = types.SimpleNamespace(
        hotkey=types.SimpleNamespace(ss58_address="5Dtest"))
    e._submitted_env = {}
    e._entry_env_name = lambda entry: "opencodeinstruct"
    e._finalize_pool_entry = lambda entry, rnd: ([], "merkle-x")
    e._build_signed_request_sync = (
        lambda subs, mr, pi, state, hk, nonce: (123, {"req": nonce}))
    e._record_drop = lambda dropped: None
    return e


def _run(responses, monkeypatch, flag):
    monkeypatch.setenv("RELIQUARY_STALE_FAST_REFIRE", flag)
    calls = []

    async def fake_submit(url, request, *, client, wallet, randomness):
        calls.append(request)
        return responses[min(len(calls) - 1, len(responses) - 1)]

    import reliquary.miner.submitter as sub
    monkeypatch.setattr(sub, "submit_batch_v2", fake_submit)
    e = _engine()
    state = types.SimpleNamespace(randomness="ab" * 32, window_n=7)
    results = []
    entry = {"prompt_idx": 42}
    out = asyncio.run(e._submit_entry(entry, state, "http://v", None, results))
    return out, calls, results


def test_refire_immediat_sur_stale(monkeypatch):
    (entry, resp), calls, results = _run(
        [_resp(False, "stale_round"), _resp(True, "accepted")],
        monkeypatch, "1")
    assert len(calls) == 2, "un re-tir immédiat attendu"
    assert calls[0]["req"] != calls[1]["req"], "nonce frais au re-tir"
    assert resp.accepted is True
    assert len(results) == 1, "une seule réponse finale dans results"


def test_un_seul_refire(monkeypatch):
    (entry, resp), calls, _ = _run(
        [_resp(False, "stale_round"), _resp(False, "stale_round")],
        monkeypatch, "1")
    assert len(calls) == 2, "jamais plus d'UN re-tir"
    assert str(resp.reason) == "stale_round"


def test_flag_off_comportement_actuel(monkeypatch):
    (entry, resp), calls, _ = _run(
        [_resp(False, "stale_round"), _resp(True, "accepted")],
        monkeypatch, "0")
    assert len(calls) == 1, "flag OFF = aucun re-tir"


def test_pas_de_refire_sur_autre_rejet(monkeypatch):
    (entry, resp), calls, _ = _run(
        [_resp(False, "batch_filled"), _resp(True, "accepted")],
        monkeypatch, "1")
    assert len(calls) == 1
