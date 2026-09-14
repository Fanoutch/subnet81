"""Reveal du corps borné par la grâce d'upload du reçu (V1, #251).

Le precommit accordé porte ``upload_deadline_ts`` (horloge validateur, grâce
33 s). Un corps qui n'arrive pas avant compte un ``reveal_deadline_missed`` ;
4 fenêtres touchées sur 10 arment le disjoncteur no-reveal (cooldown 10 → 50
→ 250 fenêtres, soit des heures à 30 min la fenêtre). L'ancien envoi (timeout
60 s × 3 essais + 1/2/4 s de pause) pouvait déborder largement : chaque essai
est désormais borné par le temps restant avant l'échéance, et on n'en relance
pas un qui ne peut plus arriver à temps.
"""
from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from reliquary.miner import submitter as sub


class _Client:
    def __init__(self, behaviours):
        self.behaviours = list(behaviours)
        self.timeouts: list[float] = []

    async def post(self, url, *, content, headers, timeout):
        self.timeouts.append(timeout)
        b = self.behaviours.pop(0)
        if isinstance(b, Exception):
            raise b
        return b


def _ok():
    return httpx.Response(200, json={"accepted": True, "reason": "submitted"})


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    async def fake_sleep(s):
        return None
    monkeypatch.setattr(sub.asyncio, "sleep", fake_sleep)


def test_timeout_dun_essai_borne_par_lecheance():
    cli = _Client([_ok()])
    deadline = time.time() + 20.0
    asyncio.run(sub._post_bytes_with_retry(
        "http://v/submit", b"{}", headers={}, client=cli, timeout=60.0,
        deadline_ts=deadline))
    assert cli.timeouts[0] <= 20.0 - sub._REVEAL_DEADLINE_MARGIN_S + 0.1


def test_pas_de_nouvel_essai_apres_lecheance(monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr(sub.time, "time", lambda: clock["t"])

    class _SlowClient(_Client):
        async def post(self, url, *, content, headers, timeout):
            clock["t"] += timeout          # l'essai consomme tout son budget
            return await super().post(url, content=content, headers=headers,
                                      timeout=timeout)

    cli = _SlowClient([httpx.ReadTimeout("x"), _ok()])
    with pytest.raises(sub.SubmissionError, match="deadline"):
        asyncio.run(sub._post_bytes_with_retry(
            "http://v/submit", b"{}", headers={}, client=cli, timeout=60.0,
            deadline_ts=1000.0 + 30.0))
    assert len(cli.timeouts) == 1


def test_echeance_deja_passee_nenvoie_rien():
    cli = _Client([_ok()])
    with pytest.raises(sub.SubmissionError, match="deadline"):
        asyncio.run(sub._post_bytes_with_retry(
            "http://v/submit", b"{}", headers={}, client=cli, timeout=60.0,
            deadline_ts=time.time() - 1.0))
    assert cli.timeouts == []


def test_sans_echeance_comportement_historique():
    cli = _Client([httpx.ConnectError("x"), _ok()])
    resp = asyncio.run(sub._post_bytes_with_retry(
        "http://v/submit", b"{}", headers={}, client=cli, timeout=60.0))
    assert resp.accepted is True
    assert cli.timeouts == [60.0, 60.0]
