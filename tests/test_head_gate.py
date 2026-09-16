"""Porte de tête : les K premiers groupes d'une fenêtre prouvent seuls (16/09).

Fix 1 (``GRADE_CONCURRENCY`` 8 → 3) a bien accéléré la preuve (p50 3,67 →
2,14 s sur 125 groupes) mais les payés ont chuté (6,33 → 2,80/fen) et il a été
replié : il bridait TOUTE la fenêtre. Sur un seul GPU on ne peut pas accélérer
la tête « sans rien toucher aux suivants » : la preuve est rapide parce que peu
d'autres tournent en même temps. La forme réalisable est une EXCLUSIVITÉ
BRÈVE — les K premiers groupes de la fenêtre passent seuls, puis dès qu'ils
ont fini (ou après ``timeout``) la concurrence normale reprend. Contrairement
à ``GRADE=3`` elle ne dure que le temps de la tête.

⚠️ Elle retarde quand même les groupes K+1… de ~2-3 s : sa valeur dépend du
verdict de l'expérience en cours (repli à 8). Le retard est mesurable dans le
dump : ``t_pick − t_ready`` (0,00 s aujourd'hui).
Dormant : ``RELIQUARY_HEAD_GATE_K=0`` par défaut.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from reliquary.miner import engine


def _eng(window=100):
    e = engine.MiningEngine.__new__(engine.MiningEngine)
    e._cached_window_n = window
    return e


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_dormant_par_defaut(monkeypatch):
    monkeypatch.delenv("RELIQUARY_HEAD_GATE_K", raising=False)
    e = _eng()

    async def go():
        return [await e._head_gate_enter() for _ in range(5)]
    assert _run(go()) == [False] * 5


def test_les_k_premiers_passent_les_suivants_attendent_la_tete(monkeypatch):
    monkeypatch.setenv("RELIQUARY_HEAD_GATE_K", "2")
    monkeypatch.setenv("RELIQUARY_HEAD_GATE_TIMEOUT_S", "5")
    e = _eng()
    order = []

    async def head(i):
        is_head = await e._head_gate_enter()
        order.append(("start", i, is_head))
        await asyncio.sleep(0.2)
        order.append(("end", i))
        e._head_gate_exit(is_head)

    async def tail(i):
        await asyncio.sleep(0.01)                 # arrive après les têtes
        is_head = await e._head_gate_enter()
        order.append(("start", i, is_head))

    async def go():
        await asyncio.gather(head(1), head(2), tail(3))
    _run(go())
    starts = [x for x in order if x[0] == "start"]
    assert (("start", 1, True) in starts) and (("start", 2, True) in starts)
    assert ("start", 3, False) in starts
    # la queue ne démarre qu'après la fin des DEUX têtes
    assert order.index(("start", 3, False)) > order.index(("end", 1))
    assert order.index(("start", 3, False)) > order.index(("end", 2))


def test_attente_bornee_par_le_timeout(monkeypatch):
    """Une tête qui ne rend jamais la main ne doit pas geler la fenêtre."""
    monkeypatch.setenv("RELIQUARY_HEAD_GATE_K", "1")
    monkeypatch.setenv("RELIQUARY_HEAD_GATE_TIMEOUT_S", "0.3")
    e = _eng()

    async def go():
        assert await e._head_gate_enter() is True          # tête, jamais relâchée
        t0 = time.monotonic()
        assert await e._head_gate_enter() is False
        return time.monotonic() - t0
    waited = _run(go())
    assert 0.2 <= waited <= 1.0


def test_nouvelle_fenetre_repart_de_zero(monkeypatch):
    monkeypatch.setenv("RELIQUARY_HEAD_GATE_K", "1")
    monkeypatch.setenv("RELIQUARY_HEAD_GATE_TIMEOUT_S", "0.2")
    e = _eng(window=100)

    async def go():
        a = await e._head_gate_enter()
        e._head_gate_exit(a)
        b = await e._head_gate_enter()                   # même fenêtre -> queue
        e._cached_window_n = 101
        c = await e._head_gate_enter()                   # nouvelle fenêtre -> tête
        return a, b, c
    assert _run(go()) == (True, False, True)


def test_valeurs_illisibles_reviennent_au_dormant(monkeypatch):
    monkeypatch.setenv("RELIQUARY_HEAD_GATE_K", "beaucoup")
    e = _eng()
    assert _run(e._head_gate_enter()) is False


# ------------------------------------------------------ câblage dans le flux
def test_flux_la_queue_attend_la_fin_de_la_tete(monkeypatch):
    """Dans _grade_chunk_streaming : K=1, le 2e _pre_bake_entry ne commence
    qu'après la fin du 1er, alors que le sémaphore (8) les laisserait passer."""
    monkeypatch.setenv("RELIQUARY_HEAD_GATE_K", "1")
    monkeypatch.setenv("RELIQUARY_HEAD_GATE_TIMEOUT_S", "5")
    monkeypatch.setenv("RELIQUARY_GRADE_CONCURRENCY", "8")
    e = _eng()
    log = []

    def pre_bake(prompt_idx, problem, expected_ckpt_n, env):
        log.append(("start", prompt_idx, time.monotonic()))
        time.sleep(0.25)
        log.append(("end", prompt_idx, time.monotonic()))
        return None

    async def post(entry, prompt_idx, entries, env):
        return None

    e._pre_bake_entry = pre_bake
    e._post_grade_entry = post
    _run(e._grade_chunk_streaming([(1, {}), (2, {})], [], expected_ckpt_n=1, env=None))
    t = {(k, p): ts for k, p, ts in log}
    assert t[("start", 2)] >= t[("end", 1)] - 0.02


def test_flux_dormant_inchange(monkeypatch):
    monkeypatch.delenv("RELIQUARY_HEAD_GATE_K", raising=False)
    monkeypatch.setenv("RELIQUARY_GRADE_CONCURRENCY", "8")
    e = _eng()
    log = []

    def pre_bake(prompt_idx, problem, expected_ckpt_n, env):
        log.append(("start", prompt_idx, time.monotonic()))
        time.sleep(0.25)
        log.append(("end", prompt_idx, time.monotonic()))

    async def post(entry, prompt_idx, entries, env):
        return None

    e._pre_bake_entry = pre_bake
    e._post_grade_entry = post
    _run(e._grade_chunk_streaming([(1, {}), (2, {})], [], expected_ckpt_n=1, env=None))
    t = {(k, p): ts for k, p, ts in log}
    assert t[("start", 2)] < t[("end", 1)]               # en parallèle, comme avant
