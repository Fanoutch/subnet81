"""Réveiller le générateur dès le flip de fenêtre (16/09).

Pendant le trou 503, ``_generator_loop`` passe par la branche
``should_pause_bake`` et dort ``asyncio.sleep(1.0)`` avant de re-tester. Rien
ne le réveille quand ``_trigger_loop`` voit la nouvelle ``randomness`` : le
bake démarre donc 0 à 1 s APRÈS la détection (0,5 s en moyenne), en plus des
~0,7 s de téléchargement de ``/state``. Or la tête est jugée à la seconde près :
la moitié des 112 places code part désormais avant ~15 s.

Correctif : le flip lève un ``asyncio.Event`` ; la branche de pause attend cet
événement au plus 1,0 s au lieu de dormir 1,0 s. Même borne haute, réveil
immédiat au flip. Dormant : ``RELIQUARY_WAKE_ON_FLIP=1`` pour l'activer.
Signature : délai ouverture → début du bake (journal « groupe 1/10 prêt à X s »).
"""
from __future__ import annotations

import asyncio
import time

from reliquary.miner import engine


def _eng():
    return engine.MiningEngine.__new__(engine.MiningEngine)


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_dormant_dort_la_seconde_entiere(monkeypatch):
    monkeypatch.delenv("RELIQUARY_WAKE_ON_FLIP", raising=False)
    e = _eng()

    async def go():
        t0 = time.monotonic()
        asyncio.get_running_loop().call_later(0.1, e._signal_flip)
        await e._pause_wait(0.4)
        return time.monotonic() - t0
    assert _run(go()) >= 0.35                      # le flip n'écourte rien


def test_actif_reveil_immediat_au_flip(monkeypatch):
    monkeypatch.setenv("RELIQUARY_WAKE_ON_FLIP", "1")
    e = _eng()

    async def go():
        t0 = time.monotonic()
        asyncio.get_running_loop().call_later(0.1, e._signal_flip)
        await e._pause_wait(2.0)
        return time.monotonic() - t0
    assert _run(go()) < 0.5


def test_actif_sans_flip_garde_la_borne(monkeypatch):
    monkeypatch.setenv("RELIQUARY_WAKE_ON_FLIP", "1")
    e = _eng()

    async def go():
        t0 = time.monotonic()
        await e._pause_wait(0.3)
        return time.monotonic() - t0
    assert 0.25 <= _run(go()) < 1.0


def test_un_flip_ne_reveille_qu_une_fois(monkeypatch):
    """L'événement est consommé : pas de boucle chaude après un flip."""
    monkeypatch.setenv("RELIQUARY_WAKE_ON_FLIP", "1")
    e = _eng()

    async def go():
        e._signal_flip()
        t0 = time.monotonic(); await e._pause_wait(1.0); first = time.monotonic() - t0
        t0 = time.monotonic(); await e._pause_wait(0.3); second = time.monotonic() - t0
        return first, second
    first, second = _run(go())
    assert first < 0.1 and second >= 0.25


def test_signal_avant_toute_attente_ne_casse_rien(monkeypatch):
    monkeypatch.setenv("RELIQUARY_WAKE_ON_FLIP", "1")
    e = _eng()
    e._signal_flip()                                # hors boucle : aucun crash
