"""Observabilité du filtre LTA : chaque drop s'auto-classifie.

Question du 28/08 : le filtre local (marges volontaires : argmax 0,985 vs
0,99 validateur, dur 3e-8 vs 1e-8) jette-t-il des groupes qui seraient
PASSÉS chez le validateur ? Pour le mesurer sur données réelles, le drop
embarque un verdict simulé contre les seuils RÉELS (constants c0b01d1 :
TOKEN_AUTH_THRESHOLD=1e-8, ALL_TOKEN_AUTH 1e-5 & argmax 0.99) :
`marge_seule=True` = drop imputable à notre marge uniquement.

Invariants :
- même raison que local_verif_screen (aucun changement de comportement) ;
- token dans la bande de marge -> marge_seule=True ;
- token sous le seuil réel -> marge_seule=False ;
- rollout sain -> (None, None).
"""
from __future__ import annotations

import math

import pytest

from reliquary.miner import engine


def _lp(p):
    return math.log(p)


def test_sain(monkeypatch):
    monkeypatch.setenv("RELIQUARY_LTA_HARD_MIN", "3e-8")
    reason, detail = engine.local_verif_screen_detail(
        [_lp(0.5)] * 10, [0.6] * 10,
    )
    assert reason is None and detail is None


def test_hard_marge_seule(monkeypatch):
    # token à 2e-8 : sous NOTRE seuil (3e-8) mais AU-DESSUS du sien (1e-8)
    monkeypatch.setenv("RELIQUARY_LTA_HARD_MIN", "3e-8")
    lps = [_lp(0.5)] * 9 + [_lp(2e-8)]
    reason, detail = engine.local_verif_screen_detail(lps, [0.5] * 10)
    assert reason == "local_token_auth_hard"
    assert detail["marge_seule"] is True
    assert detail["pire_p"] == pytest.approx(2e-8, rel=1e-6)


def test_hard_reel(monkeypatch):
    # token à 5e-9 : échouerait AUSSI chez le validateur (1e-8)
    monkeypatch.setenv("RELIQUARY_LTA_HARD_MIN", "3e-8")
    lps = [_lp(0.5)] * 9 + [_lp(5e-9)]
    reason, detail = engine.local_verif_screen_detail(lps, [0.5] * 10)
    assert reason == "local_token_auth_hard"
    assert detail["marge_seule"] is False


def test_conditionnel_marge_seule(monkeypatch):
    # p<1e-5 avec argmax 0,987 : NOTRE bande [0,985; 0,99) — lui exige >=0,99
    monkeypatch.setenv("RELIQUARY_LOCAL_TOKEN_AUTH", "1")
    monkeypatch.setenv("RELIQUARY_LTA_CHOSEN_MAX", "1e-5")
    monkeypatch.setenv("RELIQUARY_LTA_ARGMAX_MIN", "0.985")
    lps = [_lp(0.5)] * 9 + [_lp(5e-6)]
    amx = [0.5] * 9 + [0.987]
    reason, detail = engine.local_verif_screen_detail(lps, amx)
    assert reason == "local_token_auth"
    assert detail["marge_seule"] is True


def test_conditionnel_reel(monkeypatch):
    monkeypatch.setenv("RELIQUARY_LOCAL_TOKEN_AUTH", "1")
    monkeypatch.setenv("RELIQUARY_LTA_CHOSEN_MAX", "1e-5")
    monkeypatch.setenv("RELIQUARY_LTA_ARGMAX_MIN", "0.985")
    lps = [_lp(0.5)] * 9 + [_lp(5e-6)]
    amx = [0.5] * 9 + [0.995]   # >= 0,99 : rejeté chez lui aussi
    reason, detail = engine.local_verif_screen_detail(lps, amx)
    assert reason == "local_token_auth"
    assert detail["marge_seule"] is False


def test_meme_raison_que_le_screen_historique(monkeypatch):
    monkeypatch.setenv("RELIQUARY_LTA_HARD_MIN", "3e-8")
    lps = [_lp(0.5)] * 9 + [_lp(2e-8)]
    assert engine.local_verif_screen(lps, [0.5] * 10) == \
        engine.local_verif_screen_detail(lps, [0.5] * 10)[0]
