"""Cadence de réessai après ``batch_filled`` (16/09).

Mesuré sur 91 fenêtres : un groupe refusé au 1er tir vers 37 s n'est réadmis
qu'à **64,8 s** (médiane) — 28 s perdues — et celui refusé vers 25 s ne repasse
qu'à 45,1 s. La chaîne prêt→envoi, elle, est PLATE à 4,3 s : le retard vient
donc de notre échelle de réessais ``min(1,0 × essais, 5,0)``, qui demande 6-7
tours pour en arriver là.

Or ``batch_filled`` est un fourre-tout côté validateur : la PR #270 (a205b1c)
montre qu'il couvre aussi ``precommit_signature_busy`` — le pool de vérification
de signature, ``MAX_PENDING_PROOF_QUEUE_DEPTH = 64`` PARTAGÉ par tout le marché,
donc saturé à l'ouverture — et le serveur y répond ``Retry-After: 1``. On attend
5 s là où il demande 1 s. La marge existe : 520 refus ``batch_filled`` par
fenêtre contre **1,2 ``rate_limited``**.

Le pas et le plafond deviennent donc réglables, DÉFAUTS INCHANGÉS (1,0 / 5,0).
"""
from __future__ import annotations

import pytest

from reliquary.miner.engine import batch_filled_retry_delay


def test_defaut_identique_a_avant(monkeypatch):
    monkeypatch.delenv("RELIQUARY_BATCH_FILLED_RETRY_STEP", raising=False)
    monkeypatch.delenv("RELIQUARY_BATCH_FILLED_RETRY_MAX", raising=False)
    assert [batch_filled_retry_delay(k) for k in range(1, 8)] == [
        1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0]


def test_plafond_abaisse(monkeypatch):
    """Plafond à 1,5 s : 12 essais couvrent ~18 s au lieu de ~50."""
    monkeypatch.setenv("RELIQUARY_BATCH_FILLED_RETRY_MAX", "1.5")
    assert [batch_filled_retry_delay(k) for k in range(1, 5)] == [1.0, 1.5, 1.5, 1.5]
    assert sum(batch_filled_retry_delay(k) for k in range(1, 13)) == pytest.approx(17.5)


def test_pas_reglable(monkeypatch):
    monkeypatch.setenv("RELIQUARY_BATCH_FILLED_RETRY_STEP", "0.5")
    monkeypatch.setenv("RELIQUARY_BATCH_FILLED_RETRY_MAX", "2.0")
    assert [batch_filled_retry_delay(k) for k in range(1, 6)] == [0.5, 1.0, 1.5, 2.0, 2.0]


def test_valeurs_illisibles_reviennent_au_defaut(monkeypatch):
    monkeypatch.setenv("RELIQUARY_BATCH_FILLED_RETRY_STEP", "vite")
    monkeypatch.setenv("RELIQUARY_BATCH_FILLED_RETRY_MAX", "")
    assert batch_filled_retry_delay(3) == 3.0


def test_jamais_negatif_ni_nul(monkeypatch):
    """Un délai nul ferait tourner la file de réessais à vide sur le CPU."""
    monkeypatch.setenv("RELIQUARY_BATCH_FILLED_RETRY_STEP", "0")
    monkeypatch.setenv("RELIQUARY_BATCH_FILLED_RETRY_MAX", "0")
    assert batch_filled_retry_delay(1) >= 0.05
    monkeypatch.setenv("RELIQUARY_BATCH_FILLED_RETRY_STEP", "-3")
    assert batch_filled_retry_delay(2) >= 0.05


def test_la_decision_de_retir_utilise_la_cadence(monkeypatch):
    """``fire_retry_decision`` doit passer par la fonction, pas par 1,0 × n."""
    from types import SimpleNamespace

    from reliquary.miner import engine

    monkeypatch.setenv("RELIQUARY_BATCH_FILLED_RETRY_MAX", "1.5")
    fc = SimpleNamespace(picks_target=7, admitted={"opencodeinstruct": 10},
                         environments={"opencodeinstruct": "open"})
    monkeypatch.setattr(engine, "fill_closed_env_state", lambda _fc, _env: "open")
    monkeypatch.setattr(engine, "reject_is_requeueable",
                        lambda state, reason, stage, env=None: True)
    state = SimpleNamespace(fill_closed=fc)
    again, not_before = engine.fire_retry_decision(
        state, "batch_filled", 4, "opencodeinstruct", now=1000.0, stage=None)
    assert again is True
    assert not_before == pytest.approx(1001.5)
