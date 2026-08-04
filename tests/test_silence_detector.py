"""Détecteur de panne SILENCIEUSE : « tout est jeté, rien n'est soumis ».

Leçon du 2026-08-03/04 : deux pannes ont coûté des heures parce qu'elles ne
disaient RIEN.
  * le parse `/state` v3 échouait → le mineur pollait en boucle sans générer ;
  * le fix de terminaison, s'il tombe dans le mauvais cas (vLLM retire le token
    d'arrêt), ferait jeter 100% des groupes — zéro `bad_termination`, mais
    zéro soumission. Silence au lieu d'erreur.

Un taux de rejet local de ~100% sur un échantillon suffisant n'est jamais
normal : ça doit CRIER dans les logs, pas se fondre dans des INFO par prompt.
"""
import pytest

from reliquary.miner.engine import DropTracker


def test_no_alert_before_minimum_sample():
    t = DropTracker(min_sample=10, alert_ratio=0.9)
    for _ in range(9):
        assert t.record(dropped=True, reason="termination") is None


def test_alerts_when_almost_everything_is_dropped():
    t = DropTracker(min_sample=10, alert_ratio=0.9)
    alert = None
    for _ in range(10):
        alert = t.record(dropped=True, reason="termination")
    assert alert is not None
    assert "termination" in alert
    assert "10/10" in alert


def test_no_alert_when_drops_are_normal():
    """Jeter beaucoup pour σ hors zone est le régime NORMAL (~99%) — seul un
    motif dominé par une cause TECHNIQUE doit alerter."""
    t = DropTracker(min_sample=10, alert_ratio=0.9)
    for _ in range(10):
        assert t.record(dropped=True, reason="out_of_zone") is None


def test_alert_fires_once_then_rearms_after_window():
    t = DropTracker(min_sample=4, alert_ratio=0.9)
    for _ in range(4):
        first = t.record(dropped=True, reason="termination")
    assert first is not None
    # la fenêtre est repartie à zéro : pas de spam à chaque groupe suivant
    assert t.record(dropped=True, reason="termination") is None


def test_submission_resets_the_alarm():
    t = DropTracker(min_sample=4, alert_ratio=0.9)
    t.record(dropped=True, reason="termination")
    t.record(dropped=True, reason="termination")
    t.record(dropped=False, reason=None)          # une soumission part
    t.record(dropped=True, reason="termination")
    assert t.record(dropped=True, reason="termination") is None
