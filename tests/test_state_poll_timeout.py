"""Timeout du poll /state configurable (08/09 : /state affamé par le préchargement HF)."""
from __future__ import annotations

from reliquary.miner import engine


def test_defaut_3s(monkeypatch):
    monkeypatch.delenv("RELIQUARY_STATE_POLL_TIMEOUT_S", raising=False)
    assert engine._state_poll_timeout_s() == 3.0


def test_env_lu(monkeypatch):
    monkeypatch.setenv("RELIQUARY_STATE_POLL_TIMEOUT_S", "8")
    assert engine._state_poll_timeout_s() == 8.0


def test_plancher_et_valeur_illisible(monkeypatch):
    monkeypatch.setenv("RELIQUARY_STATE_POLL_TIMEOUT_S", "0.2")
    assert engine._state_poll_timeout_s() == 1.0
    monkeypatch.setenv("RELIQUARY_STATE_POLL_TIMEOUT_S", "abc")
    assert engine._state_poll_timeout_s() == 3.0


def test_les_deux_polls_utilisent_le_timeout_configurable():
    import inspect
    src = inspect.getsource(engine)
    assert "timeout=3.0" not in src
    assert src.count("timeout=_state_poll_timeout_s()") >= 2
