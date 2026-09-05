"""Exploration ε (2026-08-15) : les DERNIERS slots du bake tirent au hasard
pur (sans prédicteur) pour produire des labels NON BIAISÉS — remède au cercle
« le prior ne fait étiqueter que ce qu'il aime » (ses picks plafonnent à
~1,8k tok/rollout, le cap 16384 reste inutilisé, v4.3 hériterait du biais).
Les vedettes du sprint (premiers slots) restent 100 % prédicteur."""
import pytest


def test_last_slots_explore(monkeypatch):
    from reliquary.miner import engine as eng
    monkeypatch.setenv("RELIQUARY_EXPLORE_SLOTS", "3")
    # batch 16, exploration sur les slots 13, 14, 15 uniquement
    assert all(eng._use_predictor_for_slot(i, 16) for i in range(13))
    assert not any(eng._use_predictor_for_slot(i, 16) for i in (13, 14, 15))


def test_default_off(monkeypatch):
    from reliquary.miner import engine as eng
    monkeypatch.delenv("RELIQUARY_EXPLORE_SLOTS", raising=False)
    assert all(eng._use_predictor_for_slot(i, 16) for i in range(16))


def test_never_eats_the_sprint(monkeypatch):
    from reliquary.miner import engine as eng
    # exploration > batch : on garde au moins les 2 premiers slots au prior
    monkeypatch.setenv("RELIQUARY_EXPLORE_SLOTS", "99")
    assert eng._use_predictor_for_slot(0, 16)
    assert eng._use_predictor_for_slot(1, 16)
    assert not eng._use_predictor_for_slot(2, 16)


def test_garbage_env(monkeypatch):
    from reliquary.miner import engine as eng
    monkeypatch.setenv("RELIQUARY_EXPLORE_SLOTS", "abc")
    assert all(eng._use_predictor_for_slot(i, 16) for i in range(16))
