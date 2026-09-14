"""Préchargement du checkpoint AVANT l'ouverture de fenêtre (V1, 13/09).

Mesuré : le validateur publie le checkpoint de la fenêtre suivante sur HF
~138 s avant de l'ouvrir (commit ``fill_closed_boundary``), pendant que
``/state`` répond 503 (aucun batcher actif). Le mineur, lui, ne rechargeait
qu'au premier ``/state`` OPEN : 36-55 s de reconstruction vLLM DANS la
fenêtre, et 0 groupe retenu sur les fenêtres concernées (n=4 contre 7 et 3).

Le préchargement charge les POIDS de la révision préchargée pendant le trou
503. ``_local_hash`` (qui entre dans ``u_at`` et dans le ``checkpoint_hash``
signé) reste piloté par ``/state`` : on ne génère pas tant que poids et hash
divergent, et si la fenêtre ouvre sur une autre révision on recharge celle
de ``/state`` (coût = le chemin d'aujourd'hui).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from reliquary.miner import engine

OLD, NEW, OTHER = "a" * 40, "b" * 40, "c" * 40


# ------------------------------------------------------------ décision pure
def test_decision_precharge_apres_trou_suffisant():
    assert engine.preload_decision(
        prefetched_rev=NEW, local_hash=OLD, preloaded_rev=None,
        gap_s=6.0, min_gap_s=5.0) is True


def test_decision_pas_de_trou_pas_de_prechargement():
    assert engine.preload_decision(
        prefetched_rev=NEW, local_hash=OLD, preloaded_rev=None,
        gap_s=None, min_gap_s=5.0) is False
    assert engine.preload_decision(
        prefetched_rev=NEW, local_hash=OLD, preloaded_rev=None,
        gap_s=2.0, min_gap_s=5.0) is False


def test_decision_rien_a_precharger():
    assert engine.preload_decision(
        prefetched_rev=None, local_hash=OLD, preloaded_rev=None,
        gap_s=60.0, min_gap_s=5.0) is False
    assert engine.preload_decision(
        prefetched_rev=OLD, local_hash=OLD, preloaded_rev=None,
        gap_s=60.0, min_gap_s=5.0) is False
    assert engine.preload_decision(
        prefetched_rev=NEW, local_hash=OLD, preloaded_rev=NEW,
        gap_s=60.0, min_gap_s=5.0) is False


# ------------------------------------------------------------ moteur
def _eng(monkeypatch, *, local_hash=OLD):
    eng = engine.MiningEngine.__new__(engine.MiningEngine)
    eng._local_n, eng._local_hash, eng.hf_model = 5, local_hash, object()
    eng._ckpt_repo_id = "R"
    eng._pool, eng._pool_lock, eng._pool_dir = [{"prompt_idx": 1}], asyncio.Lock(), None
    calls: list = []

    async def _dl(repo_id, revision):
        calls.append(("dl", revision))
        return f"/snap/{revision}"

    monkeypatch.setattr(engine, "_hf_download", _dl)

    def _load(path):
        calls.append(("load", path))
        eng._loaded_checkpoint_path = path
        return eng.hf_model

    eng._load_checkpoint = _load
    eng._prefetched_paths = {NEW: f"/snap/{NEW}"}
    eng._prefetched_latest = NEW
    monkeypatch.setenv("RELIQUARY_CHECKPOINT_PRELOAD", "1")
    monkeypatch.delenv("RELIQUARY_DROP_POOL_ON_CKPT", raising=False)
    return eng, calls


def _st(rev, n=6):
    return SimpleNamespace(checkpoint_n=n, checkpoint_revision=rev,
                           checkpoint_repo_id="R")


def test_prechargement_charge_les_poids_sans_toucher_au_hash(monkeypatch):
    eng, calls = _eng(monkeypatch)
    assert asyncio.run(eng._maybe_preload_checkpoint(gap_s=10.0)) is True
    assert calls == [("load", f"/snap/{NEW}")]
    assert eng._local_hash == OLD
    assert eng._preloaded_rev == NEW
    assert eng._pool == []                 # entrées de l'ancien modèle jetées
    assert eng._weights_ahead_of_hash() is True
    # pas deux fois
    assert asyncio.run(eng._maybe_preload_checkpoint(gap_s=20.0)) is False


def test_prechargement_desactive_par_defaut(monkeypatch):
    eng, calls = _eng(monkeypatch)
    monkeypatch.delenv("RELIQUARY_CHECKPOINT_PRELOAD", raising=False)
    assert asyncio.run(eng._maybe_preload_checkpoint(gap_s=10.0)) is False
    assert calls == []


def test_ouverture_sur_la_revision_prechargee(monkeypatch):
    eng, calls = _eng(monkeypatch)
    asyncio.run(eng._maybe_preload_checkpoint(gap_s=10.0))
    calls.clear()
    assert asyncio.run(eng._apply_checkpoint_pull(_st(NEW))) is True
    assert eng._local_hash == NEW
    assert eng._preloaded_rev is None
    assert eng._weights_ahead_of_hash() is False
    assert calls == [("dl", NEW), ("load", f"/snap/{NEW}")]


def test_ouverture_sur_lancienne_revision_recharge_celle_de_state(monkeypatch):
    eng, calls = _eng(monkeypatch)
    asyncio.run(eng._maybe_preload_checkpoint(gap_s=10.0))
    calls.clear()
    asyncio.run(eng._apply_checkpoint_pull(_st(OLD, n=5)))
    assert calls == [("dl", OLD), ("load", f"/snap/{OLD}")]
    assert eng._local_hash == OLD
    assert eng._preloaded_rev is None
    assert eng._weights_ahead_of_hash() is False


def test_ouverture_sur_une_troisieme_revision(monkeypatch):
    eng, calls = _eng(monkeypatch)
    asyncio.run(eng._maybe_preload_checkpoint(gap_s=10.0))
    calls.clear()
    assert asyncio.run(eng._apply_checkpoint_pull(_st(OTHER, n=7))) is True
    assert calls == [("dl", OTHER), ("load", f"/snap/{OTHER}")]
    assert eng._local_hash == OTHER
    assert eng._preloaded_rev is None


def test_sans_prechargement_pas_de_garde(monkeypatch):
    eng, _ = _eng(monkeypatch)
    assert eng._weights_ahead_of_hash() is False


def test_echec_de_prechargement_bloque_la_generation_jusquau_flip(monkeypatch):
    eng, calls = _eng(monkeypatch)

    def _load_fail(path):
        calls.append(("load", path))
        eng._loaded_checkpoint_path = None     # vLLM KO, modèle de preuve échangé
        return eng.hf_model

    eng._load_checkpoint = _load_fail
    assert asyncio.run(eng._maybe_preload_checkpoint(gap_s=10.0)) is False
    assert eng._weights_ahead_of_hash() is True
    eng._load_checkpoint = lambda p: (calls.append(("load", p)),
                                      setattr(eng, "_loaded_checkpoint_path", p))[0] or eng.hf_model
    calls.clear()
    asyncio.run(eng._apply_checkpoint_pull(_st(OLD, n=5)))
    assert calls == [("dl", OLD), ("load", f"/snap/{OLD}")]
    assert eng._weights_ahead_of_hash() is False
