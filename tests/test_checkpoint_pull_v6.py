"""Checkpoint sous v6 (rapport D §1.2) : chaque fenêtre pleine ouvre avec un
nouveau checkpoint, et le validateur peut republier la MÊME ``n`` avec une
révision nouvelle (rebind/reset) ou redescendre ``n`` sur un nouveau repo.

Bug corrigé : la boucle appelante ne posait ``_local_hash``/``hf_model`` que
si ``new_n != _local_n`` → même n + révision nouvelle = re-téléchargement à
chaque tour sans jamais recharger → ``checkpoint_hash`` périmé → 100 %
WRONG_CHECKPOINT. v5 inerte (n monotone, révision neuve à chaque n).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from reliquary.miner import engine


def _st(n, rev, repo="A"):
    return SimpleNamespace(checkpoint_n=n, checkpoint_revision=rev,
                           checkpoint_repo_id=repo)


def _eng(monkeypatch, *, n=5, h="old", repo="A"):
    eng = engine.MiningEngine.__new__(engine.MiningEngine)
    eng._local_n, eng._local_hash, eng.hf_model = n, h, object()
    eng._ckpt_repo_id = repo
    eng._pool, eng._pool_lock, eng._pool_dir = [], asyncio.Lock(), None
    calls: list = []

    async def _dl(repo_id, revision):
        calls.append(("dl", repo_id, revision))
        return f"/snap/{repo_id}/{revision}"

    monkeypatch.setattr(engine, "_hf_download", _dl)
    new_model = object()
    eng._load_checkpoint = lambda path: (calls.append(("load", path)), new_model)[1]
    monkeypatch.delenv("RELIQUARY_DROP_POOL_ON_CKPT", raising=False)
    return eng, calls, new_model


def test_maybe_pull_ignore_checkpoint_n_none():
    async def _boom(*a):
        raise AssertionError("download_fn ne doit pas être appelé")
    out = asyncio.run(engine.maybe_pull_checkpoint(
        _st(None, "rev"), 5, "old", "model",
        download_fn=_boom, load_fn=lambda p: pytest.fail("load"),
    ))
    assert out == (5, "old", "model")


def test_pull_applique_meme_n_revision_nouvelle(monkeypatch):
    eng, calls, new_model = _eng(monkeypatch, n=5, h="old")
    advanced = asyncio.run(eng._apply_checkpoint_pull(_st(5, "new")))
    assert advanced is True
    assert eng._local_n == 5 and eng._local_hash == "new"
    assert eng.hf_model is new_model
    assert calls == [("dl", "A", "new"), ("load", "/snap/A/new")]
    # idempotent : le tour suivant ne recharge pas
    assert asyncio.run(eng._apply_checkpoint_pull(_st(5, "new"))) is False
    assert len(calls) == 2


def test_pull_applique_n_qui_redescend_nouveau_repo(monkeypatch):
    eng, calls, new_model = _eng(monkeypatch, n=3, h="old", repo="A")
    advanced = asyncio.run(eng._apply_checkpoint_pull(_st(1, "r1", repo="B")))
    assert advanced is True
    assert eng._ckpt_repo_id == "B"
    assert (eng._local_n, eng._local_hash) == (1, "r1")
    assert eng.hf_model is new_model
    assert calls == [("dl", "B", "r1"), ("load", "/snap/B/r1")]


def test_pull_v5_inchange(monkeypatch):
    # n monotone + révision neuve → reload (comme avant) ; identique → rien
    eng, calls, new_model = _eng(monkeypatch, n=5, h="old")
    assert asyncio.run(eng._apply_checkpoint_pull(_st(5, "old"))) is False
    assert calls == []
    assert asyncio.run(eng._apply_checkpoint_pull(_st(6, "r6"))) is True
    assert (eng._local_n, eng._local_hash) == (6, "r6") and eng.hf_model is new_model


def test_pull_drop_pool_on_ckpt(monkeypatch):
    eng, calls, _ = _eng(monkeypatch, n=5, h="old")
    monkeypatch.setenv("RELIQUARY_DROP_POOL_ON_CKPT", "1")
    eng._pool = [{"prompt_idx": 1}]
    assert asyncio.run(eng._apply_checkpoint_pull(_st(5, "new"))) is True
    assert eng._pool == []
