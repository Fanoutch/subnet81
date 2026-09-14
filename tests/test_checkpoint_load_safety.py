"""Échec de chargement d'un checkpoint : ne JAMAIS signer N+1 avec les poids N.

Vécu le 14/09 (13:19 → 15:01) : 8 ``Failed to reload hf_model`` (OOM, 31,9 Mio
libres) à chaque bascule. ``_load_checkpoint`` gardait l'ancien modèle et ne
rechargeait pas vLLM, mais ``maybe_pull_checkpoint`` renvoyait quand même la
nouvelle révision → ``_local_hash`` (entrée de ``u_at`` et hash signé) passait à
N+1 pendant que génération, preuve et réplique tournaient sur les poids N.

Invariants :
- chargement raté → ``_local_hash`` inchangé, génération et tirs suspendus ;
- nouvel essai borné dans le temps (pas un rechargement à chaque tour) ;
- chargement réussi ensuite → révision adoptée, suspension levée ;
- le modèle de preuve neuf n'est monté sur le GPU qu'APRÈS avoir libéré
  l'ancien (pas deux modèles en VRAM) ; en cas d'échec l'ancien revient.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from reliquary.miner import engine

OLD, NEW = "a" * 40, "b" * 40


def _st(rev, n=6):
    return SimpleNamespace(checkpoint_n=n, checkpoint_revision=rev,
                           checkpoint_repo_id="R")


def _eng(monkeypatch, *, load_ok=False):
    eng = engine.MiningEngine.__new__(engine.MiningEngine)
    eng._local_n, eng._local_hash, eng.hf_model = 5, OLD, object()
    eng._loaded_checkpoint_path = f"/snap/{OLD}"
    eng._pool, eng._pool_lock, eng._pool_dir = [], asyncio.Lock(), None
    calls: list = []

    async def _dl(repo_id, revision):
        return f"/snap/{revision}"

    monkeypatch.setattr(engine, "_hf_download", _dl)

    def _load(path):
        calls.append(path)
        if eng._load_ok:
            eng._loaded_checkpoint_path = path
        return eng.hf_model

    eng._load_ok = load_ok
    eng._load_checkpoint = _load
    monkeypatch.delenv("RELIQUARY_DROP_POOL_ON_CKPT", raising=False)
    return eng, calls


def test_maybe_pull_ne_rend_pas_la_revision_si_le_chargement_echoue():
    model = object()

    async def dl(repo, rev):
        return f"/snap/{rev}"

    out = asyncio.run(engine.maybe_pull_checkpoint(
        _st(NEW), 5, OLD, model, download_fn=dl, load_fn=lambda p: model,
        loaded_fn=lambda p: False))
    assert out == (5, OLD, model)


def test_echec_de_chargement_garde_le_hash_et_suspend(monkeypatch):
    eng, calls = _eng(monkeypatch)
    assert asyncio.run(eng._apply_checkpoint_pull(_st(NEW))) is False
    assert calls == [f"/snap/{NEW}"]
    assert eng._local_hash == OLD
    assert eng._ckpt_load_failed_rev == NEW
    assert eng._weights_out_of_sync() is True
    assert eng._ckpt_fire_blocked() is True


def test_nouvel_essai_borne_dans_le_temps(monkeypatch):
    eng, calls = _eng(monkeypatch)
    t = {"now": 1000.0}
    monkeypatch.setattr(engine.time, "monotonic", lambda: t["now"])
    asyncio.run(eng._apply_checkpoint_pull(_st(NEW)))
    asyncio.run(eng._apply_checkpoint_pull(_st(NEW)))      # trop tôt
    assert len(calls) == 1
    t["now"] += 25.0
    asyncio.run(eng._apply_checkpoint_pull(_st(NEW)))
    assert len(calls) == 2


def test_reussite_apres_echec_leve_la_suspension(monkeypatch):
    eng, calls = _eng(monkeypatch)
    t = {"now": 1000.0}
    monkeypatch.setattr(engine.time, "monotonic", lambda: t["now"])
    asyncio.run(eng._apply_checkpoint_pull(_st(NEW)))
    eng._load_ok = True
    t["now"] += 25.0
    assert asyncio.run(eng._apply_checkpoint_pull(_st(NEW))) is True
    assert eng._local_hash == NEW
    assert eng._ckpt_load_failed_rev is None
    assert eng._weights_out_of_sync() is False
    assert eng._ckpt_fire_blocked() is False


def test_reussite_directe_inchangee(monkeypatch):
    eng, calls = _eng(monkeypatch, load_ok=True)
    assert asyncio.run(eng._apply_checkpoint_pull(_st(NEW))) is True
    assert eng._local_hash == NEW
    assert getattr(eng, "_ckpt_load_failed_rev", None) is None


def test_revision_de_state_redevenue_la_notre_leve_la_suspension(monkeypatch):
    eng, _ = _eng(monkeypatch)
    asyncio.run(eng._apply_checkpoint_pull(_st(NEW)))
    assert eng._ckpt_fire_blocked() is True
    asyncio.run(eng._apply_checkpoint_pull(_st(OLD, n=5)))
    assert eng._ckpt_load_failed_rev is None


# ------------------------------------------- ordre mémoire du modèle de preuve
class _FakeModel:
    def __init__(self, name, log, fail_cuda=False):
        self.name, self.log, self.fail_cuda = name, log, fail_cuda
        self.device = "cpu"

    def to(self, device):
        if str(device).startswith("cuda") and self.fail_cuda:
            self.log.append((self.name, "to", str(device), "OOM"))
            raise RuntimeError("CUDA out of memory")
        self.log.append((self.name, "to", str(device)))
        self.device = str(device)
        return self

    def eval(self):
        return self


def _load_engine(monkeypatch, *, fail_cuda):
    log: list = []
    eng = engine.MiningEngine.__new__(engine.MiningEngine)
    eng.proof_gpu = 0
    old = _FakeModel("old", log)
    old.device = "cuda:0"
    eng.hf_model = old
    eng._loaded_checkpoint_path = "/snap/old"
    eng._vllm_backend = None
    eng.vllm_gpu = 0
    eng.vllm_model = None
    new = _FakeModel("new", log, fail_cuda=fail_cuda)
    monkeypatch.setattr(engine, "load_text_generation_model",
                        lambda path, **kw: (log.append(("new", "load", kw.get("device_map"))), new)[1])
    eng._resolve_eos_ids = lambda: [1]
    return eng, old, new, log


def test_modele_de_preuve_libere_avant_de_monter_le_neuf(monkeypatch):
    eng, old, new, log = _load_engine(monkeypatch, fail_cuda=False)
    try:
        eng._load_checkpoint("/snap/new")
    except Exception:
        pass                         # la suite (générateur HF factice) peut lever
    assert eng.hf_model is new
    i_old_cpu = log.index(("old", "to", "cpu"))
    i_new_cuda = log.index(("new", "to", "cuda:0"))
    assert i_old_cpu < i_new_cuda


def test_echec_gpu_du_neuf_remet_lancien(monkeypatch):
    eng, old, new, log = _load_engine(monkeypatch, fail_cuda=True)
    out = eng._load_checkpoint("/snap/new")
    assert out is old and eng.hf_model is old
    assert old.device == "cuda:0"
    assert eng._loaded_checkpoint_path == "/snap/old"
