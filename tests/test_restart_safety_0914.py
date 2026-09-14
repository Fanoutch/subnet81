"""Correctifs du 14/09 soir : redémarrage et rechargement sans fenêtre perdue.

1. Dépôt du checkpoint connu dès le démarrage : ``_ckpt_repo_id`` n'était posé
   qu'au premier ``/state`` 200 → un redémarrage pendant le trou 503 ne
   préchargeait rien → rechargement DANS la fenêtre (1er groupe à +60-90 s).
2. Garde mémoire avant de construire vLLM : à 16:12 l'ancien EngineCore (non
   référencé) tenait encore la VRAM → 2 init-OOM, ~45 s perdues.
3. Budget de réparation 2048 (512 jetait ~34 % des groupes en zone).
4. ``miner.log`` conservé au redémarrage (logs de la 45899 perdus).
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from reliquary.miner import engine
from reliquary.miner import vllm_backend as vb

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------ 1. dépôt du checkpoint
def _bare():
    return engine.MiningEngine.__new__(engine.MiningEngine)


def test_depot_connu_via_fichier_au_demarrage(tmp_path, monkeypatch):
    f = tmp_path / "repo"
    f.write_text("ReliquaryForge/qwen3-4b-base-dapo-v4\n")
    monkeypatch.setenv("RELIQUARY_CKPT_REPO_FILE", str(f))
    monkeypatch.delenv("RELIQUARY_CHECKPOINT_REPO_DEFAULT", raising=False)
    assert _bare()._active_ckpt_repo() == "ReliquaryForge/qwen3-4b-base-dapo-v4"


def test_depot_par_defaut_du_launcher(tmp_path, monkeypatch):
    monkeypatch.setenv("RELIQUARY_CKPT_REPO_FILE", str(tmp_path / "absent"))
    monkeypatch.setenv("RELIQUARY_CHECKPOINT_REPO_DEFAULT", "R/def")
    assert _bare()._active_ckpt_repo() == "R/def"


def test_state_prime_et_est_memorise(tmp_path, monkeypatch):
    import asyncio
    from types import SimpleNamespace

    f = tmp_path / "repo"
    monkeypatch.setenv("RELIQUARY_CKPT_REPO_FILE", str(f))
    monkeypatch.setenv("RELIQUARY_CHECKPOINT_REPO_DEFAULT", "R/def")
    eng = _bare()
    eng._local_n, eng._local_hash, eng.hf_model = 5, "a" * 40, object()
    eng._pool, eng._pool_lock, eng._pool_dir = [], asyncio.Lock(), None
    st = SimpleNamespace(checkpoint_n=5, checkpoint_revision="a" * 40,
                         checkpoint_repo_id="R/live")
    asyncio.run(eng._apply_checkpoint_pull(st))
    assert eng._active_ckpt_repo() == "R/live"
    assert f.read_text().strip() == "R/live"
    assert _bare()._active_ckpt_repo() == "R/live"      # après redémarrage


def test_sans_rien_pas_de_depot(tmp_path, monkeypatch):
    monkeypatch.setenv("RELIQUARY_CKPT_REPO_FILE", str(tmp_path / "absent"))
    monkeypatch.delenv("RELIQUARY_CHECKPOINT_REPO_DEFAULT", raising=False)
    assert _bare()._active_ckpt_repo() is None


# ------------------------------------------------ 2. garde mémoire vLLM
GIB = 1024 ** 3


def _backend(monkeypatch, frees, *, util=0.70, total=139.8):
    be = vb.VLLMBackend(model_path="/tmp/fake", gpu_id=0,
                        gpu_memory_utilization=util)
    seq = list(frees)
    monkeypatch.setattr(be, "_vram_free_total",
                        lambda: (seq.pop(0) if len(seq) > 1 else seq[0], total))
    kills = []
    monkeypatch.setattr(vb, "_kill_stale_engine_cores",
                        lambda *a, **k: kills.append(1) or 1)
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    return be, kills


def test_garde_memoire_tue_lorphelin_et_attend(monkeypatch):
    be, kills = _backend(monkeypatch, [33.0, 60.0, 120.0])
    built = []
    monkeypatch.setattr(vb, "_build_llm", lambda **kw: built.append(1) or object())
    be._ensure_loaded()
    assert kills == [1]
    assert built == [1]                      # une seule construction, sans OOM


def test_garde_memoire_rien_si_assez_de_place(monkeypatch):
    be, kills = _backend(monkeypatch, [121.0])
    monkeypatch.setattr(vb, "_build_llm", lambda **kw: object())
    be._ensure_loaded()
    assert kills == []


def test_garde_memoire_sans_gpu_inerte(monkeypatch):
    be = vb.VLLMBackend(model_path="/tmp/fake", gpu_id=0)
    monkeypatch.setattr(be, "_vram_free_total", lambda: None)
    kills = []
    monkeypatch.setattr(vb, "_kill_stale_engine_cores",
                        lambda *a, **k: kills.append(1) or 0)
    monkeypatch.setattr(vb, "_build_llm", lambda **kw: object())
    be._ensure_loaded()
    assert kills == []


# ------------------------------------------------ 3. budget de réparation
def test_launcher_budget_reparation_2048():
    txt = (ROOT / "ops" / "launch_miner_v4.sh").read_text()
    assert "RELIQUARY_TERMINAL_REPAIR_MAX_NEW=${RELIQUARY_TERMINAL_REPAIR_MAX_NEW:-2048}" in txt


def test_launcher_depot_checkpoint_par_defaut():
    txt = (ROOT / "ops" / "launch_miner_v4.sh").read_text()
    assert "RELIQUARY_CHECKPOINT_REPO_DEFAULT=" in txt


# ------------------------------------------------ 4. journal conservé
def test_restart_conserve_le_journal():
    txt = (ROOT / "ops" / "restart_miner.sh").read_text()
    i_rot = txt.index("miner.log.")
    i_tee = txt.index('tee /workspace/miner.log"')
    assert i_rot < i_tee
