"""Fix OOM reload 2026-08-16 : kill des EngineCore zombies.

Incident 13:24-13:38 : au reload de checkpoint, l'ancien EngineCore a gardé
115 GiB pendant ~4 min ; 4 tentatives _build_llm en init-OOM, et les échecs
répétés ont gonflé le processus principal (23→27,5 GiB) → toutes les preuves
GRAIL en OOM ensuite (revenu zéro, watchdog aveugle).

Contrat testé ici :
1. `_kill_stale_engine_cores()` tue les enfants directs dont la cmdline
   contient "EngineCore", épargne tout le reste, ne lève jamais.
2. `_ensure_loaded()` l'appelle à partir du 2e échec init-OOM (le 1er échec
   laisse une chance à la mort naturelle de l'ancien moteur).
"""
import os
import signal
import subprocess
import time

import pytest

from reliquary.miner import vllm_backend as vb


def _spawn(argv0: str) -> subprocess.Popen:
    """Enfant direct dont argv[0] est contrôlé (simule le setproctitle vLLM)."""
    return subprocess.Popen(
        ["bash", "-c", f'exec -a "{argv0}" sleep 30'],
    )


def _wait_dead(proc: subprocess.Popen, timeout: float = 6.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return True
        time.sleep(0.1)
    return False


class TestKillStaleEngineCores:
    def test_kills_child_named_enginecore(self):
        child = _spawn("VLLM::EngineCore")
        try:
            time.sleep(0.3)  # laisse exec -a s'appliquer
            killed = vb._kill_stale_engine_cores(grace_s=0.5)
            assert killed >= 1
            assert _wait_dead(child), "l'EngineCore zombie doit être mort"
        finally:
            if child.poll() is None:
                child.kill()
            child.wait()

    def test_spares_unrelated_children(self):
        bystander = _spawn("innocent-worker")
        try:
            time.sleep(0.3)
            vb._kill_stale_engine_cores(grace_s=0.5)
            time.sleep(0.5)
            assert bystander.poll() is None, "un enfant non-EngineCore doit survivre"
        finally:
            if bystander.poll() is None:
                bystander.kill()
            bystander.wait()

    def test_zero_when_no_matching_children(self):
        assert vb._kill_stale_engine_cores(grace_s=0.1) == 0

    def test_never_raises_on_proc_errors(self, monkeypatch):
        monkeypatch.setattr(vb.os, "listdir", lambda *_: (_ for _ in ()).throw(OSError("boom")))
        assert vb._kill_stale_engine_cores(grace_s=0.1) == 0


class TestEnsureLoadedCallsKiller:
    def _backend(self):
        return vb.VLLMBackend(model_path="/tmp/fake", gpu_id=0)

    def test_killer_called_from_second_oom_failure(self, monkeypatch):
        calls = []
        monkeypatch.setattr(vb, "_kill_stale_engine_cores",
                            lambda *a, **k: calls.append(1) or 0)
        monkeypatch.setattr(
            vb, "_build_llm",
            lambda **kw: (_ for _ in ()).throw(
                ValueError("Free memory on device cuda:0 (8.8/139.8 GiB) ...")),
        )
        monkeypatch.setattr(time, "sleep", lambda *_: None)
        be = self._backend()
        with pytest.raises(ValueError):
            be._ensure_loaded()
        # 5 tentatives ; le killer entre en jeu aux échecs 2,3,4,5 → 4 appels.
        assert len(calls) == 4

    def test_killer_not_called_on_first_failure_only(self, monkeypatch):
        """Succès à la tentative 2 → mort naturelle privilégiée, 0 kill."""
        calls = []
        attempts = {"n": 0}

        def flaky(**kw):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise ValueError("Free memory on device cuda:0 ...")
            return object()

        monkeypatch.setattr(vb, "_kill_stale_engine_cores",
                            lambda *a, **k: calls.append(1) or 0)
        monkeypatch.setattr(vb, "_build_llm", flaky)
        monkeypatch.setattr(time, "sleep", lambda *_: None)
        be = self._backend()
        be._ensure_loaded()
        assert be._llm is not None
        assert calls == []

    def test_non_oom_error_fails_fast_without_kill(self, monkeypatch):
        calls = []
        monkeypatch.setattr(vb, "_kill_stale_engine_cores",
                            lambda *a, **k: calls.append(1) or 0)
        monkeypatch.setattr(
            vb, "_build_llm",
            lambda **kw: (_ for _ in ()).throw(ValueError("bad config")),
        )
        be = self._backend()
        with pytest.raises(ValueError):
            be._ensure_loaded()
        assert calls == []
