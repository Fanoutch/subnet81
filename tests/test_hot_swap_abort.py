"""Fix hot-swap 2026-08-18 : plus jamais de gel non borné.

Autopsie des gels du 15/08 (15:20 et 16:04, fenêtres perdues + watchdog) :
le checkpoint avance EN PLEINE fenêtre → le bake en vol ne s'avorte pas
(``should_abort`` ne flippe qu'au changement de randomness) et son driver
tient ``_VLLM_CALL_LOCK`` pendant toute sa durée ; la sonde du self-gate
appelait en plus ``generate`` SANS le verrou (step concurrent du moteur).

Contrat testé :
1. ``request_interrupt()`` lève un drapeau que la boucle multi-stream honore
   (avort engine + libération rapide du verrou) — testé au niveau flag.
2. ``reload_weights_inplace`` : lève le drapeau, prend le verrou avec TIMEOUT ;
   pas obtenu → False (repli rebuild, jamais de gel) ; obtenu → swap sous
   verrou, drapeau nettoyé, verrou relâché.
3. ``_hot_swap_self_gate`` : la sonde est bornée par un timeout → une sonde
   qui pend = FAIL en ≤ timeout, jamais un gel.
"""
import threading
import time
import types

import pytest

from reliquary.miner import vllm_backend as vb


class FakeLLM:
    def __init__(self):
        self.rpc_calls = []
        self.prefix_cache_resets = 0

    def collective_rpc(self, name, kwargs=None):
        self.rpc_calls.append((name, kwargs))

    # 25/08 : reload_weights_inplace purge le cache de préfixe après l'échange
    # (sûreté forced-seed) — la fixture doit l'exposer comme le vrai LLM.
    def reset_prefix_cache(self):
        self.prefix_cache_resets += 1


def _backend_with_fake_llm(monkeypatch):
    monkeypatch.setenv("RELIQUARY_HOT_SWAP", "1")
    be = vb.VLLMBackend(model_path="/tmp/old", gpu_id=0)
    be._llm = FakeLLM()
    return be


class TestReloadInplaceLockDiscipline:
    def test_flag_off_returns_false(self, monkeypatch):
        monkeypatch.setenv("RELIQUARY_HOT_SWAP", "0")
        be = vb.VLLMBackend(model_path="/tmp/old")
        be._llm = FakeLLM()
        assert be.reload_weights_inplace("/tmp/new") is False

    def test_no_engine_returns_false(self, monkeypatch):
        monkeypatch.setenv("RELIQUARY_HOT_SWAP", "1")
        be = vb.VLLMBackend(model_path="/tmp/old")
        assert be.reload_weights_inplace("/tmp/new") is False

    def test_stubborn_lock_times_out_without_freeze(self, monkeypatch):
        """Un driver qui ignore le drapeau ne doit coûter que ~timeout."""
        be = _backend_with_fake_llm(monkeypatch)
        release = threading.Event()

        def hog():
            with vb._VLLM_CALL_LOCK:
                release.wait(10)

        t = threading.Thread(target=hog, daemon=True)
        t.start()
        time.sleep(0.1)
        t0 = time.monotonic()
        ok = be.reload_weights_inplace("/tmp/new", lock_timeout_s=0.5)
        dt = time.monotonic() - t0
        release.set()
        assert ok is False
        assert dt < 3.0, f"a gelé {dt:.1f}s au lieu de ~0.5s"
        assert not be._interrupt.is_set(), "drapeau non nettoyé"
        assert be._llm.rpc_calls == []

    def test_cooperative_driver_releases_and_swap_happens(self, monkeypatch):
        """Un driver qui honore le drapeau libère le verrou → swap OK."""
        be = _backend_with_fake_llm(monkeypatch)

        def cooperative_driver():
            with vb._VLLM_CALL_LOCK:
                # simule la boucle : step jusqu'au drapeau
                while not be._interrupt.is_set():
                    time.sleep(0.02)

        t = threading.Thread(target=cooperative_driver, daemon=True)
        t.start()
        time.sleep(0.1)
        ok = be.reload_weights_inplace("/tmp/new", lock_timeout_s=5.0)
        assert ok is True
        assert be._llm.rpc_calls == [
            ("reload_weights", {"weights_path": "/tmp/new"})]
        assert be._model_path == "/tmp/new"
        assert not be._interrupt.is_set()
        # le verrou doit être relâché
        assert vb._VLLM_CALL_LOCK.acquire(timeout=1.0)
        vb._VLLM_CALL_LOCK.release()

    def test_rpc_failure_returns_false_and_releases_lock(self, monkeypatch):
        be = _backend_with_fake_llm(monkeypatch)

        def boom(name, kwargs=None):
            raise RuntimeError("swap KO")

        be._llm.collective_rpc = boom
        assert be.reload_weights_inplace("/tmp/new", lock_timeout_s=1.0) is False
        assert vb._VLLM_CALL_LOCK.acquire(timeout=1.0)
        vb._VLLM_CALL_LOCK.release()


class TestSelfGateTimeout:
    def _engine_self(self):
        from reliquary.miner.engine import MiningEngine
        return types.SimpleNamespace(
            _hot_swap_self_gate=MiningEngine._hot_swap_self_gate.__get__(
                types.SimpleNamespace(hf_model=None)))

    def test_hanging_probe_fails_within_timeout(self):
        from reliquary.miner import engine as eng

        class HangingBackend:
            def generate_forced_probe(self, *a, **k):
                time.sleep(30)

        dummy = types.SimpleNamespace(hf_model=None)
        gate = eng.MiningEngine._hot_swap_self_gate.__get__(dummy)
        t0 = time.monotonic()
        ok = gate(HangingBackend(), probe_timeout_s=0.5)
        assert ok is False
        assert time.monotonic() - t0 < 3.0

    def test_probe_exception_fails_cleanly(self):
        from reliquary.miner import engine as eng

        class BoomBackend:
            def generate_forced_probe(self, *a, **k):
                raise RuntimeError("probe KO")

        dummy = types.SimpleNamespace(hf_model=None)
        gate = eng.MiningEngine._hot_swap_self_gate.__get__(dummy)
        assert gate(BoomBackend(), probe_timeout_s=1.0) is False
