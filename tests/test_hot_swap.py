"""Hot-swap des poids au checkpoint-advance (2026-08-15).

Rebuild complet ~150 s = fenêtre perdue quand la collecte chevauche le reload
(28964 servie par d'autres mineurs). Même architecture → échange en place via
Worker.reload_weights (~5-15 s), gardé par un auto-gate forced-seed."""
import sys
import pytest


def _backend():
    from reliquary.miner.vllm_backend import VLLMBackend
    return VLLMBackend("old-path", forced_seed=True)


def test_flag_off_returns_false(monkeypatch):
    monkeypatch.delenv("RELIQUARY_HOT_SWAP", raising=False)
    b = _backend()
    b._llm = object()
    assert b.reload_weights_inplace("new-path") is False


def test_no_engine_returns_false(monkeypatch):
    monkeypatch.setenv("RELIQUARY_HOT_SWAP", "1")
    b = _backend()
    assert b.reload_weights_inplace("new-path") is False


def test_success_calls_rpc_and_updates_path(monkeypatch):
    monkeypatch.setenv("RELIQUARY_HOT_SWAP", "1")
    b = _backend()
    calls = []

    class _LLM:
        def collective_rpc(self, method, kwargs=None):
            calls.append((method, kwargs))
    b._llm = _LLM()
    assert b.reload_weights_inplace("new-path") is True
    assert calls == [("reload_weights", {"weights_path": "new-path"})]
    assert b._model_path == "new-path"


def test_rpc_failure_returns_false_keeps_path(monkeypatch):
    monkeypatch.setenv("RELIQUARY_HOT_SWAP", "1")
    b = _backend()

    class _LLM:
        def collective_rpc(self, method, kwargs=None):
            raise RuntimeError("boom")
    b._llm = _LLM()
    assert b.reload_weights_inplace("new-path") is False
    assert b._model_path == "old-path"
