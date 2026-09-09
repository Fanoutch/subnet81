"""L'échange de poids à chaud DOIT purger le cache de préfixe.

``enable_prefix_caching=True`` est le défaut de vLLM 0.24 et notre moteur
tourne bien avec (dump de config du moteur, 25/08).
``GPUModelRunner.reload_weights`` remet à zéro le cache encodeur et le cache
multimodal — mais PAS le cache de préfixe. Des blocs KV calculés sous les
ANCIENS poids survivraient donc à l'échange et seraient réutilisés par le
nouveau modèle.

Sous forced-seed c'est une faute FATALE : chaque token est imposé par CDF
publique et le validateur reconstitue la séquence en teacher-forcing avec les
nouveaux poids. Des hidden states issus d'anciens poids décalent les logits
aux frontières de CDF => SEED_MISMATCH en masse.

Le rebuild complet n'a jamais eu ce défaut : il détruit le processus
EngineCore, donc le cache avec. Le chemin à chaud, lui, garde le moteur
vivant — d'où ce test.
"""
from __future__ import annotations

import threading

from reliquary.miner import vllm_backend


class _FakeLLM:
    def __init__(self):
        self.calls: list[str] = []

    def collective_rpc(self, method, kwargs=None):
        self.calls.append(f"rpc:{method}")

    def reset_prefix_cache(self):
        self.calls.append("reset_prefix_cache")


def _backend_with(llm):
    b = vllm_backend.VLLMBackend.__new__(vllm_backend.VLLMBackend)
    b._llm = llm
    b._model_path = "/ancien/snapshot"
    b._interrupt = threading.Event()
    return b


def test_prefix_cache_is_purged_after_the_swap(monkeypatch):
    monkeypatch.setenv("RELIQUARY_HOT_SWAP", "1")
    llm = _FakeLLM()
    b = _backend_with(llm)

    assert b.reload_weights_inplace("/nouveau/snapshot") is True

    assert "reset_prefix_cache" in llm.calls, (
        "cache de préfixe NON purgé : des blocs KV calculés sous les anciens "
        "poids seraient réutilisés => SEED_MISMATCH"
    )
    # L'ordre compte : purger AVANT l'échange laisserait le moteur re-cacher
    # avec les anciens poids pendant l'échange.
    assert llm.calls.index("rpc:reload_weights") < llm.calls.index(
        "reset_prefix_cache"
    )
    assert b._model_path == "/nouveau/snapshot"


def test_disarmed_by_default(monkeypatch):
    """Repli : sans RELIQUARY_HOT_SWAP=1, rien n'est touché et l'appelant
    fait le rebuild complet, l'état d'avant."""
    monkeypatch.delenv("RELIQUARY_HOT_SWAP", raising=False)
    llm = _FakeLLM()
    b = _backend_with(llm)

    assert b.reload_weights_inplace("/nouveau/snapshot") is False
    assert llm.calls == []
    assert b._model_path == "/ancien/snapshot"


def test_swap_failure_falls_back_and_keeps_the_old_path(monkeypatch):
    """Toute exception => False => rebuild complet. Le chemin du modèle ne
    doit PAS avoir bougé, sinon le backend mentirait sur ce qu'il sert."""
    monkeypatch.setenv("RELIQUARY_HOT_SWAP", "1")

    class _Boom(_FakeLLM):
        def collective_rpc(self, method, kwargs=None):
            raise RuntimeError("reload_weights KO")

    llm = _Boom()
    b = _backend_with(llm)

    assert b.reload_weights_inplace("/nouveau/snapshot") is False
    assert b._model_path == "/ancien/snapshot"


def test_prefix_cache_failure_is_not_swallowed(monkeypatch):
    """Si la purge échoue, l'échange NE DOIT PAS être annoncé réussi : mieux
    vaut un rebuild complet qu'un moteur qui sert des KV périmés."""
    monkeypatch.setenv("RELIQUARY_HOT_SWAP", "1")

    class _NoReset(_FakeLLM):
        def reset_prefix_cache(self):
            raise RuntimeError("purge KO")

    llm = _NoReset()
    b = _backend_with(llm)

    assert b.reload_weights_inplace("/nouveau/snapshot") is False
