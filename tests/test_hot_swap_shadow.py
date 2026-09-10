"""Mode OMBRE du hot-swap + paramétrage de la gate (2026-09-10).

Sous V1/fill-closed le rechargement de checkpoint tombe sur ~85 % des
fenêtres et gèle la boucle 41-43 s, dont **36-37 s de reconstruction vLLM**
(mesuré sur 11 avancées, 10/09) — pile pendant les ~100 s où la porte du
batch code se remplit. Le hot-swap vise ces 36 s.

Il est resté NO-GO pour une raison qui tient toujours : la gate
``_hot_swap_self_gate`` compare 48 picks vLLM contre le teacher-forcing HF
avec un plancher 0,80 **jamais recalibré** depuis v4. On ne peut pas exiger
des ids identiques (vLLM ≠ HF numériquement : 0,9793 groupe / 0,9231 pire
rollout à la gate de conformité du 06/08), donc le plancher doit être
CHOISI SUR MESURE, pas deviné.

D'où le mode ``shadow`` : on échange, on mesure la gate, on JOURNALISE, puis
on reconstruit quand même. Le moteur qui sert la fenêtre est donc exactement
celui d'aujourd'hui — zéro risque de conformité — et on récolte la
distribution du taux sur un moteur sain pour fixer le plancher.
"""
import pytest

from reliquary.miner.hot_swap_policy import (
    hot_swap_mode,
    hot_swap_decision,
    hot_swap_gate_params,
)


class TestMode:
    def test_absent_is_off(self, monkeypatch):
        monkeypatch.delenv("RELIQUARY_HOT_SWAP", raising=False)
        assert hot_swap_mode() == "off"

    def test_zero_is_off(self, monkeypatch):
        monkeypatch.setenv("RELIQUARY_HOT_SWAP", "0")
        assert hot_swap_mode() == "off"

    def test_one_is_armed(self, monkeypatch):
        monkeypatch.setenv("RELIQUARY_HOT_SWAP", "1")
        assert hot_swap_mode() == "armed"

    def test_shadow_is_shadow(self, monkeypatch):
        monkeypatch.setenv("RELIQUARY_HOT_SWAP", "shadow")
        assert hot_swap_mode() == "shadow"

    def test_unknown_value_is_off(self, monkeypatch):
        """Une faute de frappe ne doit JAMAIS armer un chemin de conformité."""
        monkeypatch.setenv("RELIQUARY_HOT_SWAP", "yes")
        assert hot_swap_mode() == "off"


class TestDecision:
    def test_armed_and_gate_pass_keeps_the_swapped_engine(self):
        assert hot_swap_decision("armed", swapped=True, gate_ok=True) == "keep"

    def test_armed_and_gate_fail_rebuilds(self):
        assert hot_swap_decision("armed", swapped=True, gate_ok=False) == "rebuild"

    def test_swap_that_did_not_happen_rebuilds(self):
        assert hot_swap_decision("armed", swapped=False, gate_ok=True) == "rebuild"

    def test_shadow_rebuilds_even_when_the_gate_passes(self):
        """LE point du mode ombre : on mesure, on ne sert jamais l'échange."""
        assert hot_swap_decision("shadow", swapped=True, gate_ok=True) == "rebuild"

    def test_off_rebuilds(self):
        assert hot_swap_decision("off", swapped=False, gate_ok=False) == "rebuild"


class TestGateParams:
    def test_defaults_are_the_historical_ones(self, monkeypatch):
        monkeypatch.delenv("RELIQUARY_HOT_SWAP_GATE_TOKENS", raising=False)
        monkeypatch.delenv("RELIQUARY_HOT_SWAP_GATE_FLOOR", raising=False)
        assert hot_swap_gate_params() == (48, 0.80)

    def test_tokens_and_floor_are_tunable_without_a_deploy(self, monkeypatch):
        monkeypatch.setenv("RELIQUARY_HOT_SWAP_GATE_TOKENS", "256")
        monkeypatch.setenv("RELIQUARY_HOT_SWAP_GATE_FLOOR", "0.93")
        assert hot_swap_gate_params() == (256, 0.93)

    def test_garbage_falls_back_to_the_defaults(self, monkeypatch):
        monkeypatch.setenv("RELIQUARY_HOT_SWAP_GATE_TOKENS", "beaucoup")
        monkeypatch.setenv("RELIQUARY_HOT_SWAP_GATE_FLOOR", "")
        assert hot_swap_gate_params() == (48, 0.80)


class TestBackendHonoursShadow:
    """En mode ombre l'échange doit VRAIMENT avoir lieu : sans lui la gate
    n'a rien à mesurer et la calibration est vide."""

    def _backend(self):
        from reliquary.miner.vllm_backend import VLLMBackend
        return VLLMBackend("old-path", forced_seed=True)

    def _llm(self, calls):
        class _LLM:
            def collective_rpc(self, method, kwargs=None):
                calls.append((method, kwargs))

            def reset_prefix_cache(self):
                calls.append(("reset_prefix_cache", None))
        return _LLM()

    def test_shadow_swaps_and_purges_the_prefix_cache(self, monkeypatch):
        monkeypatch.setenv("RELIQUARY_HOT_SWAP", "shadow")
        b, calls = self._backend(), []
        b._llm = self._llm(calls)
        assert b.reload_weights_inplace("new-path") is True
        assert calls == [
            ("reload_weights", {"weights_path": "new-path"}),
            ("reset_prefix_cache", None),
        ]

    def test_off_still_refuses_to_swap(self, monkeypatch):
        monkeypatch.setenv("RELIQUARY_HOT_SWAP", "0")
        b, calls = self._backend(), []
        b._llm = self._llm(calls)
        assert b.reload_weights_inplace("new-path") is False
        assert calls == []


class TestGateReadsItsParamsFromTheEnvironment:
    """Le plancher et la taille d'échantillon doivent se régler par variable :
    ils seront fixés depuis la MESURE du mode ombre, et un redéploiement de
    code coûte un restart donc une fenêtre."""

    def _gate(self):
        import types
        from reliquary.miner.engine import MiningEngine
        dummy = types.SimpleNamespace(hf_model=None)
        return MiningEngine._hot_swap_self_gate.__get__(dummy)

    def test_probe_is_asked_for_the_configured_number_of_tokens(self, monkeypatch):
        monkeypatch.setenv("RELIQUARY_HOT_SWAP_GATE_TOKENS", "256")
        vus = []

        class _Backend:
            def generate_forced_probe(self, prompt_ids, n_tokens, **k):
                vus.append(n_tokens)
                return []          # échantillon vide → la gate FAIL, sans risque

        assert self._gate()(_Backend(), probe_timeout_s=5.0) is False
        assert vus == [256]


class TestEngineWiring:
    """Le câblage lui-même, pas seulement la politique : c'est là que se
    joue « est-ce que le moteur qui sert la fenêtre est l'échangé ? »."""

    def _engine(self, gate_ok):
        import types
        from reliquary.miner.engine import MiningEngine
        dummy = types.SimpleNamespace(
            _hot_swap_self_gate=lambda backend, label=None: gate_ok,
            _loaded_checkpoint_path=None,
        )
        return dummy, MiningEngine._hot_swap_attempt.__get__(dummy)

    class _Backend:
        def __init__(self):
            self.swaps = []

        def reload_weights_inplace(self, path):
            self.swaps.append(path)
            return True

    def test_armed_with_a_passing_gate_keeps_the_swapped_engine(self, monkeypatch):
        monkeypatch.setenv("RELIQUARY_HOT_SWAP", "1")
        _, attempt = self._engine(gate_ok=True)
        b = self._Backend()
        assert attempt(b, "/ckpt/new") == "keep"
        assert b.swaps == ["/ckpt/new"]

    def test_shadow_measures_then_rebuilds(self, monkeypatch):
        monkeypatch.setenv("RELIQUARY_HOT_SWAP", "shadow")
        _, attempt = self._engine(gate_ok=True)
        b = self._Backend()
        assert attempt(b, "/ckpt/new") == "rebuild"
        assert b.swaps == ["/ckpt/new"], "l'échange doit avoir eu lieu pour être mesuré"

    def test_off_never_touches_the_engine(self, monkeypatch):
        monkeypatch.setenv("RELIQUARY_HOT_SWAP", "0")
        _, attempt = self._engine(gate_ok=True)
        b = self._Backend()
        assert attempt(b, "/ckpt/new") == "rebuild"
        assert b.swaps == []

    def test_armed_with_a_failing_gate_rebuilds(self, monkeypatch):
        monkeypatch.setenv("RELIQUARY_HOT_SWAP", "1")
        _, attempt = self._engine(gate_ok=False)
        assert attempt(self._Backend(), "/ckpt/new") == "rebuild"

    def test_backend_without_hot_swap_support_rebuilds(self, monkeypatch):
        monkeypatch.setenv("RELIQUARY_HOT_SWAP", "1")
        _, attempt = self._engine(gate_ok=True)

        class _Old:
            pass
        assert attempt(_Old(), "/ckpt/new") == "rebuild"


class TestShadowMeasuresTheControl:
    """Le VRAI risque (commentaire du launcher, 28/08) : un ``reload_weights``
    qui échoue EN SILENCE laisse les anciens poids. Or deux checkpoints
    consécutifs ne diffèrent que d'UN pas d'entraînement — les picks se
    ressemblent, donc un plancher bas passerait quand même.

    Au moment de l'échange ``hf_model`` porte déjà les NOUVEAUX poids et vLLM
    les ANCIENS : une sonde AVANT l'échange donne donc, gratuitement, la
    signature exacte d'un échange raté. Le mode ombre doit journaliser les
    deux — sans les deux, aucun plancher n'est défendable.
    """

    def _engine(self, appels):
        import types
        from reliquary.miner.engine import MiningEngine
        dummy = types.SimpleNamespace(
            _hot_swap_self_gate=lambda backend, label=None: (
                appels.append(label) or True),
            _loaded_checkpoint_path=None,
        )
        return MiningEngine._hot_swap_attempt.__get__(dummy)

    class _Backend:
        def __init__(self, appels):
            self.appels = appels

        def reload_weights_inplace(self, path):
            self.appels.append("swap")
            return True

    def test_shadow_probes_before_and_after_the_swap(self, monkeypatch):
        monkeypatch.setenv("RELIQUARY_HOT_SWAP", "shadow")
        appels = []
        assert self._engine(appels)(self._Backend(appels), "/ckpt/new") == "rebuild"
        assert appels == ["temoin-avant-echange", "swap", "apres-echange"]

    def test_armed_does_not_pay_for_the_control_probe(self, monkeypatch):
        """En production armée la sonde témoin serait du gel pur : interdite."""
        monkeypatch.setenv("RELIQUARY_HOT_SWAP", "1")
        appels = []
        assert self._engine(appels)(self._Backend(appels), "/ckpt/new") == "keep"
        assert appels == ["swap", "apres-echange"]
