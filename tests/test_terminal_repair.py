"""Réparation de l'EOS final avant grading et preuve (V1, #253).

Aucun réglage vLLM ne rend l'EOS final exact (plafond ~90 % mesuré le 14/09).
Le moteur fait donc vérifier chaque rollout fini sur un stop par la réplique du
validateur ; un rollout refusé est prolongé par vLLM depuis ``tokens[:-1] +
[pick]`` (le token que le validateur tire à cette position), puis revérifié.
Le groupe ne part que si tous ses EOS sont validés.
"""
from __future__ import annotations

import types

import pytest

from reliquary.miner import engine, replica_client

EOS = 99
PLEN = 2


def _gens():
    return [{"tokens": [1, 2, 10 + r, EOS], "prompt_length": PLEN}
            for r in range(4)]


def _eng(monkeypatch, verdict_rounds, cont_fn):
    """``verdict_rounds`` : liste (un tour) de dict rollout → (ok, pick)."""
    eng = engine.MiningEngine.__new__(engine.MiningEngine)
    eng._eos_ids = [EOS]
    eng._cached_randomness = "ab" * 32
    eng._local_hash = "c" * 40
    eng._loaded_checkpoint_path = "/snap/rev"
    eng.max_new_tokens = 64
    calls = {"verify": [], "cont": []}
    rounds = list(verdict_rounds)

    def fake_verdicts(sock, **kw):
        calls["verify"].append([it["rollout"] for it in kw["items"]])
        table = rounds.pop(0)
        return [{"ok": table.get(it["rollout"], (True, EOS))[0],
                 "pick": table.get(it["rollout"], (True, EOS))[1],
                 "cdf_miss": 0.0} for it in kw["items"]]

    monkeypatch.setattr(replica_client, "terminal_verdicts", fake_verdicts)
    monkeypatch.setenv("RELIQUARY_REPLICA_SOCKET", "/tmp/replica.sock")

    def run_cont(items, **kw):
        calls["cont"].append([(it["rollout_index"], it["prefix_tokens"]) for it in items])
        return [cont_fn(it) for it in items]

    eng._vllm_backend = types.SimpleNamespace(run_forced_continuations=run_cont)
    return eng, calls


def test_tous_valides_rien_ne_change(monkeypatch):
    eng, calls = _eng(monkeypatch, [{}], lambda it: [EOS])
    gens = _gens()
    out = eng._repair_terminal_eos(gens, prompt_idx=7, env=None)
    assert [g["tokens"] for g in out] == [g["tokens"] for g in _gens()]
    assert all(g.get("terminal_ok") for g in out)
    assert calls["cont"] == []


def test_rollout_refuse_prolonge_puis_valide(monkeypatch):
    eng, calls = _eng(monkeypatch, [{2: (False, 55)}, {}],
                      lambda it: [66, EOS])
    out = eng._repair_terminal_eos(_gens(), prompt_idx=7, env=None)
    assert calls["cont"] == [[(2, [1, 2, 12, 55])]]
    assert out[2]["tokens"] == [1, 2, 12, 55, 66, EOS]
    assert calls["verify"] == [[0, 1, 2, 3], [2]]      # revérifie le seul réparé
    assert all(g.get("terminal_ok") for g in out)


def test_refus_persistant_groupe_abandonne(monkeypatch):
    eng, _ = _eng(monkeypatch, [{1: (False, 55)}] * 5, lambda it: [66, EOS])
    assert eng._repair_terminal_eos(_gens(), prompt_idx=7, env=None) is None


def test_continuation_sans_eos_groupe_abandonne(monkeypatch):
    eng, _ = _eng(monkeypatch, [{1: (False, 55)}], lambda it: [66, 67])
    assert eng._repair_terminal_eos(_gens(), prompt_idx=7, env=None) is None


def test_moteur_indisponible_groupe_abandonne(monkeypatch):
    eng, _ = _eng(monkeypatch, [{1: (False, 55)}], lambda it: [66, EOS])
    eng._vllm_backend = types.SimpleNamespace(
        run_forced_continuations=lambda items, **kw: None)
    assert eng._repair_terminal_eos(_gens(), prompt_idx=7, env=None) is None


def test_replique_absente_generations_intactes_non_verifiees(monkeypatch):
    eng, _ = _eng(monkeypatch, [], lambda it: [EOS])
    monkeypatch.setattr(replica_client, "terminal_verdicts", lambda s, **kw: None)
    out = eng._repair_terminal_eos(_gens(), prompt_idx=7, env=None)
    assert [g["tokens"] for g in out] == [g["tokens"] for g in _gens()]
    assert not any(g.get("terminal_ok") for g in out)


def test_sans_socket_rien(monkeypatch):
    eng, calls = _eng(monkeypatch, [], lambda it: [EOS])
    monkeypatch.delenv("RELIQUARY_REPLICA_SOCKET")
    gens = _gens()
    assert eng._repair_terminal_eos(gens, prompt_idx=7, env=None) is gens
    assert calls["verify"] == []


def test_rollout_tronque_au_plafond_non_juge(monkeypatch):
    eng, calls = _eng(monkeypatch, [{}], lambda it: [EOS])
    gens = _gens()
    gens[3]["tokens"] = [1, 2, 5, 6]            # pas d'EOS final
    out = eng._repair_terminal_eos(gens, prompt_idx=7, env=None)
    assert calls["verify"] == [[0, 1, 2]]
    assert out is not None


def test_pre_bake_entry_abandonne_si_reparation_impossible(monkeypatch):
    eng, _ = _eng(monkeypatch, [], lambda it: [EOS])
    eng._generate_m_rollouts = lambda problem, rand, env, prompt_idx: [
        {"tokens": [1, 2, 3, EOS], "prompt_length": 2}] * engine.M_ROLLOUTS
    eng._repair_terminal_eos = lambda gens, prompt_idx, env=None: None
    drops = []
    eng._record_drop = lambda dropped, reason=None: drops.append(reason)
    out = eng._pre_bake_entry(7, {"prompt": "p"}, 1,
                              env=types.SimpleNamespace(name="opencodeinstruct"))
    assert out is None
    assert drops == ["terminal_repair_failed"]
