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
    eng.tokenizer = types.SimpleNamespace(decode=lambda toks: "t")
    monkeypatch.setattr(engine, "grade_group_parallel_ex",
                        lambda env, pairs, max_workers=8: (
                            [0.0, 1.0] * (len(pairs) // 2), [False] * len(pairs)))
    drops = []
    eng._record_drop = lambda dropped, reason=None: drops.append(reason)
    out = eng._pre_bake_entry(7, {"prompt": "p"}, 1,
                              env=types.SimpleNamespace(name="opencodeinstruct"))
    assert out is None
    assert drops == ["terminal_repair_failed"]


# ------------------------------------------- grader AVANT de réparer (14/09)
# Fen 45896 : des groupes σ=0 passaient 5-30 s en réparation avant d'être jetés
# hors zone. On grade d'abord : hors zone → jeté sans réparation ; sinon on ne
# re-grade que les rollouts modifiés par la réparation.
def _prebake_engine(monkeypatch, rewards, repaired_idx=None):
    eng, _ = _eng(monkeypatch, [], lambda it: [EOS])
    gens = [{"tokens": [1, 2, 10 + r, EOS], "prompt_length": 2}
            for r in range(engine.M_ROLLOUTS)]
    eng._generate_m_rollouts = lambda problem, rand, env, prompt_idx: gens
    eng.tokenizer = types.SimpleNamespace(decode=lambda toks: "t%d" % toks[0])
    repair_calls = []

    def fake_repair(g, prompt_idx, env=None):
        repair_calls.append(prompt_idx)
        out = [dict(x) for x in g]
        for i in repaired_idx or []:
            out[i]["tokens"] = out[i]["tokens"][:-1] + [77, EOS]
        return out

    eng._repair_terminal_eos = fake_repair
    grade_calls = []

    def fake_grade(env, pairs, max_workers=8):
        grade_calls.append(len(pairs))
        if len(pairs) == engine.M_ROLLOUTS:
            return list(rewards), [False] * len(pairs)
        return [1.0] * len(pairs), [False] * len(pairs)

    monkeypatch.setattr(engine, "grade_group_parallel_ex", fake_grade)
    monkeypatch.setattr(engine, "spec_proof_enabled", lambda: False)
    drops = []
    eng._record_drop = lambda dropped, reason=None: drops.append(reason)
    eng._sz_note = lambda *a, **k: None
    return eng, repair_calls, grade_calls, drops


def test_hors_zone_avant_reparation_jete_sans_reparer(monkeypatch):
    eng, repair_calls, grade_calls, drops = _prebake_engine(
        monkeypatch, [0.0] * engine.M_ROLLOUTS)
    out = eng._pre_bake_entry(7, {"prompt": "p"}, 1,
                              env=types.SimpleNamespace(name="opencodeinstruct"))
    assert out is None
    assert repair_calls == []
    assert drops == ["out_of_zone"]
    assert grade_calls == [engine.M_ROLLOUTS]


def test_en_zone_repare_puis_regrade_seulement_les_modifies(monkeypatch):
    rewards = [0.0, 1.0] * (engine.M_ROLLOUTS // 2)
    eng, repair_calls, grade_calls, drops = _prebake_engine(
        monkeypatch, rewards, repaired_idx=[3])
    seen = {}

    real_skip = engine._skip_for_out_of_zone

    def stop_at_zone(r):
        seen.setdefault("calls", 0)
        seen["calls"] += 1
        if seen["calls"] == 1:           # décision AVANT réparation : en zone
            return real_skip(r)
        seen["rewards"] = list(r)
        return True                      # arrête le pipeline après réparation

    monkeypatch.setattr(engine, "_skip_for_out_of_zone", stop_at_zone)
    monkeypatch.setenv("RELIQUARY_MIN_ROLLOUT_LEN", "0")   # rollouts factices courts
    eng._pre_bake_entry(7, {"prompt": "p"}, 1,
                        env=types.SimpleNamespace(name="opencodeinstruct"))
    assert repair_calls == [7]
    assert grade_calls == [engine.M_ROLLOUTS, 1]        # 1 seul re-gradé
    expected = list(rewards)
    expected[3] = 1.0
    assert seen["rewards"] == expected


# ------------------------------------------ budget de réparation (14/09 soir)
# 7 réparations sur 51 ajoutaient >650 tokens par rollout (20-28 s) et 2 ont
# filé vers le plafond de 8 192 : groupes arrivés trop tard de toute façon, et
# bake suivant retardé. Chaque continuation est plafonnée ; au-delà, le groupe
# est abandonné.
def _eng_kw(monkeypatch, verdict_rounds, cont_fn):
    eng, calls = _eng(monkeypatch, verdict_rounds, cont_fn)
    seen = {}

    def run_cont(items, **kw):
        seen.update(kw)
        calls["cont"].append([(it["rollout_index"], it["prefix_tokens"]) for it in items])
        return [cont_fn(it) for it in items]

    eng._vllm_backend = types.SimpleNamespace(run_forced_continuations=run_cont)
    return eng, seen


def test_budget_de_reparation_par_defaut(monkeypatch):
    monkeypatch.delenv("RELIQUARY_TERMINAL_REPAIR_MAX_NEW", raising=False)
    eng, seen = _eng_kw(monkeypatch, [{2: (False, 55)}, {}], lambda it: [66, EOS])
    assert eng._repair_terminal_eos(_gens(), prompt_idx=7, env=None) is not None
    assert seen["max_new_tokens"] == 512


def test_budget_de_reparation_reglable(monkeypatch):
    monkeypatch.setenv("RELIQUARY_TERMINAL_REPAIR_MAX_NEW", "64")
    eng, seen = _eng_kw(monkeypatch, [{2: (False, 55)}, {}], lambda it: [66, EOS])
    eng._repair_terminal_eos(_gens(), prompt_idx=7, env=None)
    assert seen["max_new_tokens"] == 64


def test_budget_epuise_groupe_abandonne(monkeypatch, caplog):
    monkeypatch.setenv("RELIQUARY_TERMINAL_REPAIR_MAX_NEW", "3")
    eng, _ = _eng_kw(monkeypatch, [{2: (False, 55)}], lambda it: [66, 67, 68])
    import logging
    with caplog.at_level(logging.INFO):
        assert eng._repair_terminal_eos(_gens(), prompt_idx=7, env=None) is None
    assert "budget" in caplog.text
