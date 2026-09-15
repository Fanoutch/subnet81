"""Preuve GRAIL en parallèle de la vérification EOS (15/09).

Mesuré (26 fenêtres, tirs < 30 s) : prêt → précommit envoyé 4,7 s p50, dont
2,5 s d'attente de la vérification EOS par la réplique puis 1,2 s de preuve,
en série. Or l'admission du validateur ferme tôt : tirs à 12-18 s admis à
100 %, 18-25 s à ~50 %, au-delà ~0 % ; notre tir médian est à 18,6 s. Depuis
le correctif des poids, 1,5 % seulement des groupes en zone sont réparés
(5/~330) : la preuve calculée sur les tokens d'origine est presque toujours
la bonne. On la lance dès le grading en zone ; après la vérification, seuls
les rollouts réparés sont re-prouvés, et une vérification absente (réplique
injoignable) ramène au chemin historique.
"""
from __future__ import annotations

import threading
import types

from reliquary.miner import engine

EOS = 99


def _gens(n=4):
    return [{"tokens": [1, 2, 10 + r, EOS], "prompt_length": 2} for r in range(n)]


def _eng(repair_fn, proof_log, proof_started=None):
    eng = engine.MiningEngine.__new__(engine.MiningEngine)
    eng._eos_ids = [EOS]
    eng._cached_randomness = "ab" * 32
    eng._local_hash = "c" * 40
    eng.tokenizer = types.SimpleNamespace(decode=lambda toks: "t%d" % toks[-2])
    eng._repair_terminal_eos = repair_fn

    def fake_proof(gens, texts=None, *, device=None, terminal_ctx=None):
        proof_log.append({"tokens": [list(g["tokens"]) for g in gens],
                          "texts": list(texts or []), "terminal_ctx": terminal_ctx})
        if proof_started is not None:
            proof_started.set()
        return [{"all_tokens": list(g["tokens"]), "local_screen": None} for g in gens]

    eng._proof_rollouts = fake_proof
    return eng


def _ok(gens):
    out = [dict(g) for g in gens]
    for g in out:
        g["terminal_ok"] = True
    return out


def test_preuve_demarre_pendant_la_verification():
    started = threading.Event()
    log = []

    def repair(gens, prompt_idx, env=None):
        # la vérification ne se termine qu'une fois la preuve lancée
        assert started.wait(5.0), "la preuve n'a pas démarré pendant la vérification"
        return _ok(gens)

    eng = _eng(repair, log, started)
    gens = _gens()
    out, cache = eng._repair_with_early_proof(
        gens, ["t10", "t11", "t12", "t13"], prompt_idx=7, env=None)
    assert all(g["terminal_ok"] for g in out)
    assert len(log) == 1 and log[0]["terminal_ctx"] is None
    assert [c["all_tokens"] for c in cache] == [g["tokens"] for g in gens]


def test_rollout_repare_seul_reprouve_et_remplace():
    log = []

    def repair(gens, prompt_idx, env=None):
        out = _ok(gens)
        out[2]["tokens"] = [1, 2, 12, 55, 66, EOS]
        return out

    eng = _eng(repair, log)
    out, cache = eng._repair_with_early_proof(
        _gens(), ["a", "b", "c", "d"], prompt_idx=7, env=None)
    assert len(log) == 2
    assert log[1]["tokens"] == [[1, 2, 12, 55, 66, EOS]]
    assert log[1]["texts"] == ["t66"]
    assert cache[2]["all_tokens"] == [1, 2, 12, 55, 66, EOS]
    assert cache[0]["all_tokens"] == [1, 2, 10, EOS]
    assert len(cache) == 4


def test_verification_absente_pas_de_preuve_anticipee():
    """Réplique injoignable : EOS non vérifiés → la garde locale de la preuve
    doit tourner (terminal_ctx), donc chemin historique."""
    log = []
    eng = _eng(lambda gens, prompt_idx, env=None: gens, log)
    out, cache = eng._repair_with_early_proof(
        _gens(), ["a"] * 4, prompt_idx=7, env=None)
    assert cache is None
    assert out is not None


def test_reparation_impossible():
    log = []
    eng = _eng(lambda gens, prompt_idx, env=None: None, log)
    out, cache = eng._repair_with_early_proof(
        _gens(), ["a"] * 4, prompt_idx=7, env=None)
    assert out is None and cache is None


def test_preuve_anticipee_en_echec_retombe_sur_le_chemin_historique():
    eng = _eng(lambda gens, prompt_idx, env=None: _ok(gens), [])

    def boom(gens, texts=None, *, device=None, terminal_ctx=None):
        raise RuntimeError("OOM")

    eng._proof_rollouts = boom
    out, cache = eng._repair_with_early_proof(
        _gens(), ["a"] * 4, prompt_idx=7, env=None)
    assert out is not None and cache is None


def test_rollout_sans_stop_final_ne_bloque_pas_la_reutilisation():
    log = []
    gens = _gens()
    gens[3]["tokens"] = [1, 2, 5, 6]               # tronqué : pas de verdict EOS

    def repair(g, prompt_idx, env=None):
        out = [dict(x) for x in g]
        for x in out[:3]:
            x["terminal_ok"] = True
        return out

    eng = _eng(repair, log)
    out, cache = eng._repair_with_early_proof(gens, ["a"] * 4, prompt_idx=7, env=None)
    assert cache is not None and len(log) == 1


# ----------------------------------------------------------- _pre_bake_entry
def _prebake(monkeypatch, flag):
    from reliquary.miner import replica_client

    monkeypatch.setenv("RELIQUARY_REPLICA_SOCKET", "/tmp/replica.sock")
    monkeypatch.setenv("RELIQUARY_MIN_ROLLOUT_LEN", "0")
    monkeypatch.setenv("RELIQUARY_PROOF_DURING_REPAIR", flag)
    order = []

    def repair(gens, prompt_idx, env=None):
        order.append("repair")
        return _ok(gens)

    log = []
    eng = _eng(repair, log)
    gens = [{"tokens": [1, 2, 10 + r, EOS], "prompt_length": 2}
            for r in range(engine.M_ROLLOUTS)]
    eng._generate_m_rollouts = lambda problem, rand, env, prompt_idx: gens
    real_proof = eng._proof_rollouts

    def proof(g, texts=None, *, device=None, terminal_ctx=None):
        order.append("proof")
        out = real_proof(g, texts, device=device, terminal_ctx=terminal_ctx)
        for r in out:
            r["local_screen"] = "stop_ici"         # arrête le pipeline juste après
        return out

    eng._proof_rollouts = proof
    monkeypatch.setattr(engine, "grade_group_parallel_ex",
                        lambda env, pairs, max_workers=8: (
                            [0.0, 1.0] * (len(pairs) // 2), [False] * len(pairs)))
    monkeypatch.setattr(engine, "spec_proof_enabled", lambda: False)
    monkeypatch.setattr(engine, "passes_auction_gate", lambda r: True)
    monkeypatch.setattr(engine, "should_drop_for_termination", lambda *a, **k: False)
    monkeypatch.setattr(engine, "v4_uncertain_guard", lambda *a, **k: (None, None))
    monkeypatch.setattr(engine, "dump_group_sample", lambda **k: None)
    drops = []
    eng._record_drop = lambda dropped, reason=None: drops.append(reason)
    eng._sz_note = lambda *a, **k: None
    return eng, order, log, drops


def test_pre_bake_entry_preuve_unique_lancee_avant_la_fin_de_verification(monkeypatch):
    eng, order, log, drops = _prebake(monkeypatch, "1")
    eng._pre_bake_entry(7, {"prompt": "p"}, 1,
                        env=types.SimpleNamespace(name="opencodeinstruct"))
    assert order.count("proof") == 1
    assert log[0]["terminal_ctx"] is None
    assert drops == ["stop_ici"]


def test_pre_bake_entry_variable_coupee_chemin_historique(monkeypatch):
    eng, order, log, drops = _prebake(monkeypatch, "0")
    eng._pre_bake_entry(7, {"prompt": "p"}, 1,
                        env=types.SimpleNamespace(name="opencodeinstruct"))
    assert order == ["repair", "proof"]
    assert log[0]["terminal_ctx"] is not None
    assert drops == ["stop_ici"]
