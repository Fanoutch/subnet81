"""Vérification EOS sélective (18/09, RELIQUARY_EOS_SELECTIVE)."""
from __future__ import annotations

import types

from reliquary.miner import engine

EOS = 99


def _gens(n=4):
    return [{"tokens": [1, 2, 10 + r, EOS], "prompt_length": 2} for r in range(n)]


def _eng(margins, repair_log, proof_raises=False):
    eng = engine.MiningEngine.__new__(engine.MiningEngine)
    eng._eos_ids = [EOS]
    eng._cached_randomness = "ab" * 32
    eng._local_hash = "c" * 40
    eng.tokenizer = types.SimpleNamespace(decode=lambda toks: "t")

    def fake_proof(gens, texts=None, *, device=None, terminal_ctx=None):
        if proof_raises:
            raise RuntimeError("OOM")
        eng.__dict__["_margin_tls"].margins = dict(margins)
        return [{"all_tokens": list(g["tokens"])} for g in gens]

    def repair(gens, prompt_idx, env=None):
        repair_log.append([bool(g.get("terminal_ok")) for g in gens])
        out = [dict(g) for g in gens]
        for g in out:
            g["terminal_ok"] = True
        return out

    eng._proof_rollouts = fake_proof
    eng._repair_terminal_eos = repair
    return eng


def test_marge_large_non_envoyee_a_la_replique(monkeypatch):
    monkeypatch.setenv("RELIQUARY_EOS_SELECTIVE", "1")
    monkeypatch.setenv("RELIQUARY_EOS_SELECTIVE_MARGIN", "0.15")
    log, timing = [], {}
    eng = _eng({0: 0.5, 1: 0.10, 2: 0.2, 3: -0.01}, log)
    out, cache = eng._repair_with_early_proof(_gens(), ["a"] * 4, prompt_idx=7, env=None, timing=timing)
    assert log == [[True, False, True, False]]          # 1 et 3 vérifiés seulement
    assert timing["eos_skipped"] == 2 and timing["eos_checked"] == 2
    assert cache is not None and all(g["terminal_ok"] for g in out)


def test_marge_absente_verifiee(monkeypatch):
    monkeypatch.setenv("RELIQUARY_EOS_SELECTIVE", "1")
    log, timing = [], {}
    eng = _eng({0: 0.9}, log)                             # 1-3 sans marge
    eng._repair_with_early_proof(_gens(), ["a"] * 4, prompt_idx=7, env=None, timing=timing)
    assert log == [[True, False, False, False]]


def test_preuve_en_echec_verification_complete(monkeypatch):
    monkeypatch.setenv("RELIQUARY_EOS_SELECTIVE", "1")
    log, timing = [], {}
    eng = _eng({0: 0.9, 1: 0.9, 2: 0.9, 3: 0.9}, log, proof_raises=True)
    out, cache = eng._repair_with_early_proof(_gens(), ["a"] * 4, prompt_idx=7, env=None, timing=timing)
    assert log == [[False, False, False, False]] and cache is None
    assert "eos_skipped" not in timing


def test_drapeau_coupe_comportement_historique(monkeypatch):
    monkeypatch.delenv("RELIQUARY_EOS_SELECTIVE", raising=False)
    log, timing = [], {}
    eng = _eng({0: 0.9, 1: 0.9, 2: 0.9, 3: 0.9}, log)
    eng._repair_with_early_proof(_gens(), ["a"] * 4, prompt_idx=7, env=None, timing=timing)
    assert log == [[False, False, False, False]] and "eos_skipped" not in timing


def test_rollout_sans_eos_final_ni_saute_ni_compte(monkeypatch):
    monkeypatch.setenv("RELIQUARY_EOS_SELECTIVE", "1")
    log, timing = [], {}
    g = _gens(); g[2]["tokens"] = [1, 2, 5, 6]            # tronqué
    eng = _eng({0: 0.9, 1: 0.9, 3: 0.9}, log)
    eng._repair_with_early_proof(g, ["a"] * 4, prompt_idx=7, env=None, timing=timing)
    assert log == [[True, True, False, True]]
    assert timing["eos_skipped"] == 3 and timing["eos_checked"] == 0


def test_reparation_ignore_les_rollouts_deja_valides(monkeypatch):
    from reliquary.miner import replica_client
    monkeypatch.setenv("RELIQUARY_REPLICA_SOCKET", "/tmp/r.sock")
    asked = []
    monkeypatch.setattr(replica_client, "terminal_verdicts",
                        lambda sock, **kw: asked.append([it["rollout"] for it in kw["items"]]) or
                        [{"ok": True} for _ in kw["items"]])
    eng = engine.MiningEngine.__new__(engine.MiningEngine)
    eng._eos_ids = [EOS]
    eng._cached_randomness = "ab" * 32
    eng._local_hash = "c" * 40
    eng._loaded_checkpoint_path = "/m"
    eng.max_new_tokens = 8192
    g = _gens(); g[0]["terminal_ok"] = True; g[2]["terminal_ok"] = True
    out = eng._repair_terminal_eos(g, prompt_idx=7, env=None)
    assert asked == [[1, 3]] and all(x.get("terminal_ok") for x in out)
