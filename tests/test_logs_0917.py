"""Logs de mesure du 17/09 (restart C) — aucun changement de comportement.

- réplique : attente du jeton vs calcul, cumulés par fil côté client ;
- preuve : preuves déjà en vol au lancement, durée murale (et GPU si CUDA) ;
- garde drand : attente avant signature relue par nonce ;
- début de bake : décalage depuis l'ouverture.
"""
from __future__ import annotations

import inspect
import json
import threading
import types

from reliquary.miner import engine, replica_client


def test_cumul_replique_par_fil(monkeypatch):
    monkeypatch.setattr(replica_client, "_call", lambda sock, req, timeout: {
        "ok": True, "results": [{"ok": True}] * len(req["items"]),
        "t_wait": 0.25, "t_compute": 0.5})
    replica_client.terminal_stats_reset()
    for _ in range(2):
        replica_client.terminal_verdicts("s", model_path="m", randomness="r",
                                         checkpoint_hash="c", prompt_idx=1,
                                         items=[{"rollout": 0}, {"rollout": 1}])
    st = replica_client.terminal_stats()
    assert st == {"wait": 0.5, "compute": 1.0, "calls": 2, "items": 4}
    seen = {}

    def other():
        seen["v"] = replica_client.terminal_stats()
    t = threading.Thread(target=other); t.start(); t.join()
    assert seen["v"] == {}                      # cumul propre au fil


def test_ancienne_replique_sans_cles_de_temps(monkeypatch):
    monkeypatch.setattr(replica_client, "_call", lambda sock, req, timeout: {
        "ok": True, "results": [{"ok": True}]})
    replica_client.terminal_stats_reset()
    assert replica_client.terminal_verdicts("s", model_path="m", randomness="r",
                                            checkpoint_hash="c", prompt_idx=1,
                                            items=[{"rollout": 0}]) == [{"ok": True}]
    assert replica_client.terminal_stats()["calls"] == 1


def test_preuve_mesuree_sans_changer_le_resultat():
    eng = engine.MiningEngine.__new__(engine.MiningEngine)
    calls = []

    def proof(gens, texts=None, *, device=None, terminal_ctx=None):
        calls.append((len(gens), texts, terminal_ctx))
        return ["preuve"]

    eng._proof_rollouts = proof
    timing = {}
    out = eng._timed_proof(timing)([{"t": 1}], texts=["a"], terminal_ctx=None)
    assert out == ["preuve"] and calls == [(1, ["a"], None)]
    assert timing["proof_concurrent"] == 0 and "proof_wall_s" in timing
    assert eng._proofs_inflight == 0


def test_preuve_compteur_rendu_meme_si_exception():
    eng = engine.MiningEngine.__new__(engine.MiningEngine)

    def boom(*a, **k):
        raise RuntimeError("OOM")
    eng._proof_rollouts = boom
    timing = {}
    try:
        eng._timed_proof(timing)([], texts=[])
    except RuntimeError:
        pass
    assert eng._proofs_inflight == 0 and "proof_wall_s" in timing


def test_lignes_de_mesure_presentes_dans_le_code():
    src = inspect.getsource(engine.MiningEngine._bake_stream_fire)
    assert "bake_start: window=" in src
    src = inspect.getsource(engine.MiningEngine._submit_entry)
    assert 'headroom_wait_s' in src
    src = inspect.getsource(engine.MiningEngine._repair_with_early_proof)
    assert "eos_wait_s" in src and "_timed_proof" in src
