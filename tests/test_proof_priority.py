"""Priorité des preuves de tête (18/09, RELIQUARY_PROOF_PRIORITY)."""
from __future__ import annotations

import threading
import time

from reliquary.miner import engine


def test_porte_sert_la_plus_prioritaire_d_abord():
    g = engine.ProofPriorityGate()
    order = []
    started = threading.Event()

    def holder():
        with g.slot((0, 0.0)):
            started.set()
            time.sleep(0.3)                 # pendant ce temps 3 groupes attendent
        order.append("holder")

    def waiter(key, name):
        with g.slot(key):
            order.append(name)

    th = threading.Thread(target=holder); th.start()
    assert started.wait(2)
    ws = [threading.Thread(target=waiter, args=(k, n)) for k, n in
          (((0, 30.0), "tardif"), ((0, 10.0), "tete"), ((0, 20.0), "milieu"))]
    for w in ws:
        w.start(); time.sleep(0.02)
    th.join(); [w.join() for w in ws]
    assert [x for x in order if x != "holder"] == ["tete", "milieu", "tardif"]


def test_nouvelle_fenetre_passe_devant_l_ancienne():
    assert (-46300, 99.0) < (-46299, 1.0)


def test_sans_concurrent_entree_immediate():
    g = engine.ProofPriorityGate()
    t0 = time.time()
    with g.slot((0, 1.0)):
        pass
    assert time.time() - t0 < 0.05 and not g.has_waiters()


def test_exception_libere_la_porte():
    g = engine.ProofPriorityGate()
    try:
        with g.slot((0, 1.0)):
            raise RuntimeError("OOM")
    except RuntimeError:
        pass
    with g.slot((0, 2.0)):
        pass


def test_inerte_sans_drapeau(monkeypatch):
    monkeypatch.delenv("RELIQUARY_PROOF_PRIORITY", raising=False)
    eng = engine.MiningEngine.__new__(engine.MiningEngine)
    timing = {"t_ready": 5.0}
    seen = {}

    def proof(gens, texts=None, *, device=None, terminal_ctx=None):
        seen["prio"] = getattr(eng.__dict__.get("_margin_tls"), "prio", "absent")
        seen["cm"] = type(eng._proof_priority_slot("cuda:0")).__name__
        return []
    eng._proof_rollouts = proof
    eng._timed_proof(timing)([])
    assert seen["prio"] is None and seen["cm"] == "nullcontext"
    assert "proof_gate_wait_s" not in timing


def test_active_pose_la_cle_et_mesure_l_attente(monkeypatch):
    monkeypatch.setenv("RELIQUARY_PROOF_PRIORITY", "1")
    eng = engine.MiningEngine.__new__(engine.MiningEngine)
    eng._cached_window_n = 46300
    timing = {"t_ready": 5.0}
    seen = {}

    def proof(gens, texts=None, *, device=None, terminal_ctx=None):
        seen["prio"] = eng.__dict__["_margin_tls"].prio
        with eng._proof_priority_slot("cuda:0"):
            pass
        return []
    eng._proof_rollouts = proof
    eng._timed_proof(timing)([])
    assert seen["prio"] == (-46300, 5.0)
    assert timing["proof_gate_wait_s"] >= 0.0
    assert eng.__dict__["_margin_tls"].prio is None      # remise à zéro
