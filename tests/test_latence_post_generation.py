"""Gardes de latence de l'étage POST-GÉNÉRATION (26/08).

Trois réglages, tous DÉSACTIVÉS PAR DÉFAUT — ces tests vérifient d'abord que
le comportement historique est byte-identique quand les variables ne sont pas
posées, puis que chaque garde fait ce qu'elle annonce quand on la pose.
"""
import importlib
import os
import time

import pytest


# ---------------------------------------------------------------- grading cap
def _reload_constants():
    import reliquary.constants as c
    return importlib.reload(c)


def test_grade_timeout_defaut_inchange(monkeypatch):
    monkeypatch.delenv("RELIQUARY_GRADE_TIMEOUT_S", raising=False)
    c = _reload_constants()
    assert c.GRADER_EVAL_TIMEOUT_SECONDS == 5.0


def test_grade_timeout_surchargeable(monkeypatch):
    monkeypatch.setenv("RELIQUARY_GRADE_TIMEOUT_S", "1.0")
    c = _reload_constants()
    assert c.GRADER_EVAL_TIMEOUT_SECONDS == 1.0
    monkeypatch.delenv("RELIQUARY_GRADE_TIMEOUT_S", raising=False)
    _reload_constants()


def test_grade_timeout_est_bien_celui_passe_au_grader(monkeypatch):
    """Le budget doit ARRIVER jusqu'au subprocess, pas rester décoratif."""
    monkeypatch.setenv("RELIQUARY_GRADE_TIMEOUT_S", "0.4")
    _reload_constants()
    import reliquary.environment.opencodeinstruct as oci
    importlib.reload(oci)
    seen = {}

    def fake_grade(code, cases, timeout_s=5.0):
        seen["t"] = timeout_s
        return 0.0

    monkeypatch.setattr(oci, "grade_structured_cases", fake_grade)
    env = oci.OpenCodeInstructEnvironment.__new__(
        oci.OpenCodeInstructEnvironment)
    env._cases_by_id = {"cid": [{"entry": {"name": "f"}, "args": [], "expected": 1}]}
    env.compute_reward({"ground_truth": "cid"}, "```python\ndef f():\n    return 1\n```")
    assert seen["t"] == pytest.approx(0.4)
    monkeypatch.delenv("RELIQUARY_GRADE_TIMEOUT_S", raising=False)
    _reload_constants()
    importlib.reload(oci)


# ------------------------------------------------------------ couvre-feu fire
class _Diag(dict):
    def __missing__(self, k):
        self[k] = 0
        return 0


def _engine_stub(open_ts):
    from reliquary.miner.engine import MiningEngine
    from reliquary.protocol.submission import WindowState

    eng = MiningEngine.__new__(MiningEngine)
    st = type("S", (), {})()
    st.state = WindowState.OPEN
    st.randomness = "ab" * 16
    st.window_n = 42
    eng._last_state = st
    eng._fire_ctx = ("http://x", None, [])
    eng._cached_randomness = st.randomness
    eng._cached_window_n = st.window_n
    eng._sealed_window = None
    eng._window_open_ts = open_ts
    eng._inflight_fire_tasks = set()
    eng._submitted_count = {}
    eng.__dict__["_fire_diag_map"] = {42: _Diag()}
    eng._fire_as_ready = lambda *a, **k: True
    return eng, st


def test_couvrefeu_absent_par_defaut(monkeypatch):
    """Sans la variable, la garde ne doit RIEN faire (elle laisse passer)."""
    monkeypatch.delenv("RELIQUARY_FIRE_CURFEW_S", raising=False)
    eng, _ = _engine_stub(time.time() - 300.0)   # très en retard
    # on n'exécute pas le vrai create_task : on vérifie juste que la garde
    # couvre-feu n'a PAS incrémenté son compteur.
    try:
        eng._maybe_fire_on_append()
    except Exception:
        pass
    assert eng._fire_diag[42].get("curfew", 0) == 0


def test_couvrefeu_bloque_apres_le_seuil(monkeypatch):
    monkeypatch.setenv("RELIQUARY_FIRE_CURFEW_S", "85")
    eng, _ = _engine_stub(time.time() - 90.0)
    assert eng._maybe_fire_on_append() is False
    assert eng._fire_diag[42]["curfew"] == 1
    monkeypatch.delenv("RELIQUARY_FIRE_CURFEW_S", raising=False)


def test_couvrefeu_laisse_passer_avant_le_seuil(monkeypatch):
    monkeypatch.setenv("RELIQUARY_FIRE_CURFEW_S", "85")
    eng, _ = _engine_stub(time.time() - 10.0)
    try:
        eng._maybe_fire_on_append()
    except Exception:
        pass
    assert eng._fire_diag[42].get("curfew", 0) == 0
    monkeypatch.delenv("RELIQUARY_FIRE_CURFEW_S", raising=False)


# ------------------------------------------------------- garde de round drand
def test_garde_drand_ne_dort_jamais_par_defaut(monkeypatch):
    """RELIQUARY_DRAND_MIN_HEADROOM_S absent = zéro attente ajoutée."""
    monkeypatch.delenv("RELIQUARY_DRAND_MIN_HEADROOM_S", raising=False)
    import reliquary.miner.engine as E
    from reliquary.infrastructure import drand as D
    slept = []
    monkeypatch.setattr(E.time, "sleep", lambda s: slept.append(s))
    # chaîne figée : on isole la garde de tout aléa réseau/cache
    monkeypatch.setattr(D, "get_current_chain",
                        lambda *a, **k: {"genesis_time": 0, "period": 3})
    monkeypatch.setattr(E.time, "time", lambda: 302.8)   # 0.2 s de marge
    eng = E.MiningEngine.__new__(E.MiningEngine)
    eng._local_hash = "cd" * 32
    with pytest.raises(Exception):
        # l'appel échouera plus loin (wallet absent) — seul le sleep compte
        eng._build_signed_request_sync([], "ab" * 32, 1, None, "hk", "n")
    assert slept == []


def test_garde_drand_dort_quand_la_marge_est_trop_courte(monkeypatch):
    monkeypatch.setenv("RELIQUARY_DRAND_MIN_HEADROOM_S", "0.6")
    import reliquary.miner.engine as E
    from reliquary.infrastructure import drand as D
    slept = []
    monkeypatch.setattr(E.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr(D, "get_current_chain",
                        lambda *a, **k: {"genesis_time": 0, "period": 3})
    # horloge à 2.8 s dans le round → 0.2 s de marge, sous le seuil
    monkeypatch.setattr(E.time, "time", lambda: 302.8)
    eng = E.MiningEngine.__new__(E.MiningEngine)
    eng._local_hash = "cd" * 32
    with pytest.raises(Exception):
        eng._build_signed_request_sync([], "ab" * 32, 1, None, "hk", "n")
    assert len(slept) == 1 and 0.19 < slept[0] < 0.25
    monkeypatch.delenv("RELIQUARY_DRAND_MIN_HEADROOM_S", raising=False)


def test_couvrefeu_couvre_aussi_le_chemin_ARMED(monkeypatch):
    """``_fire_for_window`` est appelé DIRECTEMENT par le trigger loop —
    la garde doit y être aussi, sinon la moitié des tirs tardifs passe."""
    import asyncio

    monkeypatch.setenv("RELIQUARY_FIRE_CURFEW_S", "85")
    eng, st = _engine_stub(time.time() - 120.0)
    eng._pool = [{"prompt_idx": 1}]
    st.cooldown_prompts = []
    asyncio.run(eng._fire_for_window(st, "http://x", None, []))
    assert eng._fire_diag[42]["curfew"] == 1
    assert eng._pool == [{"prompt_idx": 1}]   # rien n'a été drainé
    monkeypatch.delenv("RELIQUARY_FIRE_CURFEW_S", raising=False)


def test_chemin_ARMED_intact_sans_couvrefeu(monkeypatch):
    import asyncio

    monkeypatch.delenv("RELIQUARY_FIRE_CURFEW_S", raising=False)
    eng, st = _engine_stub(time.time() - 120.0)
    eng._pool = []
    eng._pool_lock = asyncio.Lock()
    st.cooldown_prompts = []
    eng.active_envs = []
    eng.envs = {}
    asyncio.run(eng._fire_for_window(st, "http://x", None, []))
    assert eng._fire_diag[42].get("curfew", 0) == 0
