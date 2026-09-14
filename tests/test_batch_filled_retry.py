"""Réessais ``batch_filled`` sous V1 : ne plus abandonner au 2e refus.

Mesuré le 14/09 : sur les fenêtres 45900-45905, ~la moitié des 112 groupes
code payés arrivent APRÈS 20 s (p90 55-80 s). Un ``batch_filled`` au corps
alors que l'env admet encore = file de grading du validateur momentanément
pleine (64 places, 1 worker) : d'autres passent juste après. Le plafond hérité
de v4 (2 essais, calibré pour ``stale_round``) a jeté 36 groupes dans la
fenêtre 45912.

Règle : ``batch_filled`` + env ouvert + admis sous la cible prouvable (+ marge)
→ nouvel essai après une pause croissante, jusqu'à
``RELIQUARY_BATCH_FILLED_MAX_RETRIES``. Tout le reste : règle historique
(2 essais au total).
"""
from __future__ import annotations

import asyncio
import time as _time
from types import SimpleNamespace

from reliquary.miner import engine

CODE, MATH = "opencodeinstruct", "openmathinstruct"


def _fc(admitted_code=50, budget=224, proven_code=0, picks_target=7):
    return SimpleNamespace(
        phase="collecting", precommit_cutoff_ts=_time.time() + 900.0,
        precommit_seconds=1407.0, picks_target=picks_target,
        admission_budgets={MATH: 224, CODE: budget},
        admitted={MATH: 0, CODE: admitted_code},
        proven={MATH: 0, CODE: proven_code},
        picks_by_environment={MATH: 0, CODE: 0},
        remaining={MATH: 224, CODE: budget - admitted_code},
    )


def _st(fc):
    return SimpleNamespace(fill_closed=fc, window_n=7, state="open",
                           randomness="ab" * 16, cooldown_prompts=[])


def test_batch_filled_env_ouvert_reessaie_avec_pause(monkeypatch):
    monkeypatch.delenv("RELIQUARY_BATCH_FILLED_MAX_RETRIES", raising=False)
    st = _st(_fc(admitted_code=60))
    ok, nb = engine.fire_retry_decision(st, "batch_filled", 1, CODE, now=100.0)
    assert ok is True and nb is not None and nb > 100.0
    ok5, nb5 = engine.fire_retry_decision(st, "batch_filled", 5, CODE, now=100.0)
    assert ok5 is True and nb5 >= nb                       # pause croissante


def test_batch_filled_plafond_de_reessais(monkeypatch):
    monkeypatch.setenv("RELIQUARY_BATCH_FILLED_MAX_RETRIES", "4")
    st = _st(_fc(admitted_code=60))
    assert engine.fire_retry_decision(st, "batch_filled", 3, CODE, now=0.0)[0] is True
    assert engine.fire_retry_decision(st, "batch_filled", 4, CODE, now=0.0)[0] is False


def test_batch_filled_admis_au_dela_de_la_cible_regle_historique():
    # 7 × 16 = 112 prouvables ; au-delà de 112 + marge, nos groupes passeraient
    # derrière tous les autres dans l'ordre FIFO → règle historique.
    st = _st(_fc(admitted_code=200))
    assert engine.fire_retry_decision(st, "batch_filled", 1, CODE, now=0.0) == (True, None)
    assert engine.fire_retry_decision(st, "batch_filled", 2, CODE, now=0.0) == (False, None)


def test_batch_filled_env_ferme_abandon():
    st = _st(_fc(admitted_code=224))
    assert engine.fire_retry_decision(st, "batch_filled", 1, CODE, now=0.0) == (False, None)


def test_stale_round_regle_historique():
    st = _st(_fc(admitted_code=60))
    assert engine.fire_retry_decision(st, "stale_round", 1, CODE, now=0.0) == (True, None)
    assert engine.fire_retry_decision(st, "stale_round", 2, CODE, now=0.0) == (False, None)


def test_motif_non_reessayable():
    st = _st(_fc(admitted_code=60))
    assert engine.fire_retry_decision(st, "out_of_zone", 1, CODE, now=0.0) == (False, None)


def test_v5_regle_historique():
    st = _st(None)
    assert engine.fire_retry_decision(st, "batch_filled", 1, CODE, now=0.0) == (True, None)
    assert engine.fire_retry_decision(st, "batch_filled", 2, CODE, now=0.0) == (False, None)


def test_entree_en_pause_non_tirable():
    fc = _fc(admitted_code=60)
    env_of = lambda e: e["env_name"]
    pool = [{"env_name": CODE, "_retry_not_before": 110.0}]
    assert engine.pool_has_fireable_entry(pool, fc, env_of, now=100.0) is False
    assert engine.pool_has_fireable_entry(pool, fc, env_of, now=111.0) is True
    assert engine.pool_has_fireable_entry(pool, fc, env_of) is True   # sans horloge


def test_fire_for_window_garde_les_entrees_en_pause():
    fc = _fc(admitted_code=60)
    eng = engine.MiningEngine.__new__(engine.MiningEngine)
    eng._pool = [{"prompt_idx": 1, "env_name": CODE,
                  "_retry_not_before": _time.time() + 60.0}]
    eng._pool_lock = asyncio.Lock()
    eng._submitted_count = {}
    eng._last_state_ts = _time.time()
    eng.active_envs = [CODE]
    eng.envs = {CODE: SimpleNamespace(name=CODE)}
    eng._active_prompt_range = lambda *a, **k: None
    eng._sealed_window = None
    asyncio.run(eng._fire_for_window(_st(fc), "http://v", None, [], budget=10))
    assert len(eng._pool) == 1
    assert eng._fire_diag[7]["retry_backoff"] == 1
    assert eng._submitted_count.get(7, 0) == 0
