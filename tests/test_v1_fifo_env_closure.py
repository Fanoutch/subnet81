"""V1 FIFO (validateur ≥ 1f1cc16/#253) — fermeture d'admission PAR ENV.

Deux faits validateur (origin/main 0a69244) que le port v6 ignorait :

1. ``/state.fill_closed.phase`` et ``remaining`` sont calculés sur UN SEUL
   batcher (``server.py::_fill_closed_state_payload`` : ``admission_closed``
   du batcher servi met TOUS les ``remaining`` à 0 et la phase à
   ``draining``), alors que ``admitted`` / ``admission_budgets`` / ``proven``
   / ``picks_by_environment`` viennent du ``fill_state`` PARTAGÉ et sont
   exacts pour chaque env. On ferme donc un env sur SES compteurs.
2. ``fill_window.py`` : ``_admitted`` ne décroît jamais — un env dont
   ``admitted >= budget`` (ou ``proven >= picks_target × B_BATCH``) est
   fermé pour le reste de la fenêtre. Garder ses entrées en pool « au cas où
   le budget se remplirait à nouveau » faisait tourner le tir à vide
   (5,19 M lignes ``veto_env_budget`` sur une fenêtre, 13/09).
"""
from __future__ import annotations

import time as _time
from types import SimpleNamespace

from reliquary.miner import engine

CODE, MATH = "opencodeinstruct", "openmathinstruct"
CUTOFF = _time.time() + 900.0


def _fc(phase="collecting", admitted=None, budgets=None, proven=None,
        picks=None, picks_target=7, remaining=None):
    envs = (MATH, CODE)
    return SimpleNamespace(
        phase=phase,
        precommit_cutoff_ts=CUTOFF,
        precommit_seconds=1407.0,
        picks_target=picks_target,
        admission_budgets={e: 224 for e in envs} if budgets is None else budgets,
        admitted={e: 0 for e in envs} if admitted is None else admitted,
        proven={e: 0 for e in envs} if proven is None else proven,
        picks_by_environment={e: 0 for e in envs} if picks is None else picks,
        remaining={e: 224 for e in envs} if remaining is None else remaining,
    )


def _state(fc):
    return SimpleNamespace(fill_closed=fc, window_n=7, state="open",
                           randomness="ab" * 16)


# ----------------------------------------------------- état d'un env
def test_env_state_open_quand_budget_restant():
    assert engine.fill_closed_env_state(_fc(), CODE) == "open"


def test_env_state_closed_quand_admission_pleine():
    fc = _fc(admitted={MATH: 10, CODE: 224})
    assert engine.fill_closed_env_state(fc, CODE) == "closed"
    assert engine.fill_closed_env_state(fc, MATH) == "open"


def test_env_state_closed_quand_prouves_atteignent_la_cible():
    fc = _fc(proven={MATH: 0, CODE: 7 * 16})
    assert engine.fill_closed_env_state(fc, CODE) == "closed"


def test_env_state_closed_quand_picks_atteints():
    fc = _fc(picks={MATH: 0, CODE: 7})
    assert engine.fill_closed_env_state(fc, CODE) == "closed"


def test_env_state_unknown_sans_compteurs():
    fc = SimpleNamespace(phase="collecting", remaining={CODE: 0})
    assert engine.fill_closed_env_state(fc, CODE) == "unknown"
    assert engine.fill_closed_env_state(None, CODE) == "unknown"
    assert engine.fill_closed_env_state(_fc(), None) == "unknown"


# ------------------------------------- le remaining global ne ferme plus un env
def test_remaining_zero_global_nignore_pas_les_compteurs_exacts():
    # batcher servi (math) fermé → remaining tous à 0, mais code a encore
    # 60 places d'admission : l'env code N'EST PAS épuisé.
    fc = _fc(phase="draining", admitted={MATH: 224, CODE: 164},
             remaining={MATH: 0, CODE: 0})
    assert engine.fill_closed_env_exhausted(fc, CODE) is False
    assert engine.fill_closed_env_exhausted(fc, MATH) is True


def test_exhausted_repli_remaining_sans_compteurs():
    fc = SimpleNamespace(phase="collecting", remaining={CODE: 0})
    assert engine.fill_closed_env_exhausted(fc, CODE) is True


# --------------------------------------------------------- veto de tir
def test_veto_draining_ignore_si_env_encore_ouvert():
    fc = _fc(phase="draining", admitted={MATH: 224, CODE: 100})
    assert engine.fill_closed_fire_veto(fc, CUTOFF - 1000, 0, CODE) is None


def test_veto_draining_sans_env_laisse_le_filtre_par_entree_decider():
    fc = _fc(phase="draining", admitted={MATH: 224, CODE: 100})
    assert engine.fill_closed_fire_veto(fc, CUTOFF - 1000, 0, None) is None


def test_veto_draining_env_ferme():
    fc = _fc(phase="draining", admitted={MATH: 224, CODE: 224})
    assert engine.fill_closed_fire_veto(fc, CUTOFF - 1000, 0, CODE) == "phase_draining"


def test_veto_sealed_toujours():
    fc = _fc(phase="sealed")
    assert engine.fill_closed_fire_veto(fc, CUTOFF - 1000, 0, CODE) == "phase_sealed"
    assert engine.fill_closed_fire_veto(fc, CUTOFF - 1000, 0, None) == "phase_sealed"


# ------------------------------------------------------ bake et pool
def test_pause_bake_draining_env_ouvert_ne_pause_pas():
    st = _state(_fc(phase="draining", admitted={MATH: 224, CODE: 50}))
    assert engine.should_pause_bake(st, 0.0, 0, CODE) is False


def test_pause_bake_env_ferme_pause():
    st = _state(_fc(admitted={MATH: 0, CODE: 224}))
    assert engine.should_pause_bake(st, 0.0, 0, CODE) is True


def test_skip_pool_env_ferme_definitivement():
    st = _state(_fc(admitted={MATH: 0, CODE: 224}))
    assert engine.should_skip_pool(st, 0, CODE) is True


def test_skip_pool_draining_env_ouvert_garde_le_groupe():
    st = _state(_fc(phase="draining", admitted={MATH: 224, CODE: 10}))
    assert engine.should_skip_pool(st, 0, CODE) is False


# ------------------------------------------------ tir : plus de boucle à vide
def test_pool_has_fireable_entry():
    fc = _fc(admitted={MATH: 0, CODE: 224})
    env_of = lambda e: e["env"]
    assert engine.pool_has_fireable_entry([{"env": CODE}], fc, env_of) is False
    assert engine.pool_has_fireable_entry(
        [{"env": CODE}, {"env": MATH}], fc, env_of) is True
    assert engine.pool_has_fireable_entry([], fc, env_of) is False
    # v5 (fc None) : tout est tirable
    assert engine.pool_has_fireable_entry([{"env": CODE}], None, env_of) is True


def _engine_for_fire(pool, fc):
    import asyncio

    eng = engine.MiningEngine.__new__(engine.MiningEngine)
    eng._pool = list(pool)
    eng._pool_lock = asyncio.Lock()
    eng._submitted_count = {}
    eng._last_state_ts = _time.time()
    eng.active_envs = [CODE]
    eng.envs = {CODE: SimpleNamespace(name=CODE)}
    eng._active_prompt_range = lambda *a, **k: None
    eng._sealed_window = None
    return eng


def test_fire_for_window_jette_les_entrees_dun_env_ferme():
    import asyncio

    fc = _fc(admitted={MATH: 0, CODE: 224})
    eng = _engine_for_fire(
        [{"prompt_idx": 1, "env_name": CODE}, {"prompt_idx": 2, "env_name": CODE}], fc)
    st = SimpleNamespace(fill_closed=fc, window_n=7, state="open",
                         randomness="ab" * 16, cooldown_prompts=[])
    asyncio.run(eng._fire_for_window(st, "http://v", None, [], budget=10))
    assert eng._pool == []
    assert eng._fire_diag[7]["dropped_env_closed"] == 2
    assert eng._fire_diag[7]["veto_env_budget"] == 0


def test_maybe_fire_on_append_ne_relance_pas_sur_pool_intirable():
    import asyncio

    fc = _fc(remaining={CODE: 0})
    fc.admitted = {}      # compteurs absents → repli legacy, entrée gardée
    eng = _engine_for_fire([{"prompt_idx": 1, "env_name": CODE}], fc)
    st = SimpleNamespace(fill_closed=fc, window_n=7, state="open",
                         randomness="ab" * 16, cooldown_prompts=[])
    eng._last_state = st
    eng._cached_randomness = st.randomness
    eng._cached_window_n = 7
    eng._inflight_fire_tasks = set()
    eng._fire_ctx = ("http://v", None, [])

    async def run():
        return eng._maybe_fire_on_append()

    assert asyncio.run(run()) is False
    assert eng._fire_diag[7]["no_fireable_entry"] == 1
    assert eng._inflight_fire_tasks == set()


# ------------------------- batch_filled au stade corps : transitoire si l'env admet
# Mesuré en vol le 14/09 (fen 45883, +30-60 s) : ``batch_filled`` au corps =
# file d'admission code PLEINE (64 en attente, 1 worker, plus vieux job 84 s)
# alors que l'env n'avait que 81/224 admis — refus transitoire. La fermeture
# réelle de l'env est traitée par ``fill_closed_env_state`` au tir suivant.
def test_batch_filled_corps_requeue_si_env_ouvert():
    st = _state(_fc(admitted={MATH: 224, CODE: 81}))
    assert engine.reject_is_requeueable(st, "batch_filled", None, env=CODE) is True
    assert engine.reject_is_requeueable(st, "batch_filled", "precommit", env=CODE) is True
    assert engine.reject_is_requeueable(st, "stale_round", None, env=CODE) is True
    assert engine.reject_is_requeueable(st, "out_of_zone", None, env=CODE) is False


def test_batch_filled_corps_jete_si_env_ferme():
    st = _state(_fc(admitted={MATH: 0, CODE: 224}))
    assert engine.reject_is_requeueable(st, "batch_filled", None, env=CODE) is False


def test_batch_filled_corps_requeue_sous_v5():
    st = _state(None)
    assert engine.reject_is_requeueable(st, "batch_filled", None, env=CODE) is True
