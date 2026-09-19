"""19/09 (H100) : le bake n'attend plus le GET /state après un flip /miner-state.

Mesuré (flip_diag → bake_start, même fenêtre) : 1,21 s médiane sur H200
(31 fen), 1,31 s sur H100 (24 fen), p90 2,2-2,3 s. Au réveil, le générateur
repasse par ``should_pause_bake(self._last_state, self._state_age_s(), …)`` ;
or ``_last_state`` n'est rempli que par le GET /state complet (669 ko depuis
Dallas) — il date d'avant le trou 503, a plus de 5 s, donc « pause » jusqu'à ce
que /state arrive ET que la pause d'1 s expire. Tout ce que lit le bake
(randomness, fenêtre, checkpoint, cooldown de la tranche) vient déjà de
/miner-state.

Correctif : entre un flip /miner-state et sa confirmation par /state (au plus
RELIQUARY_MS_FLIP_BAKE_MAX_S), l'ancien ``_last_state`` n'est plus consulté —
une fenêtre qui vient d'ouvrir est en collecte, quota vide. La vérification des
poids reste active. Repli : ``RELIQUARY_MS_FLIP_BAKE=0``.
"""
from __future__ import annotations

import asyncio
import inspect
import time
from types import SimpleNamespace

from reliquary.miner import engine

CODE = "opencodeinstruct"


def _eng(*, flip_age_s=0.3, confirmed=False):
    e = engine.MiningEngine.__new__(engine.MiningEngine)
    now = time.time()
    # /state d'AVANT le trou 503 : vieux de 200 s (→ should_pause_bake = True)
    e._last_state = SimpleNamespace(fill_closed=SimpleNamespace(phase="collecting"))
    e._last_state_ts = now - 200.0
    e._submitted_count = {}
    e._cached_window_n = 46420
    e._ms_flip_window = None if confirmed else 46420
    e._ms_flip_ts = now - flip_age_s
    e._weights_out_of_sync = lambda: False
    return e


def test_etat_perime_seul_met_bien_en_pause():
    # témoin : sans flip miner-state en attente, l'ancien /state met en pause
    e = _eng(confirmed=True)
    assert e._bake_paused(CODE, now=time.time()) is True


def test_flip_miner_state_en_attente_ne_bloque_plus_le_bake(monkeypatch):
    monkeypatch.setenv("RELIQUARY_MS_FLIP_BAKE", "1")
    e = _eng()
    assert e._bake_paused(CODE, now=time.time()) is False


def test_repli_flag_off_comportement_historique(monkeypatch):
    monkeypatch.setenv("RELIQUARY_MS_FLIP_BAKE", "0")
    e = _eng()
    assert e._bake_paused(CODE, now=time.time()) is True


def test_flag_off_par_defaut(monkeypatch):
    monkeypatch.delenv("RELIQUARY_MS_FLIP_BAKE", raising=False)
    assert engine.ms_flip_bake_enabled() is False


def test_borne_de_securite_si_state_ne_confirme_jamais(monkeypatch):
    monkeypatch.setenv("RELIQUARY_MS_FLIP_BAKE", "1")
    monkeypatch.setenv("RELIQUARY_MS_FLIP_BAKE_MAX_S", "15")
    e = _eng(flip_age_s=16.0)
    assert e._bake_paused(CODE, now=time.time()) is True


def test_flip_d_une_autre_fenetre_ne_compte_pas(monkeypatch):
    monkeypatch.setenv("RELIQUARY_MS_FLIP_BAKE", "1")
    e = _eng()
    e._cached_window_n = 46421
    assert e._bake_paused(CODE, now=time.time()) is True


def test_poids_desynchronises_bloquent_toujours(monkeypatch):
    monkeypatch.setenv("RELIQUARY_MS_FLIP_BAKE", "1")
    e = _eng()
    e._weights_out_of_sync = lambda: True
    assert e._bake_paused(CODE, now=time.time()) is True


def test_flip_miner_state_date_le_flip(monkeypatch):
    monkeypatch.setattr(engine, "study_dump", lambda *a, **k: None)
    e = engine.MiningEngine.__new__(engine.MiningEngine)
    e._pool_lock = asyncio.Lock()
    e._pool, e._retry_by_env, e._cooldowns = [], {CODE: []}, {CODE: set()}
    e._cached_randomness = "bb" * 32
    e._active_prompt_range = lambda w, r: (0, 1)

    async def pull(state):
        return False
    e._apply_checkpoint_pull = pull
    e._signal_flip = lambda: None
    view = SimpleNamespace(window_n=46420, randomness="aa" * 32,
                           window_opened_at=None, cooldowns={CODE: set()},
                           checkpoint_repo_id="r", checkpoint_revision="x",
                           checkpoint_n=1)
    assert asyncio.run(e._flip_from_miner_state(view, t_send=10.0, t_recv=11.5))
    assert e._ms_flip_window == 46420 and e._ms_flip_ts == 11.5


def test_le_generateur_passe_par_bake_paused():
    src = inspect.getsource(engine.MiningEngine._generator_loop)
    assert "self._bake_paused(env_name" in src
    assert "should_pause_bake(" not in src


def test_confirmation_par_state_journalise_le_delai_evite():
    # mesure : combien de temps le bake aurait attendu /state (à comparer au
    # flip → bake_start, journal flip_diag / bake_start)
    src = inspect.getsource(engine.MiningEngine._trigger_loop)
    i = src.index("miner-state flip confirmé par /state")
    assert "ms_flip_confirm: window=%d /state reçu %.2f s après le flip" in src[i:i + 900]
