"""Compteurs de diagnostic de la file d'envoi (24/08).

Contexte mesuré ce jour-là : on produit 8 à 10 groupes postables par fenêtre,
on n'en TENTE que 5,5 et on en place 3,0. Trois à quatre groupes disparaissent
sans laisser la moindre trace dans les journaux, parce que les deux fonctions
qui décident d'envoyer écartent silencieusement :

  * ``_maybe_fire_on_append`` retourne ``False`` sur une condition composée de
    six gardes — dont ``len(_inflight_fire_tasks) >= _MAX_INFLIGHT_FIRES``.
    Quand le validateur rame (18 ``ReadTimeout`` mesurés, POST à 4-18 s), les
    créneaux se remplissent et un groupe PRÊT attend sans qu'on le sache.
  * ``_fire_for_window`` jette les entrées en cooldown et hors-tranche via un
    ``continue`` muet.

Ces tests verrouillent des COMPTEURS, pas un comportement : les valeurs de
retour et les entrées effectivement tirées doivent rester identiques.
"""
from __future__ import annotations

import asyncio
import types

import pytest

from reliquary.miner import engine as eng
from reliquary.miner.engine import MiningEngine
from reliquary.miner.engine import MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW as CAP
from reliquary.protocol.submission import WindowState


def _state(window_n=100, cooldown=()):
    return types.SimpleNamespace(
        window_n=window_n,
        randomness="ab" * 32,
        state=WindowState.OPEN,
        cooldown_prompts=list(cooldown),
    )


def _engine_for_append(*, inflight=0, submitted=0, sealed=None, window_n=100):
    """Moteur minimal pour exercer ``_maybe_fire_on_append``."""
    e = MiningEngine.__new__(MiningEngine)
    st = _state(window_n)
    e._last_state = st
    e._fire_ctx = ("http://val", object(), [])
    e._cached_randomness = st.randomness
    e._cached_window_n = st.window_n
    e._sealed_window = sealed
    e._pool = [{"prompt_idx": 1, "env_name": "opencodeinstruct"}]
    e._pool_lock = asyncio.Lock()
    e._submitted_count = {window_n: submitted}
    e._inflight_fire_tasks = set(range(inflight))  # sentinelles: seul len() compte
    e._fire_as_ready = lambda *a, **k: True
    return e


def test_compteur_file_saturee():
    """File d'envoi pleine : on n'envoie pas, et on doit SAVOIR pourquoi."""
    e = _engine_for_append(inflight=eng._MAX_INFLIGHT_FIRES)

    assert e._maybe_fire_on_append() is False
    assert e._fire_diag[100]["inflight_saturated"] == 1


def test_compteur_budget_epuise():
    """Quota de la fenêtre consommé : motif distinct de la file saturée."""
    e = _engine_for_append(submitted=CAP)

    assert e._maybe_fire_on_append() is False
    assert e._fire_diag[100]["budget_exhausted"] == 1
    assert e._fire_diag[100]["inflight_saturated"] == 0


def test_compteur_fenetre_scellee():
    """Fenêtre scellée : ni une saturation ni un manque de budget."""
    e = _engine_for_append(sealed=100)

    assert e._maybe_fire_on_append() is False
    assert e._fire_diag[100]["sealed"] == 1


def test_aucun_compteur_quand_le_tir_part():
    """Chemin vert : le tir part, aucun motif de blocage n'est compté."""
    async def _run():
        e = _engine_for_append()
        fired = []

        async def _fire(*a, **k):
            fired.append(1)

        e._fire_for_window = _fire
        assert e._maybe_fire_on_append() is True
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert fired == [1]
        assert sum(e._fire_diag[100].values()) == 0

    asyncio.run(_run())


def _engine_for_fire(pool_idx, cooldown=(), prompt_range=None):
    """Moteur minimal pour exercer ``_fire_for_window``."""
    e = MiningEngine.__new__(MiningEngine)
    e._pool = [{"prompt_idx": i, "env_name": "opencodeinstruct"} for i in pool_idx]
    e._pool_lock = asyncio.Lock()
    e._submitted_count = {}
    e._sealed_window = None
    e.active_envs = ["opencodeinstruct"]
    e.envs = {"opencodeinstruct": object()}
    e._entry_env_name = lambda entry: "opencodeinstruct"
    e._active_prompt_range = lambda *a, **k: prompt_range
    e._persist_path = None

    async def _submit_entry(entry, state, url, client, results):
        return (entry, types.SimpleNamespace(accepted=True, reason="submitted"))

    e._submit_entry = _submit_entry
    return e


def test_compteur_entrees_en_cooldown():
    """Les prompts en cooldown sont jetés — il faut les compter."""
    async def _run():
        e = _engine_for_fire([1, 2, 3], cooldown=(1, 2))
        await e._fire_for_window(
            _state(100, cooldown=(1, 2)), "http://val", object(), [], budget=CAP,
        )
        assert e._fire_diag[100]["dropped_cooldown"] == 2

    asyncio.run(_run())


def test_compteur_entrees_hors_tranche():
    """Hors [lo, hi) : jeté aussi, et pour un motif DIFFÉRENT du cooldown."""
    async def _run():
        e = _engine_for_fire([1, 2, 50], prompt_range=(0, 10))
        await e._fire_for_window(
            _state(100), "http://val", object(), [], budget=CAP,
        )
        assert e._fire_diag[100]["dropped_out_of_slice"] == 1
        assert e._fire_diag[100]["dropped_cooldown"] == 0

    asyncio.run(_run())


def test_compteur_reste_en_pool_faute_de_budget():
    """Budget insuffisant : les entrées restent en pool, on doit le savoir."""
    async def _run():
        e = _engine_for_fire([1, 2, 3, 4, 5])
        await e._fire_for_window(
            _state(100), "http://val", object(), [], budget=2,
        )
        assert e._fire_diag[100]["left_no_budget"] == 3

    asyncio.run(_run())


def test_les_compteurs_sont_vides_pour_les_fenetres_closes():
    """Les compteurs d'une fenêtre ne sont complets qu'une fois close.

    On les émet donc quand une fenêtre PLUS RÉCENTE s'ouvre, et on purge —
    sinon le dictionnaire grossit sans fin sur un mineur qui tourne des jours.
    """
    e = MiningEngine.__new__(MiningEngine)
    e._fire_diag[100]["inflight_saturated"] = 3
    e._fire_diag[100]["dropped_cooldown"] = 2
    e._fire_diag[101]["sealed"] = 1  # fenêtre COURANTE : ne doit pas partir

    emis = e._flush_fire_diag(current_window=101)

    assert [r["window_n"] for r in emis] == [100]
    assert emis[0]["inflight_saturated"] == 3
    assert emis[0]["dropped_cooldown"] == 2
    assert 100 not in e._fire_diag          # purgée
    assert e._fire_diag[101]["sealed"] == 1  # la courante survit


def test_flush_ignore_les_fenetres_sans_evenement():
    """Une fenêtre où rien n'a été écarté ne produit pas de ligne inutile."""
    e = MiningEngine.__new__(MiningEngine)
    e._fire_diag[100]  # touchée mais vide (defaultdict)

    assert e._flush_fire_diag(current_window=101) == []
