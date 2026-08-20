"""File d'envoi concurrente (20/08) — le plafond par fenêtre reste ÉTANCHE.

Jusqu'ici un seul tir pouvait être en vol : l'entrée N+1 attendait la fin
complète du POST de l'entrée N (~3-5 s), d'où des entrées 2 à 8 horodatées à
+16-23 s alors qu'elles étaient prêtes bien avant. Le rang étant estampillé à
l'arrivée du precommit, ce retard coûtait des places directement.

Le risque de la levée du verrou est UNIQUE et connu : l'appelant calcule
``budget = MAX - _submitted_count[w]`` HORS ``_pool_lock``. Deux tirs
concurrents liraient donc le même compteur et se partageraient DEUX FOIS le
même budget → dépassement de MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW (le
validateur rejette, et surtout on brûle des slots pour rien).

Ces tests verrouillent les deux propriétés :
  1. sous concurrence, la somme des entrées tirées ne dépasse JAMAIS le
     plafond, même si chaque appelant croit disposer du budget entier ;
  2. le défaut (_MAX_INFLIGHT_FIRES=1) laisse le comportement inchangé.
"""
from __future__ import annotations

import asyncio
import types

import pytest

from reliquary.miner import engine as eng
from reliquary.miner.engine import MiningEngine
from reliquary.protocol.submission import WindowState
from reliquary.miner.engine import MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW as CAP


def _state(window_n=100):
    return types.SimpleNamespace(
        window_n=window_n,
        randomness="ab" * 32,
        state=WindowState.OPEN,
        cooldown_prompts=[],
    )


def _engine(pool_size, submitted):
    e = MiningEngine.__new__(MiningEngine)
    e._pool = [{"prompt_idx": i, "env_name": "opencodeinstruct"}
               for i in range(pool_size)]
    e._pool_lock = asyncio.Lock()
    e._submitted_count = {}
    e._sealed_window = None
    e.active_envs = ["opencodeinstruct"]
    e.envs = {"opencodeinstruct": object()}
    e._entry_env_name = lambda entry: "opencodeinstruct"
    e._active_prompt_range = lambda *a, **k: None
    submitted_here: list[int] = []

    async def _submit_entry(entry, state, url, client, results):
        # POST réaliste : lent, et surtout il rend la main à la boucle — c'est
        # précisément pendant cette attente qu'un second tir peut démarrer.
        await asyncio.sleep(0.02)
        submitted_here.append(entry["prompt_idx"])
        return (entry, types.SimpleNamespace(accepted=True, reason="submitted"))

    e._submit_entry = _submit_entry
    return e, submitted_here


@pytest.mark.asyncio
async def test_budget_reclampe_sous_le_lock_sous_concurrence():
    """Deux tirs concurrents, chacun croyant avoir le budget PLEIN."""
    e, submitted = _engine(pool_size=3 * CAP, submitted=0)
    st = _state()

    # Les deux appelants ont lu _submitted_count AVANT que l'autre ne réserve :
    # c'est exactement la course qu'on veut couvrir.
    await asyncio.gather(
        e._fire_for_window(st, "http://v", None, [], budget=CAP),
        e._fire_for_window(st, "http://v", None, [], budget=CAP),
        e._fire_for_window(st, "http://v", None, [], budget=CAP),
    )

    assert e._submitted_count[st.window_n] == CAP, (
        f"réservé {e._submitted_count[st.window_n]} > plafond {CAP}"
    )
    assert len(submitted) == CAP, f"{len(submitted)} POST pour un plafond de {CAP}"
    assert len(set(submitted)) == len(submitted), "une entrée tirée deux fois"
    # le reste du pool est conservé, pas perdu
    assert len(e._pool) == 3 * CAP - CAP


@pytest.mark.asyncio
async def test_budget_deja_consomme_ne_tire_rien():
    """Fenêtre déjà pleine : un tir tardif ne doit rien envoyer."""
    e, submitted = _engine(pool_size=5, submitted=0)
    st = _state()
    e._submitted_count[st.window_n] = CAP

    await e._fire_for_window(st, "http://v", None, [], budget=CAP)

    assert submitted == []
    assert e._submitted_count[st.window_n] == CAP
    assert len(e._pool) == 5, "le pool doit être conservé pour la fenêtre suivante"


def test_defaut_un_seul_tir_en_vol(monkeypatch):
    """Sans la variable d'env, le comportement reste celui d'aujourd'hui."""
    monkeypatch.delenv("RELIQUARY_MAX_INFLIGHT_FIRES", raising=False)
    import importlib
    assert eng._MAX_INFLIGHT_FIRES >= 1
    # la valeur module est lue à l'import : on vérifie la formule elle-même
    import os
    os.environ["RELIQUARY_MAX_INFLIGHT_FIRES"] = "3"
    try:
        importlib.reload(eng)
        assert eng._MAX_INFLIGHT_FIRES == 3
    finally:
        del os.environ["RELIQUARY_MAX_INFLIGHT_FIRES"]
        importlib.reload(eng)
        assert eng._MAX_INFLIGHT_FIRES == 1
