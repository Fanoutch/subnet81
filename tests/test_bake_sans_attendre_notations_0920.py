"""20/09 : le bake suivant ne attend plus les notations du bake précédent.

Mesuré (`bake_diag`, champ ``attente_notations``, 872 bakes de la nuit H100) :
1,35 s médiane, 3,37 s au p90 de GPU À L'ARRÊT entre le dernier groupe livré et
le bake suivant, le temps que les grades/preuves du lot finissent.

Pourquoi le code attend (engine.py, fin de ``_bake_stream_fire``) : il vide
ensuite ``_phase1_cache`` d'un bloc, et une tâche encore en vol qui trouverait
le cache vide RÉGÉNÉRERAIT son groupe en appel vLLM mono-prompt (~40 s).

Mais le cache est indexé par ``(prompt_idx, randomness, checkpoint_hash)`` et
chaque entrée est ``pop()``ée à la consommation : une entrée d'une autre fenêtre
NE PEUT PAS être servie. Le vidage global n'est donc que de l'hygiène. On le
remplace par une purge SÉLECTIVE (on ne garde que la fenêtre courante) et on
n'attend plus. Les tâches encore en vol restent référencées pour être attendues
plus tard. Repli : ``RELIQUARY_BAKE_WAIT_GRADES=1`` (défaut = comportement
historique).

⚠️ À MESURER en vol : le bake suivant démarre pendant que les preuves du lot
précédent tournent encore — la contention GPU peut coûter plus que les 1,35 s
gagnées (leçons du bake 14 et du sprint 3). Juge : ms/token des bakes 2+,
heure des tirs, payés.
"""
from __future__ import annotations

import asyncio
import inspect
import types

from reliquary.miner import engine

RND, CK = "aa" * 32, "ckpt1"


def _eng(cache=None, tasks=None):
    e = engine.MiningEngine.__new__(engine.MiningEngine)
    e._phase1_cache = dict(cache or {})
    e._stream_grade_tasks = list(tasks or [])
    return e


def test_flag_par_defaut_attend_les_notations(monkeypatch):
    monkeypatch.delenv("RELIQUARY_BAKE_WAIT_GRADES", raising=False)
    assert engine.bake_wait_grades_enabled() is True
    monkeypatch.setenv("RELIQUARY_BAKE_WAIT_GRADES", "0")
    assert engine.bake_wait_grades_enabled() is False


def test_defaut_attend_la_tache_et_vide_le_cache(monkeypatch):
    monkeypatch.delenv("RELIQUARY_BAKE_WAIT_GRADES", raising=False)

    async def go():
        fini = []

        async def grade():
            await asyncio.sleep(0.05)
            fini.append(1)
        e = _eng(cache={(7, RND, CK): "g"}, tasks=[asyncio.create_task(grade())])
        await e._finish_bake(randomness=RND, checkpoint_hash=CK)
        return fini, e._phase1_cache, e._stream_grade_tasks
    fini, cache, tasks = asyncio.run(go())
    assert fini == [1]                      # la notation a été attendue
    assert cache == {} and tasks == []      # cache vidé comme avant


def test_sans_attente_le_bake_rend_la_main_avant_la_notation(monkeypatch):
    monkeypatch.setenv("RELIQUARY_BAKE_WAIT_GRADES", "0")

    async def go():
        fini = []

        async def grade():
            await asyncio.sleep(0.30)
            fini.append(1)
        t = asyncio.create_task(grade())
        e = _eng(cache={(7, RND, CK): "g"}, tasks=[t])
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        await e._finish_bake(randomness=RND, checkpoint_hash=CK)
        dt = loop.time() - t0
        # l'état des tâches se lit DANS la boucle : après asyncio.run(), la
        # tâche aurait eu le temps de finir et done() serait vrai à tort.
        etat = (list(fini), dict(e._phase1_cache),
                [t.done() for t in e._stream_grade_tasks])
        await t
        return dt, etat
    dt, (fini, cache, tasks) = asyncio.run(go())
    assert dt < 0.15                        # on n'a pas attendu les 0,30 s
    assert fini == []                       # la notation tournait encore
    assert tasks == [False]                 # 1 tâche conservée, encore en vol
    assert cache == {(7, RND, CK): "g"}     # l'entrée de CETTE fenêtre survit


def test_sans_attente_purge_les_entrees_d_une_autre_fenetre(monkeypatch):
    monkeypatch.setenv("RELIQUARY_BAKE_WAIT_GRADES", "0")
    e = _eng(cache={(7, RND, CK): "a", (8, "bb" * 32, CK): "vieux",
                    (9, RND, "autre_ckpt"): "vieux2"})
    asyncio.run(e._finish_bake(randomness=RND, checkpoint_hash=CK))
    assert set(e._phase1_cache) == {(7, RND, CK)}


def test_sans_attente_les_taches_finies_sont_retirees(monkeypatch):
    monkeypatch.setenv("RELIQUARY_BAKE_WAIT_GRADES", "0")

    async def go():
        async def rien():
            return None
        t = asyncio.create_task(rien())
        await asyncio.sleep(0)
        await t
        e = _eng(tasks=[t])
        await e._finish_bake(randomness=RND, checkpoint_hash=CK)
        return e._stream_grade_tasks
    assert asyncio.run(go()) == []


def test_le_bake_passe_par_finish_bake():
    src = inspect.getsource(engine.MiningEngine._bake_stream_fire)
    assert "self._finish_bake(" in src
    # plus de vidage global en dur dans le bake
    assert "self._phase1_cache = {}" not in src.split("_finish_bake")[-1]


# ── Verrou de sûreté : sans l'attente, une notation peut finir APRÈS le flip ──
# Le garde-fou existant (_post_grade_entry) ne vérifie que la TRANCHE de la
# fenêtre courante. Deux tranches consécutives peuvent se recouvrir par hasard
# (5 000 indices sur 2,48 M) : l'entrée passerait alors dans le pool avec les
# tokens de l'ANCIENNE randomness → seed_mismatch, donc dette de preuve. On
# estampille donc l'entrée avec la randomness de son bake et on la refuse si la
# fenêtre a changé.
def _eng_pool(cached_rnd):
    e = engine.MiningEngine.__new__(engine.MiningEngine)
    e._pool, e._pool_lock = [], asyncio.Lock()
    e._cached_randomness, e._cached_window_n = cached_rnd, 46500
    e._local_n = 7
    e._active_prompt_range = lambda w, r, env=None: (0, 10_000)  # tranche PERMISSIVE
    e._entry_env_name = lambda entry: "opencodeinstruct"
    e._last_state = None                      # v5 : should_skip_pool = False
    e._submitted_count = {}
    e._maybe_fire_on_append = lambda: None
    return e


def _entry(rnd):
    return {"prompt_idx": 42, "problem": {}, "rollouts": [], "checkpoint_n": 7,
            "env_name": "opencodeinstruct", "randomness": rnd}


def test_entree_d_une_autre_fenetre_refusee_meme_dans_la_tranche():
    e = _eng_pool("bb" * 32)
    entries = []
    asyncio.run(e._post_grade_entry(_entry("aa" * 32), 42, entries, None))
    assert e._pool == [] and entries == []


def test_entree_de_la_fenetre_courante_acceptee():
    e = _eng_pool("aa" * 32)
    entries = []
    asyncio.run(e._post_grade_entry(_entry("aa" * 32), 42, entries, None))
    assert len(e._pool) == 1 and len(entries) == 1


def test_entree_sans_randomness_reste_acceptee():
    # compatibilité : les autres chemins ne l'estampillent pas
    e = _eng_pool("aa" * 32)
    entries = []
    ent = _entry("aa" * 32)
    del ent["randomness"]
    asyncio.run(e._post_grade_entry(ent, 42, entries, None))
    assert len(e._pool) == 1


def test_le_bake_publie_sa_randomness_dans_le_contexte():
    # contextvars : hérité par les tâches de notation et par asyncio.to_thread,
    # sans toucher AUCUNE signature (les doublures de test restent valides).
    src = inspect.getsource(engine.MiningEngine._bake_stream_fire)
    assert "_BAKE_RANDOMNESS.set(randomness)" in src


def _eng_gen(cached_rnd, vus):
    e = engine.MiningEngine.__new__(engine.MiningEngine)
    e._cached_randomness = cached_rnd
    e._generate_m_rollouts = lambda problem, rnd, env, prompt_idx=0: (
        vus.append(rnd) or [])
    return e


def test_pre_bake_entry_utilise_la_randomness_de_SON_bake():
    # une notation qui finit après le flip doit lire le cache phase-1 avec la
    # clé de SON bake, sinon elle rate le cache et RÉGÉNÈRE (~40 s, le poison).
    vus = []
    e = _eng_gen("nouvelle", vus)
    tok = engine._BAKE_RANDOMNESS.set("ancienne")
    try:
        e._pre_bake_entry(7, {"prompt": "p"}, 1,
                          types.SimpleNamespace(name="opencodeinstruct"))
    except Exception:
        pass                                   # peu importe la suite du chemin
    finally:
        engine._BAKE_RANDOMNESS.reset(tok)
    assert vus == ["ancienne"]


def test_pre_bake_entry_sans_contexte_prend_la_randomness_courante():
    vus = []
    e = _eng_gen("courante", vus)
    try:
        e._pre_bake_entry(7, {"prompt": "p"}, 1,
                          types.SimpleNamespace(name="opencodeinstruct"))
    except Exception:
        pass
    assert vus == ["courante"]
