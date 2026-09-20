"""20/09 : deux correctifs de SÉLECTION, mesurés sur 35 182 picks étiquetés
(union des sauvegardes ``ooz_v4.jsonl``, 263 fenêtres contiguës).

FIX A — DÉCALAGE DE RANG : ⛔ **MESURÉ NÉGATIF, JAMAIS ARMÉ** (−0,55 groupe
valide/fenêtre). Ces tests verrouillent un chemin qui doit rester INERTE.
La motivation — hors-zone non monotone par rang, 51,4 % au rang 1-10 contre
30,5 % au rang 51-100 — était un artefact de composition d'ÂGE (le sommet de la
table est un cimetière de prompts de l'ère v5 : 14,7 % de frais, âge médian
7 160 fenêtres ; un prompt vieux est jeté 43,3 % du temps car le modèle l'a
appris). À âge égal le score est plat sur tout le top-250. Et le balayage
atteint DÉJÀ la queue (8,3 picks/fenêtre au-delà du rang 250) : les slots
libérés retombent là, où les jetés valent 51,2 %.
Ce qui est vrai et exploité par le fix B : la variable qui décide est l'âge de
la dernière mesure — 18,5 % de jetés à ≤250 fenêtres contre 35,7 % pour un
prompt jamais mesuré et 43,3 % au-delà de 3 000.
Ce qui reste vérifié en source : le paiement V1 est du FIFO pur —
``_pick_sort_key`` (validateur, batcher.py) renvoie ``group.sequence``, « rate
and payload size are telemetry only ». Seuls comptent « en zone » et « tôt ».
Défaut ``RELIQUARY_RANK_OFFSET=0`` = comportement historique.

FIX B — MÉMO CONCENTRÉ SUR LE 1er BAKE. Jetés par profondeur du mémo :
d1 9,2 % · d3-4 14,6 % · d5-6 17,7 % · d17-24 30,4 % · **d25-40 52,7 %**.
Servir 5 slots coûte 13,2 % (moins que 3 slots : 13,8 %), 6 slots ~16 %,
contre **40,1 %** pour un slot classé. Mais la réserve FRAÎCHE (âge ≤250 fen)
ne vaut que 19,3 par tranche alors qu'on consomme 39-45 picks mémo par
fenêtre (3 slots × 13-15 bakes) : on épuise le bon mémo sur des bakes tardifs
qui ne paient rien. On réserve donc le mémo au 1er bake.
Repli : ``RELIQUARY_MEMO_HEAD_SLOTS_LATE`` non posé = même valeur qu'au 1er
bake = comportement historique.
"""
from __future__ import annotations

import inspect

from reliquary.miner import engine


# ─────────────────────────── FIX A : décalage de rang ───────────────────────
KEY = (46520, "aa" * 32, "opencodeinstruct")


def _ranking(ranked):
    r = engine.WindowRanking()
    r._key = KEY                       # clé posée => best() ne reconstruit pas
    r._ranked = list(ranked)
    r._pos = 0
    return r


def test_sans_decalage_le_classement_est_servi_depuis_le_rang_1(monkeypatch):
    monkeypatch.delenv("RELIQUARY_RANK_OFFSET", raising=False)
    r = _ranking([10, 11, 12, 13, 14])
    assert r.best(None, None, KEY, (0, 100), set()) == 10


def test_le_decalage_saute_la_tete_du_classement(monkeypatch):
    monkeypatch.setenv("RELIQUARY_RANK_OFFSET", "2")
    r = _ranking([10, 11, 12, 13, 14])
    assert [r.best(None, None, KEY, (0, 100), set()) for _ in range(3)] == [12, 13, 14]


def test_la_tete_ecartee_est_servie_en_dernier_recours(monkeypatch):
    # jamais d'affamement : une fois la bonne bande épuisée, on reprend la tête
    monkeypatch.setenv("RELIQUARY_RANK_OFFSET", "2")
    r = _ranking([10, 11, 12])
    got = [r.best(None, None, KEY, (0, 100), set()) for _ in range(4)]
    assert got == [12, 10, 11, None]


def test_un_decalage_plus_grand_que_la_tranche_ne_perd_aucun_prompt(monkeypatch):
    monkeypatch.setenv("RELIQUARY_RANK_OFFSET", "99")
    r = _ranking([10, 11])
    assert [r.best(None, None, KEY, (0, 100), set()) for _ in range(3)] == [10, 11, None]


def test_le_decalage_respecte_le_cooldown_et_les_pris(monkeypatch):
    monkeypatch.setenv("RELIQUARY_RANK_OFFSET", "1")
    r = _ranking([10, 11, 12, 13])
    r._taken = {12}
    assert [r.best(None, None, KEY, (0, 100), {11}) for _ in range(2)] == [13, 10]


def test_le_decalage_ne_touche_pas_le_classement_brut(monkeypatch):
    # best_heavy lit self._ranked : son « top 50 » doit garder son sens
    monkeypatch.setenv("RELIQUARY_RANK_OFFSET", "2")
    r = _ranking([10, 11, 12, 13])
    r.best(None, None, KEY, (0, 100), set())
    assert r._ranked == [10, 11, 12, 13]


def test_une_nouvelle_fenetre_repart_de_la_bonne_bande(monkeypatch):
    monkeypatch.setenv("RELIQUARY_RANK_OFFSET", "2")
    r = _ranking([10, 11, 12, 13])
    r.best(None, None, KEY, (0, 100), set())
    autre = (46521, "bb" * 32, "opencodeinstruct")
    r._build = lambda env, model, rng, cd: setattr(r, "_ranked", [20, 21, 22, 23])
    assert r.best(None, None, autre, (0, 100), set()) == 22


# ──────────────────── FIX B : mémo réservé au 1er bake ──────────────────────
def test_defaut_les_slots_memo_sont_les_memes_a_tous_les_bakes(monkeypatch):
    monkeypatch.setenv("RELIQUARY_MEMO_HEAD_SLOTS", "3")
    monkeypatch.delenv("RELIQUARY_MEMO_HEAD_SLOTS_LATE", raising=False)
    assert engine.memo_head_slots_for_bake(True) == 3
    assert engine.memo_head_slots_for_bake(False) == 3


def test_le_memo_peut_etre_reserve_au_premier_bake(monkeypatch):
    monkeypatch.setenv("RELIQUARY_MEMO_HEAD_SLOTS", "6")
    monkeypatch.setenv("RELIQUARY_MEMO_HEAD_SLOTS_LATE", "0")
    assert engine.memo_head_slots_for_bake(True) == 6
    assert engine.memo_head_slots_for_bake(False) == 0


def test_valeur_illisible_retombe_sur_les_slots_du_premier_bake(monkeypatch):
    monkeypatch.setenv("RELIQUARY_MEMO_HEAD_SLOTS", "5")
    monkeypatch.setenv("RELIQUARY_MEMO_HEAD_SLOTS_LATE", "bruit")
    assert engine.memo_head_slots_for_bake(False) == 5


def _eng(window):
    e = engine.MiningEngine.__new__(engine.MiningEngine)
    e._cached_window_n = window
    return e


def test_le_premier_bake_de_la_fenetre_est_reconnu():
    e = _eng(46520)
    assert e._is_first_bake_of_window() is True


def test_apres_un_bake_ce_n_est_plus_le_premier():
    e = _eng(46520)
    e._note_bake_started()
    assert e._is_first_bake_of_window() is False


def test_le_compteur_repart_a_la_fenetre_suivante():
    e = _eng(46520)
    e._note_bake_started()
    e._cached_window_n = 46521
    assert e._is_first_bake_of_window() is True


def test_le_bake_signale_son_depart():
    src = inspect.getsource(engine.MiningEngine._bake_stream_fire)
    assert "_note_bake_started()" in src


def test_la_boucle_de_generation_utilise_les_slots_par_bake():
    src = inspect.getsource(engine.MiningEngine._generator_loop)
    assert "memo_head_slots_for_bake(" in src
    assert "_is_first_bake_of_window()" in src


def test_le_slot_memo_herite_reste_gate_sur_la_valeur_CONFIGUREE():
    # Piège : le chemin historique « 3e slot du sprint au mémo » s'active quand
    # les slots de tête valent 0. Si les bakes tardifs passaient à 0, il se
    # réveillerait et re-servirait du mémo là où on veut justement l'économiser.
    # Il doit donc rester gaté sur MEMO_HEAD_SLOTS, pas sur la valeur du bake.
    src = inspect.getsource(engine.MiningEngine._generator_loop)
    assert "_head_slots_cfg == 0" in src


def test_fenetre_inconnue_compte_comme_premier_bake():
    # au démarrage _cached_window_n vaut None : ne pas conclure « bake tardif »,
    # sinon le tout premier bake du process perdrait ses slots mémo.
    e = engine.MiningEngine.__new__(engine.MiningEngine)
    e._cached_window_n = None
    assert e._is_first_bake_of_window() is True
