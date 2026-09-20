"""20/09 : deux correctifs de SÉLECTION, mesurés sur 35 182 picks étiquetés
(union des sauvegardes ``ooz_v4.jsonl``, 263 fenêtres contiguës).

⛔ FIX A — DÉCALAGE DE RANG GLOBAL : **SUPPRIMÉ, MESURÉ NÉGATIF** (+2,2 groupes
jetés par fenêtre). La motivation — hors-zone non monotone par rang, 51,4 % au
rang 1-10 contre 30,5 % au rang 51-100 — était un artefact de composition
d'ÂGE (le sommet de la table est un cimetière de prompts de l'ère v5 : 14,7 %
de frais, âge médian 7 160 fenêtres ; un prompt vieux est jeté 43,3 % du temps
car le modèle l'a appris). À âge égal le score est plat sur tout le top-250.
Et le balayage atteint DÉJÀ la queue (8,3 picks/fenêtre au-delà du rang 250) :
décaler pousse ~17 picks/fenêtre au-delà de 250 (51,2 %) pour n'en sauver que
17 à 38,1 %. Remplacé par le fix C (soulèvement), qui ne change PAS la
consommation totale.

Ce qui est vrai et exploité par les fix B et C : la variable qui décide est
l'âge de la dernière mesure — 18,5 % de jetés à ≤250 fenêtres contre 35,7 %
pour un prompt jamais mesuré et 43,3 % au-delà de 3 000.
Vérifié en source : le paiement V1 est du FIFO pur — ``_pick_sort_key``
(validateur, batcher.py) renvoie ``group.sequence``, « rate and payload size
are telemetry only ». Seuls comptent « en zone » et « arrivé tôt ».

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


# ───────────── FIX C : SOULÈVEMENT de la bande du 1er bake (20/09) ─────────────
# Mesuré en vol le 20/09 (9 fenêtres, comparaison appariée intra-bake) : les
# slots CLASSÉS du 1er bake sont jetés à **46,7 %** — bien au-dessus des 40,1 %
# de la moyenne des classés, parce qu'en réservant 5 slots au mémo on a
# concentré les classés restants sur le SOMMET du classement, la zone pourrie
# par l'âge (rang 1-10 : 51,4 % de jetés, âge médian 7 160 fenêtres).
# La bande 51-250 est à ~30,5 %. On SOULÈVE donc les K premiers picks depuis le
# rang ``lift`` (K = slots classés du 1er bake = batch − slots mémo de tête) et
# on remet la tête JUSTE DERRIÈRE, pour les bakes tardifs qui ne paient quasi
# rien. La consommation totale de la fenêtre est INCHANGÉE — seul l'ordre
# d'attribution change.
# ⛔ NE PAS confondre avec la ROTATION globale, mesurée NÉGATIVE et supprimée :
# elle décalait TOUTE la consommation de 50 rangs, poussant ~17 picks/fenêtre
# au-delà du rang 250 (51,2 % de jetés) pour n'en sauver que 17 à 38,1 %, soit
# +2,2 groupes jetés par fenêtre.
# Repli : ``RELIQUARY_RANK_LIFT=0`` (défaut) = comportement historique.
KEY = (46520, "aa" * 32, "opencodeinstruct")


def _ranking(ranked):
    r = engine.WindowRanking()
    r._key = KEY                       # clé posée => best() ne reconstruit pas
    r._ranked = list(ranked)
    r._pos = 0
    return r


def _serve(r, n):
    return [r.best(None, None, KEY, (0, 100), set()) for _ in range(n)]


def test_sans_soulevement_le_classement_est_servi_depuis_le_rang_1(monkeypatch):
    monkeypatch.delenv("RELIQUARY_RANK_LIFT", raising=False)
    assert _serve(_ranking(range(10, 20)), 3) == [10, 11, 12]


def test_le_soulevement_sert_la_bande_visee_au_premier_bake(monkeypatch):
    # batch 5, 2 slots mémo => K=3 slots classés au 1er bake, soulevés du rang 3
    monkeypatch.setenv("RELIQUARY_RANK_LIFT", "3")
    monkeypatch.setenv("RELIQUARY_BAKE_BATCH_SIZE", "5")
    monkeypatch.setenv("RELIQUARY_MEMO_HEAD_SLOTS", "2")
    assert _serve(_ranking(range(10, 20)), 3) == [13, 14, 15]


def test_la_tete_ecartee_passe_juste_derriere_pas_a_la_fin(monkeypatch):
    # c'est CE point qui distingue le soulèvement de la rotation : la tête doit
    # revenir tout de suite après, sinon la consommation est poussée en
    # profondeur (rang 250+, 51,2 % de jetés).
    monkeypatch.setenv("RELIQUARY_RANK_LIFT", "3")
    monkeypatch.setenv("RELIQUARY_BAKE_BATCH_SIZE", "5")
    monkeypatch.setenv("RELIQUARY_MEMO_HEAD_SLOTS", "2")
    assert _serve(_ranking(range(10, 20)), 8) == [13, 14, 15, 10, 11, 12, 16, 17]


def test_aucun_prompt_perdu_ni_duplique(monkeypatch):
    monkeypatch.setenv("RELIQUARY_RANK_LIFT", "4")
    monkeypatch.setenv("RELIQUARY_BAKE_BATCH_SIZE", "10")
    monkeypatch.setenv("RELIQUARY_MEMO_HEAD_SLOTS", "5")
    got = _serve(_ranking(range(10, 30)), 20)
    assert sorted(got) == list(range(10, 30))


def test_sans_slot_classe_au_premier_bake_rien_n_est_souleve(monkeypatch):
    # tous les slots de tête au mémo => K=0 => aucun réordonnancement
    monkeypatch.setenv("RELIQUARY_RANK_LIFT", "3")
    monkeypatch.setenv("RELIQUARY_BAKE_BATCH_SIZE", "5")
    monkeypatch.setenv("RELIQUARY_MEMO_HEAD_SLOTS", "5")
    assert _serve(_ranking(range(10, 20)), 3) == [10, 11, 12]


def test_soulevement_au_dela_de_la_tranche_ne_perd_aucun_prompt(monkeypatch):
    monkeypatch.setenv("RELIQUARY_RANK_LIFT", "99")
    monkeypatch.setenv("RELIQUARY_BAKE_BATCH_SIZE", "10")
    monkeypatch.setenv("RELIQUARY_MEMO_HEAD_SLOTS", "5")
    assert _serve(_ranking([10, 11]), 3) == [10, 11, None]


def test_valeur_illisible_vaut_pas_de_soulevement(monkeypatch):
    monkeypatch.setenv("RELIQUARY_RANK_LIFT", "bruit")
    assert _serve(_ranking(range(10, 20)), 2) == [10, 11]


def test_le_soulevement_ne_touche_pas_le_classement_brut(monkeypatch):
    # best_heavy lit self._ranked : son « top K » doit garder son sens
    monkeypatch.setenv("RELIQUARY_RANK_LIFT", "3")
    monkeypatch.setenv("RELIQUARY_BAKE_BATCH_SIZE", "5")
    monkeypatch.setenv("RELIQUARY_MEMO_HEAD_SLOTS", "2")
    r = _ranking(range(10, 20))
    _serve(r, 2)
    assert r._ranked == list(range(10, 20))


def test_le_soulevement_respecte_le_cooldown_et_les_pris(monkeypatch):
    monkeypatch.setenv("RELIQUARY_RANK_LIFT", "3")
    monkeypatch.setenv("RELIQUARY_BAKE_BATCH_SIZE", "5")
    monkeypatch.setenv("RELIQUARY_MEMO_HEAD_SLOTS", "2")
    r = _ranking(range(10, 20))
    r._taken = {14}
    assert [r.best(None, None, KEY, (0, 100), {13}) for _ in range(2)] == [15, 10]


def test_la_rotation_globale_n_est_plus_lue_nulle_part():
    # elle coûtait +2,2 groupes jetés/fenêtre. On vérifie que la variable n'est
    # plus LUE (le nom peut rester cité en docstring pour documenter le rejet).
    src = inspect.getsource(engine)
    assert 'environ.get("RELIQUARY_RANK_OFFSET"' not in src
    assert "RELIQUARY_RANK_OFFSET" not in _os_environ_reads(src)


def _os_environ_reads(src: str) -> str:
    """Les seules lignes qui lisent l'environnement, pour un test sans faux positif."""
    return "\n".join(l for l in src.splitlines() if "environ" in l)
