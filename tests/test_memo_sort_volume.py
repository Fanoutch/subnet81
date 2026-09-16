"""Tri du mémo par VOLUME sous V1 (16/09).

Le mémo classe ses ex-payables par fraîcheur (``best_in_range``) puis, pour les
slots de tête, par run courant > confirmations > fraîcheur — un critère calibré
sous v5, où le bucket valait ``volume / round`` et où générer long PAYAIT. Sous
V1 le paiement est ``fill_closed_fixed_group`` : un groupe payé rapporte pareil
quelle que soit sa longueur, donc la longueur n'est plus qu'un COÛT.

Mesuré le 16/09 : un pick mémo génère 858 tokens de traînard contre 723 pour un
pick classé (+19 %) ; passer de 2 à 5 slots mémo a retardé TOUTE la rafale de
+0,9 s (groupe 1) à +2,5 s (groupe 10), or l'acceptation vaut 100 % avant 18 s
et 62 % à 20-22 s. Sur 149 689 réapparitions d'ex-payables, la re-zone monte peu
avec le volume (83,3 % au quintile court contre 89,2 % au long) — et la bande
4 000-6 000 tient 83,7-85,4 % avec un traînard de 587-672 tokens contre 1 019
au-delà de 10 000. Sous 3 000, la re-zone tombe à 78,5 % ET le rollout le plus
court descend à 30-46 tokens, c'est-à-dire la zone à risque de ``CHALLENGE_K``.

D'où : à validité égale (run courant, confirmations), préférer le PLUS COURT
au-dessus d'un plancher de volume. Défaut inchangé (``fresh``).
"""
from __future__ import annotations

import json

from reliquary.miner.payable_memo import PayableMemo


def _memo(rows):
    """rows = (idx, volume, window, confirmations)."""
    m = PayableMemo()
    for idx, vol, w, conf in rows:
        for _ in range(conf):
            m.update(idx, True, window_n=w, volume=vol)
    return m


def test_defaut_inchange_le_plus_frais_dabord():
    m = _memo([(10, 12000, 5, 1), (11, 4500, 6, 1)])
    assert m.top_in_range(0, 100, n=2) == [11, 10]          # fraîcheur
    assert m.best_in_range(0, 100) == 11


def test_prefer_court_classe_par_volume_croissant():
    m = _memo([(10, 12000, 9, 1), (11, 4500, 5, 1), (12, 8000, 7, 1)])
    assert m.top_in_range(0, 100, n=3, prefer_short=True, min_vol=4000) == [11, 12, 10]


def test_plancher_relegue_les_groupes_trop_courts_sans_les_jeter():
    """Sous le plancher : re-zone 78,5 % et rollout min à 30-46 tokens."""
    m = _memo([(10, 1500, 9, 1), (11, 5000, 5, 1), (12, 2500, 8, 1)])
    out = m.top_in_range(0, 100, n=3, prefer_short=True, min_vol=4000)
    assert out[0] == 11                       # le seul au-dessus du plancher
    assert out[1] == 12 and out[2] == 10      # puis le moins court des courts


def test_volume_inconnu_passe_apres_les_mesures_mais_avant_les_trop_courts():
    m = PayableMemo()
    m.update(10, True, window_n=5)                       # jamais mesuré
    m.update(11, True, window_n=5, volume=5000)
    m.update(12, True, window_n=5, volume=900)
    assert m.top_in_range(0, 100, n=3, prefer_short=True, min_vol=4000) == [11, 10, 12]


def test_la_validite_prime_sur_la_longueur():
    """Run courant et confirmations passent AVANT le volume : le tri par
    longueur ne doit pas réintroduire des prompts sortis de zone."""
    m = _memo([(10, 12000, 900, 3), (11, 4500, 10, 1)])
    assert m.top_in_range(0, 100, n=2, prefer_short=True, min_vol=4000,
                          run_start=800) == [10, 11]
    m2 = _memo([(20, 12000, 900, 5), (21, 4500, 900, 1)])
    assert m2.top_in_range(0, 100, n=2, prefer_short=True, min_vol=4000,
                           run_start=800) == [20, 21]


def test_derniere_mesure_fait_foi_pour_le_volume():
    m = PayableMemo()
    m.update(10, True, window_n=5, volume=12000)
    m.update(10, True, window_n=9, volume=4200)
    m.update(11, True, window_n=9, volume=6000)
    assert m.top_in_range(0, 100, n=2, prefer_short=True, min_vol=4000) == [10, 11]


def test_non_payable_oublie_aussi_le_volume():
    m = _memo([(10, 4500, 5, 1)])
    m.update(10, False)
    assert m.top_in_range(0, 100, n=2, prefer_short=True, min_vol=4000) == []


def test_chargement_jsonl_calcule_le_volume(tmp_path):
    p = tmp_path / "samples.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [
        {"prompt_idx": 10, "in_zone": True, "n_truncated": 0, "window_n": 5,
         "completion_lens": [800] * 16},                       # 12 800
        {"prompt_idx": 11, "in_zone": True, "n_truncated": 0, "window_n": 5,
         "completion_lens": [300] * 16},                       # 4 800
    ]))
    m = PayableMemo()
    m.load_jsonl(str(p))
    assert m.top_in_range(0, 100, n=2, prefer_short=True, min_vol=4000) == [11, 10]
    assert m.top_in_range(0, 100, n=2) == [11, 10] or m.top_in_range(0, 100, n=2) == [10, 11]


# ------------------------------------------------------------ câblage moteur
def test_env_defaut_fresh(monkeypatch):
    from reliquary.miner import engine
    monkeypatch.delenv("RELIQUARY_MEMO_SORT", raising=False)
    monkeypatch.delenv("RELIQUARY_MEMO_MIN_VOL", raising=False)
    assert engine.memo_prefer_short() == (False, 4000)


def test_env_short_et_plancher(monkeypatch):
    from reliquary.miner import engine
    monkeypatch.setenv("RELIQUARY_MEMO_SORT", "short")
    monkeypatch.setenv("RELIQUARY_MEMO_MIN_VOL", "5500")
    assert engine.memo_prefer_short() == (True, 5500)
    monkeypatch.setenv("RELIQUARY_MEMO_MIN_VOL", "n'importe quoi")
    assert engine.memo_prefer_short() == (True, 4000)


def test_head_pick_suit_la_variable(monkeypatch):
    from reliquary.miner import engine
    m = _memo([(10, 12000, 5, 1), (11, 5000, 9, 1)])      # 11 est plus FRAIS
    monkeypatch.delenv("RELIQUARY_MEMO_SORT", raising=False)
    assert engine.memo_head_pick(m, (0, 100), set(), 0) == 11
    monkeypatch.setenv("RELIQUARY_MEMO_SORT", "short")
    monkeypatch.setenv("RELIQUARY_MEMO_MIN_VOL", "4000")
    assert engine.memo_head_pick(m, (0, 100), set(), 0) == 11   # déjà le + court
    m2 = _memo([(10, 5000, 5, 1), (11, 12000, 9, 1)])     # le frais est le + LONG
    assert engine.memo_head_pick(m2, (0, 100), set(), 0) == 10
    monkeypatch.setenv("RELIQUARY_MEMO_SORT", "fresh")
    assert engine.memo_head_pick(m2, (0, 100), set(), 0) == 11
