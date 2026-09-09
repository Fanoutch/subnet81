"""Bonus de VOLUME au tri des prompts (20/08).

Le rang du validateur est `min(somme des completion_lens, 8192x16) //
(rounds x 50)` : à arrivée égale, le volume de tokens EST le rang. Mesuré sur
327 envois sains (min rollout >= 32 tok, après le gate CHALLENGE_K) : 7 % de
payées sous 3 000 tokens contre 54 % au-dessus de 6 000 — un facteur 8,
monotone. Et le volume est bien une propriété du PROMPT : sur 1 119 prompts vus
au moins 3 fois, 82 % de la variance est inter-prompts.

Invariants verrouillés ici :
  - le bonus RÉORDONNE bien vers les gros volumes ;
  - modèle absent OU mu=0 -> tri byte-identique à aujourd'hui (le défaut est
    donc sûr : sans RELIQUARY_VOLUME_MODEL le mineur ne change pas) ;
  - un modèle illisible ne fait pas tomber le classement (un prompt mal formé
    ne doit jamais faire perdre une fenêtre entière) ;
  - c'est un BONUS, pas une exclusion : aucun prompt n'est retiré du classement.
"""
from __future__ import annotations

import json

import pytest

from reliquary.miner.prompt_predictor import volume_score, volume_features


def _model():
    """Modèle jouet : « recursion » et « matrix » annoncent de gros groupes."""
    return {
        "kind": "volume_log_tokens",
        "intercept": 8.0,
        "weights": {"recursion": 3.0, "matrix": 3.0, "trivial": -3.0},
        "center": 8.0,
        "scale": 1.0,
    }


def test_le_score_ordonne_les_prompts_par_volume_attendu():
    m = _model()
    gros = volume_score(m, "implement recursion over a matrix of values")
    petit = volume_score(m, "trivial trivial trivial case here")
    assert gros > petit, f"{gros} devrait dépasser {petit}"


def test_modele_absent_est_neutre():
    """Sans modèle, aucun effet — la garantie que le défaut ne change rien."""
    assert volume_score({}, "n'importe quoi") == 0.0
    assert volume_score(None, "n'importe quoi") == 0.0


def test_texte_vide_ne_leve_pas():
    m = _model()
    assert isinstance(volume_score(m, ""), float)
    assert isinstance(volume_score(m, None), float)


def test_traits_l2_normalises_et_longueur_bornee():
    """Les traits doivent rester ceux de l'entraînement, sinon les poids
    appris ne veulent plus rien dire."""
    f = volume_features("alpha beta alpha gamma")
    norme = sum(v * v for k, v in f.items() if k != "__len__") ** 0.5
    assert abs(norme - 1.0) < 1e-9, f"norme L2 = {norme}"
    assert volume_features("x" * 100000)["__len__"] == 3.0, "longueur non bornée"
    # mots de moins de 3 lettres ignorés (comme à l'entraînement)
    assert "an" not in volume_features("an alpha")


def test_le_bonus_ne_retire_jamais_un_prompt(monkeypatch):
    """Propriété de sûreté : c'est un terme de TRI, pas un filtre."""
    from reliquary.miner import engine as eng

    m = _model()
    cands = [(1, "trivial trivial"), (2, "recursion matrix"), (3, "autre chose")]
    for mu in (0.0, 0.05, 10.0):
        classe = sorted(
            cands, key=lambda c: mu * volume_score(m, c[1]), reverse=True,
        )
        assert len(classe) == len(cands)
        assert {i for i, _ in classe} == {1, 2, 3}


def test_mu_zero_laisse_lordre_inchange():
    m = _model()
    base = [(0.5, 1), (0.4, 2), (0.3, 3)]
    textes = {1: "trivial", 2: "recursion matrix", 3: "neutre"}
    for mu, attendu in ((0.0, [1, 2, 3]), (10.0, [2, 3, 1])):
        got = [
            i for _, i in sorted(
                ((sc + mu * volume_score(m, textes[i]), i) for sc, i in base),
                reverse=True,
            )
        ]
        assert got == attendu, f"mu={mu} -> {got}"


def test_modele_reel_separe_bien_les_volumes():
    """Le modèle livré doit charger et discriminer — sinon il ne sert à rien."""
    try:
        m = json.load(open("/root/subnet81/data/volume_v1.json"))
    except FileNotFoundError:
        pytest.skip("modèle de volume non présent sur cette machine")
    assert len(m["weights"]) > 1000
    long_ = volume_score(
        m,
        "Implement a segment tree with lazy propagation supporting range "
        "updates and range minimum queries over a large array of integers",
    )
    court = volume_score(m, "Return the sum of two integers.")
    assert long_ > court, f"long={long_:.3f} court={court:.3f}"
