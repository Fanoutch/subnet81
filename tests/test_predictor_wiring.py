"""Câblage du prédicteur de difficulté dans le choix des prompts.

Sans prédicteur, le mineur tire au hasard dans la TRANCHE de la fenêtre (5000
prompts imposés par le validateur, jamais le dataset entier). Le câblage tire N
candidats par ce même chemin — donc tranche et cooldown restent respectés par
construction — puis garde le mieux noté. Garder le meilleur sur N revient à
sélectionner le top 1/N.

Taux de k=2 mesuré le 2026-08-06 sur 8813 groupes, 20 découpes 80/20, contre
3.88% [3.28-4.42] au hasard :
    N=5  (top 20%)  6.83% [5.46-7.92]   x1.76
    N=10 (top 10%)  7.65% [4.92-8.74]   x1.97
    N=20 (top  5%)  7.69% [4.40-10.99]  x1.98   <- retenu
Le gain plafonne à x2 dès le top 10%. N=50 (top 2%) promet x2.87 mais extrapole
sur la pointe d'un modèle encore faible — à réévaluer sur la v1.

⚠️ PROPRIÉTÉ DE SÛRETÉ : sans modèle fourni, le comportement doit être
strictement identique à l'ancien (tirage uniforme). C'est ce qui rend le
câblage réversible par simple variable d'environnement.
"""
from __future__ import annotations

import random

import pytest

from reliquary.miner.engine import pick_prompt_idx


class _Env:
    """Env minimal : 100 prompts dont le texte encode l'indice."""

    def __init__(self, texts=None):
        self._texts = texts or {}
        self.fetched = []

    def __len__(self):
        return 100

    def get_problem(self, idx):
        self.fetched.append(idx)
        return {"prompt": self._texts.get(idx, f"prompt numero {idx}")}


def _model(scores_by_word):
    """Modèle jouet : chaque mot porte un prior, idf uniforme à 1."""
    return {
        "word_priors": dict(scores_by_word),
        "idf": {w: 1.0 for w in scores_by_word},
        "global_mean": 0.0,
    }


def test_without_predictor_behaviour_is_unchanged():
    """Propriété de sûreté : pas de modèle => tirage uniforme, comme avant."""
    env = _Env()
    a = pick_prompt_idx(
        env, set(), rng=random.Random(1), prompt_range=(0, 100),
    )
    b = pick_prompt_idx(
        env, set(), rng=random.Random(1), prompt_range=(0, 100),
    )
    assert a == b                       # déterministe pour une graine donnée
    assert env.fetched == [], "aucun texte ne doit être lu sans prédicteur"


def test_predictor_picks_the_highest_scoring_candidate():
    env = _Env(texts={i: ("facile" if i != 42 else "dur") for i in range(100)})
    # 'dur' vaut plus que 'facile' -> l'indice 42 doit gagner s'il est tiré.
    model = _model({"dur": 0.9, "facile": 0.1})

    # Avec assez de candidats, 42 finit par être dans le lot et doit sortir.
    got = pick_prompt_idx(
        env, set(), rng=random.Random(0), prompt_range=(40, 45),
        predictor=model, n_candidates=5,
    )
    assert got == 42


def test_predictor_reads_only_the_candidates_it_scores():
    """Le coût doit rester borné : N lectures, pas toute la tranche."""
    env = _Env()
    model = _model({"prompt": 0.5, "numero": 0.5})
    pick_prompt_idx(
        env, set(), rng=random.Random(0), prompt_range=(0, 100),
        predictor=model, n_candidates=4,
    )
    assert len(env.fetched) <= 4


def test_cooldown_is_still_honoured_under_the_predictor():
    """Le prédicteur ne doit jamais ressortir un prompt en cooldown."""
    env = _Env()
    model = _model({"prompt": 0.5})
    cooldown = set(range(0, 99))        # seul 99 reste libre
    got = pick_prompt_idx(
        env, cooldown, rng=random.Random(0), prompt_range=(0, 100),
        predictor=model, n_candidates=5,
    )
    assert got == 99


def test_prompt_range_is_still_honoured_under_the_predictor():
    """La tranche de fenêtre est une contrainte dure : hors tranche = rejet."""
    env = _Env()
    model = _model({"prompt": 0.5})
    for seed in range(20):
        got = pick_prompt_idx(
            env, set(), rng=random.Random(seed), prompt_range=(30, 40),
            predictor=model, n_candidates=5,
        )
        assert 30 <= got < 40


def test_a_failing_prompt_fetch_never_breaks_the_pick():
    """Un env qui lève sur get_problem ne doit pas coûter un bake."""

    class _Broken(_Env):
        def get_problem(self, idx):
            raise RuntimeError("parquet indisponible")

    got = pick_prompt_idx(
        _Broken(), set(), rng=random.Random(0), prompt_range=(0, 100),
        predictor=_model({"prompt": 0.5}), n_candidates=5,
    )
    assert 0 <= got < 100


def test_candidate_pool_is_sized_on_the_measured_plateau():
    """N=20 => top 5%. Le gain plafonne à x2 dès le top 10% (mesuré sur 8813
    groupes, 20 découpes) ; 5 candidats (top 20%) ne donnaient que x1.76.
    """
    from reliquary.miner.engine import PREDICTOR_CANDIDATES

    assert PREDICTOR_CANDIDATES >= 10, "sous le palier de gain mesuré"
    # ... sans extrapoler sur la pointe d'un modèle encore faible (N=50+).
    assert PREDICTOR_CANDIDATES <= 25, "extrapole hors du domaine mesuré"
