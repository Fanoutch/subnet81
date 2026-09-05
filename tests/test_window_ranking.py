"""Notation de la TRANCHE ENTIÈRE une fois par fenêtre, au lieu de N candidats.

Le mineur consomme ~74 prompts par fenêtre (16 par cycle, ~4.6 cycles). Les
noter tous d'avance permet de servir ces 74 depuis le haut du classement des
5000, soit le top 1.5% — mesuré x2.87 sur le taux de k=2 (2026-08-06, 8813
groupes, 20 découpes), contre x1.98 pour le meilleur-sur-20 actuel.

Coût mesuré sur la box : 2.29 ms la lecture d'un prompt, donc 11.4 s pour 5000,
UNE fois par fenêtre de 300 s (3.8%). Le classement doit donc être mis en cache
et réutilisé tant que la fenêtre ne change pas — le recalculer à chaque pioche
coûterait 16x plus cher que la génération elle-même.

⚠️ INVARIANTS : le cooldown peut GROSSIR pendant la fenêtre (nos propres
soumissions), donc il est appliqué au moment de la pioche, jamais figé dans le
classement. Et un classement périmé (fenêtre suivante) ne doit jamais servir :
ce serait piocher hors tranche, soit un rejet sec du validateur.
"""
from __future__ import annotations

import random

from reliquary.miner.engine import WindowRanking


class _Env:
    """Un mot distinct par prompt, pour que les scores soient tous différents.

    ⚠️ Répéter le MÊME mot ne marcherait pas : score_prompt fait une moyenne
    pondérée des priors, donc « dur dur dur » vaut exactement « dur ».
    """

    def __init__(self, n=200):
        self.n = n
        self.reads = 0

    def __len__(self):
        return self.n

    def get_problem(self, idx):
        self.reads += 1
        return {"prompt": f"w{idx}"}


def _model(n=200):
    """prior de 'w<i>' = i/n : le score croît strictement avec l'indice."""
    return {
        "word_priors": {f"w{i}": i / n for i in range(n)},
        "idf": {f"w{i}": 1.0 for i in range(n)},
        "global_mean": 0.0,
    }


def test_ranks_the_whole_slice_once_and_reuses_it():
    env, r = _Env(), WindowRanking()
    key = (100, "abc", "code")
    a = r.best(env, _model(), key, (0, 50), cooldown=set())
    reads_after_first = env.reads
    b = r.best(env, _model(), key, (0, 50), cooldown=set())
    assert env.reads == reads_after_first, "la 2e pioche a relu la tranche"
    assert a != b, "deux pioches doivent rendre deux prompts différents"


def test_serves_the_highest_scoring_prompts_first():
    env, r = _Env(), WindowRanking()
    key = (100, "abc", "code")
    got = [r.best(env, _model(), key, (0, 50), cooldown=set()) for _ in range(3)]
    assert got == [49, 48, 47], "doit servir le haut du classement, dans l'ordre"


def test_cooldown_is_applied_at_pick_time_not_frozen_in_the_ranking():
    """Le cooldown grossit pendant la fenêtre : le classement ne doit pas le figer."""
    env, r = _Env(), WindowRanking()
    key = (100, "abc", "code")
    assert r.best(env, _model(), key, (0, 50), cooldown=set()) == 49
    # 48 tombe en cooldown APRÈS la construction du classement
    assert r.best(env, _model(), key, (0, 50), cooldown={48}) == 47


def test_a_new_window_rebuilds_the_ranking():
    """Un classement périmé ferait piocher hors tranche = rejet du validateur."""
    env, r = _Env(), WindowRanking()
    r.best(env, _model(), (100, "abc", "code"), (0, 50), cooldown=set())
    reads = env.reads
    r.best(env, _model(), (101, "def", "code"), (60, 90), cooldown=set())
    assert env.reads > reads, "nouvelle fenêtre => nouveau classement"


def test_every_pick_stays_inside_the_slice():
    env, r = _Env(), WindowRanking()
    key = (100, "abc", "code")
    for _ in range(20):
        idx = r.best(env, _model(), key, (60, 90), cooldown=set())
        assert 60 <= idx < 90


def test_exhausted_ranking_returns_none_so_the_caller_can_fall_back():
    env, r = _Env(), WindowRanking()
    key = (100, "abc", "code")
    seen = set()
    for _ in range(5):
        got = r.best(env, _model(), key, (0, 5), cooldown=seen)
        if got is None:
            break
        seen.add(got)
    assert r.best(env, _model(), key, (0, 5), cooldown=seen) is None


def test_a_broken_env_never_breaks_the_ranking():
    class _Broken(_Env):
        def get_problem(self, idx):
            if idx % 2:
                raise RuntimeError("parquet indisponible")
            return {"prompt": "dur " * (idx + 1)}

    r = WindowRanking()
    got = r.best(_Broken(), _model(), (1, "a", "c"), (0, 20), cooldown=set())
    assert got is not None and got % 2 == 0


def test_cooldown_prompts_are_never_even_read():
    """Économie demandée : ne pas payer 2.29 ms pour un prompt inutilisable."""
    env, r = _Env(), WindowRanking()
    cooldown = set(range(0, 40))          # 40 des 50 sont déjà pris
    r.best(env, _model(), (1, "a", "c"), (0, 50), cooldown=cooldown)
    assert env.reads == 10, (
        f"a lu {env.reads} prompts au lieu des 10 réellement disponibles"
    )


def test_window_tally_reports_realised_k_distribution_per_window():
    """Bilan réalisé : publié au flip, confrontable à « prédiction tranche »."""
    from reliquary.miner.engine import WindowTally

    t = WindowTally()
    K2 = [1.0] * 2 + [0.0] * 6
    K8 = [1.0] * 8
    t.add(100, K2, 0)          # payable (intact + sigma 0.433)
    t.add(100, K8, 0)          # intact mais unanime
    t.add(100, K2, 3)          # k=2 d'apparence mais tronqué -> pas payable
    assert t._n == 3 and t._payable == 1 and t._intact == 2
    t.add(101, K2, 0)          # flip -> le bilan de 100 est publié et remis à zéro
    assert t._window == 101 and t._n == 1

    t.add(101, [0.5], 0)       # vecteur invalide : ignoré sans casser
    assert t._n == 1
