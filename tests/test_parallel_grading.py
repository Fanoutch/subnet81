"""La correction d'un groupe doit se faire en PARALLÈLE, pas en série.

Mesuré 2026-07-23 (H200, code-only) : le mineur passe 52% de son temps sur
`reward` avec le GPU à 0%. En code, `compute_reward` lance un `subprocess.run`
isolé PAR ROLLOUT (les cas de test dans un sandbox Python), exécutés en série.
Pendant ce temps le GPU dort.

Parallélisé sur threads (subprocess.run libère le GIL), mesuré ×9,6 :
128 corrections (16 prompts × 8) = 4,52 s série → 0,47 s sur 16 threads.

⚠️ Sûreté : la correction ne touche NI aux tokens NI aux preuves — seulement
un score par rollout. Contrairement à tout ce qui touche au forced-seed, aucun
risque de parité. Le SEUL invariant à préserver : l'ordre des scores doit
correspondre à l'ordre des rollouts (score[i] = reward du rollout i).
"""

from __future__ import annotations

import time

import pytest


class _SlowEnv:
    """Simule compute_reward lent (subprocess) et vérifie l'ordre."""

    name = "opencodeinstruct"

    def __init__(self, delay=0.05):
        self.delay = delay
        self.seen = []

    def compute_reward(self, problem, completion):
        time.sleep(self.delay)          # simule le subprocess
        # le completion encode l'indice du rollout pour vérifier l'ordre
        return float(completion)


def _grade(env, completions, workers):
    from reliquary.miner.engine import grade_group_parallel

    return grade_group_parallel(env, [({}, c) for c in completions],
                                max_workers=workers)


def test_scores_are_in_rollout_order():
    """score[i] doit être le reward du rollout i, malgré le parallélisme."""
    env = _SlowEnv(delay=0.0)
    completions = [str(float(i)) for i in range(8)]   # "0.0".."7.0"
    scores = _grade(env, completions, workers=8)
    assert scores == [float(i) for i in range(8)]


def test_parallel_is_faster_than_serial():
    """8 corrections de 50 ms : ~0,4 s en série, ~0,05 s en parallèle."""
    env = _SlowEnv(delay=0.05)
    completions = [str(float(i)) for i in range(8)]
    t = time.perf_counter()
    _grade(env, completions, workers=8)
    parallel = time.perf_counter() - t
    # en série ce serait ~0,4 s ; en parallèle bien moins
    assert parallel < 0.2, f"pas parallélisé: {parallel:.2f}s"


def test_single_rollout():
    env = _SlowEnv(delay=0.0)
    assert _grade(env, ["3.0"], workers=8) == [3.0]


def test_a_failing_reward_does_not_break_the_group():
    """Une correction qui lève doit donner 0.0, pas planter tout le groupe
    (le grader code renvoie déjà 0.0 sur crash, mais on protège le wrapper)."""
    class _Boom(_SlowEnv):
        def compute_reward(self, problem, completion):
            if completion == "2.0":
                raise RuntimeError("grader crash")
            return float(completion)

    scores = _grade(_Boom(), [str(float(i)) for i in range(4)], workers=4)
    assert scores == [0.0, 1.0, 0.0, 3.0]
