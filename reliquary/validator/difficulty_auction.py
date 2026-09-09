"""Port ciblé de l'enchère de difficulté — fonctions BYTE-EXACTES upstream.

Extrait de ``reliquary/validator/difficulty_auction.py`` @ a6456b4 (PR #178
« v4 release blockers ») : uniquement les fonctions pures consommées par le
miroir mineur du gate « uncertain » (rollout tronqué ou non-boxé = outcome
incertain, groupe admis/valorisé au MIN de l'utilité gatée sur toutes les
réinterprétations du lattice). Les dataclasses shadow-auction upstream (qui
importent batch_selection) ne sont PAS portées. Ne pas éditer la logique.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from math import comb
from typing import Iterable


@dataclass(frozen=True)
class DifficultyScore:
    value: float
    mean_reward: float
    reward_std: float
    reward_count: int


def difficulty_score(
    rewards: Iterable[float],
    *,
    delta: float = 1.0,
) -> DifficultyScore:
    """Return ``std(rewards) * (1 - mean(rewards)) ** delta``.

    Validator rewards are required to be finite and inside ``[0, 1]``. A bad
    reward domain is a programming/configuration error, not a zero-value group:
    silently ranking it last would hide an invalid counterfactual.
    """
    values = tuple(float(reward) for reward in rewards)
    if not math.isfinite(delta) or delta < 0.0:
        raise ValueError("difficulty delta must be finite and non-negative")
    if any(not math.isfinite(reward) for reward in values):
        raise ValueError("difficulty rewards must be finite")
    if any(reward < 0.0 or reward > 1.0 for reward in values):
        raise ValueError("difficulty rewards must be in [0, 1]")

    count = len(values)
    if count == 0:
        return DifficultyScore(0.0, 0.0, 0.0, 0)

    mean_reward = sum(values) / count
    variance = sum(
        (reward - mean_reward) ** 2 for reward in values
    ) / count
    reward_std = variance**0.5
    value = reward_std * (1.0 - mean_reward) ** delta
    return DifficultyScore(value, mean_reward, reward_std, count)


def gated_difficulty_utility(
    rewards: Iterable[float],
    *,
    sigma_min: float,
    delta: float = 1.0,
) -> float:
    """Return auction difficulty only when the reward vector passes its gate.

    This mirrors the validator's population-sigma eligibility boundary without
    importing environment or bootstrap constants. Callers must supply the
    threshold that applies to the candidate being valued.
    """
    if not math.isfinite(sigma_min) or sigma_min < 0.0:
        raise ValueError("sigma minimum must be finite and non-negative")

    score = difficulty_score(rewards, delta=delta)
    if score.reward_std < 1e-8 or score.reward_std < sigma_min:
        return 0.0
    return score.value


def fractional_reward_lattice(total_tests: int) -> tuple[float, ...]:
    """Return every attainable ``passed / total_tests`` reward.

    ``total_tests=1`` is the Math lattice ``{0, 1}``; larger denominators model
    the fractional rewards emitted by the Code grader.
    """
    if (
        isinstance(total_tests, bool)
        or not isinstance(total_tests, int)
        or total_tests <= 0
    ):
        raise ValueError("total tests must be a positive integer")
    return tuple(passed / total_tests for passed in range(total_tests + 1))


ROBUST_UTILITY_MAX_OUTCOMES = 200_000


def robust_uncertain_reward_utility(
    rewards: Iterable[float],
    *,
    sigma_min: float,
    uncertain_indices: Iterable[int] = (),
    attainable_rewards: Iterable[float] = (),
    delta: float = 1.0,
) -> float:
    """Return the least utility across all outcomes of uncertain rollouts.

    A truncated or off-format rollout has an untrusted observed reward.
    Replacing it with every value in its exact environment-specific lattice and
    minimizing the *gated* utility prevents the uncertainty from improving
    either sigma eligibility or auction difficulty. Several rollouts may be
    unknown; every joint assignment is priced, so the guarantee does not weaken
    as the uncertain set grows.
    """
    values = tuple(float(reward) for reward in rewards)
    indices = tuple(dict.fromkeys(uncertain_indices))
    if not indices:
        return gated_difficulty_utility(
            values,
            sigma_min=sigma_min,
            delta=delta,
        )
    for index in indices:
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= len(values)
        ):
            raise ValueError("uncertain index must identify one reward")

    lattice = tuple(dict.fromkeys(float(reward) for reward in attainable_rewards))
    if not lattice:
        raise ValueError("attainable rewards must not be empty")
    if any(not math.isfinite(reward) for reward in lattice):
        raise ValueError("attainable rewards must be finite")
    if any(reward < 0.0 or reward > 1.0 for reward in lattice):
        raise ValueError("attainable rewards must be in [0, 1]")

    # Validate the observed vector too, even though the unknown position will be
    # replaced below. A malformed validator reward should never be hidden.
    difficulty_score(values, delta=delta)

    # The score is symmetric in the rewards, so distinct OUTCOMES are multisets
    # of assignments, not ordered tuples: enumerate combinations with
    # replacement rather than the full product.
    from itertools import combinations_with_replacement

    outcome_count = comb(len(lattice) + len(indices) - 1, len(indices))
    if outcome_count > ROBUST_UTILITY_MAX_OUTCOMES:
        # Refuse to guess. Returning 0 makes the candidate ineligible, which is
        # the SAFE direction: a miner can never widen its own lattice to buy a
        # pass, only to be rejected.
        return 0.0

    utilities: list[float] = []
    for assignment in combinations_with_replacement(lattice, len(indices)):
        outcome = list(values)
        for index, attainable_reward in zip(indices, assignment):
            outcome[index] = attainable_reward
        utilities.append(
            gated_difficulty_utility(
                outcome,
                sigma_min=sigma_min,
                delta=delta,
            )
        )
    return min(utilities)
