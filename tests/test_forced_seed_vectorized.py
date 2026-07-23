"""Version vectorisée du forçage forced-seed — MÊME token, mais tout le batch
en une passe au lieu d'une boucle Python ligne par ligne.

Mesuré le 2026-07-23 : le processeur forced-seed coûte 12,5× (2 081 tok/s avec,
26 112 sans, H200). Cause double, dans `ForcedSeedLogitsProcessor.__call__` :
  1. `for r in range(scores.shape[0])` — boucle Python sur les ~200 séquences
  2. dans chaque tour, `warp()` fait topk + sort + cumsum + scatter sur les
     151 000 entrées du vocabulaire alors que seules `top_k`=20 sont non nulles.

`force_rows_batched` fait les deux d'un coup : une seule opération tensorielle
sur toutes les lignes, et le travail post-topk restreint aux 20 valeurs utiles.

⚠️ CRITÈRE ABSOLU : le token choisi pour chaque ligne doit être IDENTIQUE à la
boucle `pick(warp(row), u)` actuelle. `forced_sampling.py` est partagé avec le
validateur ; un seul token différent = SEED_MISMATCH. On teste donc l'égalité
EXACTE des token-ids, pas une tolérance.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from reliquary.environment.forced_sampling import pick, warp

T, TOP_K, TOP_P = 0.6, 20, 0.95
VOCAB = 4096


def _reference_row(logits_row, u):
    """Ce que fait la production aujourd'hui, une ligne à la fois."""
    return pick(warp(logits_row, t=T, top_k=TOP_K, top_p=TOP_P), u)


def _run_batched(logits, us):
    from reliquary.environment.forced_sampling import force_rows_batched

    return force_rows_batched(logits, us, t=T, top_k=TOP_K, top_p=TOP_P)


@pytest.mark.parametrize("seed", range(6))
def test_batched_matches_reference_token_for_token(seed):
    """Sur des logits aléatoires, chaque ligne doit donner le MÊME token-id."""
    torch.manual_seed(seed)
    n = 200
    logits = torch.randn(n, VOCAB)
    # u variés, dont des valeurs près des bords (0 et 1) où le pick est sensible
    us = torch.rand(n).tolist()
    us[0], us[1], us[2] = 0.0, 0.999999, 0.5

    got = _run_batched(logits, us)
    ref = [_reference_row(logits[r], us[r]) for r in range(n)]
    assert got.tolist() == ref, "un token diverge de la reference -> SEED_MISMATCH"


def test_single_row_batch():
    torch.manual_seed(0)
    logits = torch.randn(1, VOCAB)
    got = _run_batched(logits, [0.3])
    assert got.tolist() == [_reference_row(logits[0], 0.3)]


def test_u_exactly_zero_takes_the_first_cdf_token():
    """u=0 est un bord classique : le pick doit rester deterministe."""
    torch.manual_seed(1)
    logits = torch.randn(4, VOCAB)
    got = _run_batched(logits, [0.0, 0.0, 0.0, 0.0])
    ref = [_reference_row(logits[r], 0.0) for r in range(4)]
    assert got.tolist() == ref


def test_peaked_distribution_all_u_same_token():
    """Un logit tres pique : toutes les valeurs de u tombent sur le meme token,
    la version batchee doit le retrouver comme la reference."""
    logits = torch.full((3, VOCAB), -10.0)
    logits[:, 42] = 20.0            # token 42 domine ecrasamment
    got = _run_batched(logits, [0.01, 0.5, 0.99])
    ref = [_reference_row(logits[r], u) for r, u in enumerate([0.01, 0.5, 0.99])]
    assert got.tolist() == ref == [42, 42, 42]
