"""Le processeur HF forced-seed doit produire le MÊME masque, sans boucle Python.

Aujourd'hui `ForcedSeedLogitsProcessor.__call__` boucle `for r in range(n)` sur
les ~200 lignes du batch — mesuré 65,6 ms/pas sur H200, contre 4,5 ms vectorisé
(×14,5, forced_sampling.force_rows_batched). Sur une phase-1 de 512 tokens :
33,6 s → 2,3 s.

Le contrat de sortie ne change pas : une matrice [n, vocab] où chaque ligne vaut
-inf partout SAUF le token forcé, à 0.0 — ce que le sampler greedy lit ensuite.
On teste l'égalité EXACTE avec la version boucle (parité forced-seed).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from reliquary.miner.forced_seed_sampler import ForcedSeedLogitsProcessor

RAND = "ab" * 32
CKPT = "ckpt-hash"
VOCAB = 4096


def _proc(n, start_len=4):
    return ForcedSeedLogitsProcessor(
        randomness=RAND, hotkey="unused-v2",
        prompt_idx=[100 + r for r in range(n)],
        checkpoint_hash=CKPT,
        rollout_indices=list(range(n)),
        base_offsets=[0] * n, start_len=start_len,
    )


def _reference_call(proc, input_ids, scores):
    """La version boucle historique, recopiée pour servir d'oracle."""
    from reliquary.environment.forced_sampling import pick, warp, u_at

    s = int(input_ids.shape[1]) - proc.start_len
    out = torch.full_like(scores, float("-inf"))
    for r in range(scores.shape[0]):
        t = proc.base_offsets[r] + s
        u = u_at(proc.randomness, proc.prompt_indices[r],
                 proc.checkpoint_hash, proc.rollout_indices[r], t)
        probs = warp(scores[r], t=proc.temperature,
                     top_k=proc.top_k, top_p=proc.top_p)
        out[r, pick(probs, u)] = 0.0
    return out


@pytest.mark.parametrize("seed", range(4))
def test_vectorized_call_matches_the_loop_exactly(seed):
    torch.manual_seed(seed)
    n = 200
    proc = _proc(n)
    scores = torch.randn(n, VOCAB)
    input_ids = torch.zeros(n, proc.start_len + 7, dtype=torch.long)  # s=7

    got = proc(input_ids, scores)
    ref = _reference_call(proc, input_ids, scores)
    assert torch.equal(got, ref), "le masque diverge de la boucle -> SEED_MISMATCH"


def test_each_row_has_exactly_one_forced_token():
    torch.manual_seed(0)
    proc = _proc(5)
    scores = torch.randn(5, VOCAB)
    input_ids = torch.zeros(5, proc.start_len + 3, dtype=torch.long)
    out = proc(input_ids, scores)
    for r in range(5):
        assert int((out[r] == 0.0).sum()) == 1
        assert int(torch.isinf(out[r]).sum()) == VOCAB - 1


def test_position_advances_with_generated_length():
    """t = base_offset + (input_ids.shape[1] - start_len) : deux longueurs
    differentes doivent forcer des tokens (potentiellement) differents."""
    torch.manual_seed(0)
    proc = _proc(3)
    scores = torch.randn(3, VOCAB)
    a = proc(torch.zeros(3, proc.start_len + 1, dtype=torch.long), scores)
    b = proc(torch.zeros(3, proc.start_len + 50, dtype=torch.long), scores)
    # au moins une ligne change de token forcé entre t=1 et t=50
    ta = [int((a[r] == 0.0).nonzero()[0]) for r in range(3)]
    tb = [int((b[r] == 0.0).nonzero()[0]) for r in range(3)]
    assert ta != tb
