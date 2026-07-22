"""Le processeur forced-seed doit accepter un prompt_idx PAR LIGNE.

Pourquoi : la phase-2 (BFT answer) est déjà batchée sur les 8 rollouts d'un
prompt, mais PAS entre prompts, parce que `prompt_idx` est scalaire alors que
`rollout_indices` et `base_offsets` sont déjà des listes. Résultat mesuré le
2026-07-21 : après une phase-1 batchée (tous les prompts d'un coup), la phase-2
repasse prompt par prompt (~4 s chacun) — un prompt payable en 10e position
n'est examiné qu'à ~87 s et, une fois sa preuve calculée, dépasse la fenêtre de
100 s. L'effet de POSITION fait perdre des groupes payables, qui sont rares.

Rendre prompt_idx per-row permet de batcher la phase-2 entre prompts : tous les
groupes sont alors notés au même instant, et le payable part quelle que soit sa
position.

⚠️ Sécurité : `u_at(randomness, prompt_idx, ckpt, rollout, t)` détermine CHAQUE
token. Une ligne qui reçoit le prompt_idx d'une autre est forcée sur le mauvais
flux → SEED_MISMATCH côté validateur, sans rien d'anormal en local.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from reliquary.environment.forced_sampling import pick, u_at, warp
from reliquary.miner.forced_seed_sampler import ForcedSeedLogitsProcessor

RANDOMNESS = "ab" * 32
CKPT = "ckpt-hash"
VOCAB = 32


def _scores(n_rows: int) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(n_rows, VOCAB)


def _proc(prompt_idx, rollouts, offsets, start_len=4):
    return ForcedSeedLogitsProcessor(
        randomness=RANDOMNESS, hotkey="unused-v2", prompt_idx=prompt_idx,
        checkpoint_hash=CKPT, rollout_indices=rollouts,
        base_offsets=offsets, start_len=start_len,
    )


def test_scalar_prompt_idx_still_works_unchanged():
    """Compatibilité : tout le code existant passe un entier."""
    scores = _scores(2)
    ids = torch.zeros(2, 4, dtype=torch.long)
    out = _proc(42, [0, 1], [0, 0])(ids, scores)
    assert out.shape == scores.shape
    # chaque ligne force exactement un token (0.0), le reste -inf
    for r in range(2):
        assert int((out[r] == 0.0).sum()) == 1


def test_per_row_prompt_idx_forces_each_row_on_its_own_stream():
    """Le coeur du fix : la ligne r doit suivre prompt_indices[r]."""
    scores = _scores(3)
    ids = torch.zeros(3, 4, dtype=torch.long)
    idxs = [11, 22, 33]
    out = _proc(idxs, [0, 0, 0], [0, 0, 0])(ids, scores)

    for r, pidx in enumerate(idxs):
        # recalcul indépendant du pick attendu pour CE prompt_idx
        u = u_at(RANDOMNESS, pidx, CKPT, 0, 0)
        probs = warp(scores[r], t=0.6, top_k=20, top_p=0.95)
        assert int(out[r].argmax()) == pick(probs, u), f"ligne {r} sur le mauvais flux"


def test_per_row_differs_from_broadcasting_a_single_index():
    """Garde-fou : si le fix diffusait un seul index, ce test passerait à tort.

    Scores PLATS volontairement : la distribution warpee est alors uniforme sur
    top_k, donc deux u differents donnent presque surement des tokens
    differents. Avec des scores aleatoires la distribution est concentree et
    deux u distincts retombent souvent sur le meme token — le test ne
    discriminerait rien."""
    flat = torch.zeros(2, VOCAB)
    ids = torch.zeros(2, 4, dtype=torch.long)
    # cherche une paire d'index qui donne des picks distincts sur la ligne 1
    base = _proc([11, 11], [0, 0], [0, 0])(ids, flat)[1].argmax()
    for other in range(12, 60):
        got = _proc([11, other], [0, 0], [0, 0])(ids, flat)[1].argmax()
        if int(got) != int(base):
            return  # per-row bien applique
    pytest.fail("aucun prompt_idx ne change le pick de la ligne 1 — "
                "le per-row n'est pas applique")


def test_length_mismatch_is_rejected():
    """Un zip silencieux mélangerait les flux."""
    with pytest.raises(ValueError):
        _proc([11, 22], [0, 0, 0], [0, 0, 0])


def test_rollout_and_prompt_indices_combine_independently():
    """Batch multi-prompts ET multi-rollouts : chaque paire est distincte."""
    scores = _scores(4)
    ids = torch.zeros(4, 4, dtype=torch.long)
    pidx = [11, 11, 22, 22]
    roll = [0, 1, 0, 1]
    out = _proc(pidx, roll, [0, 0, 0, 0])(ids, scores)
    for r in range(4):
        u = u_at(RANDOMNESS, pidx[r], CKPT, roll[r], 0)
        probs = warp(scores[r], t=0.6, top_k=20, top_p=0.95)
        assert int(out[r].argmax()) == pick(probs, u)
