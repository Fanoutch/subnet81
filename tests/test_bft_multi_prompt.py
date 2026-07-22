"""Phase-2 BFT batchée ENTRE prompts (et non plus un prompt à la fois).

Mesuré le 2026-07-21 : la phase-1 est batchée (tous les prompts en un appel
vLLM) mais la phase-2 repasse prompt par prompt (~4 s chacun). Un groupe
payable en 10e position n'est donc noté qu'à ~87 s, et une fois sa preuve
calculée (27 s) il dépasse la fenêtre de collecte de 100 s. Les groupes
payables étant rares (~0,2%), en perdre un pour une question de RANG est cher.

Batcher la phase-2 rend l'instant de notation identique pour tous les prompts :
le payable part quelle que soit sa position.

⚠️ QUATRE pièges, chacun invisible en local et fatal côté validateur
(SEED_MISMATCH / rollout prouvé pour le mauvais flux) :
  1. ``plen`` doit être celui de CHAQUE prompt (longueurs différentes)
  2. ``tokens`` doit repartir du préfixe de SON prompt
  3. ``base_offsets`` se calcule avec le plen de SA ligne
  4. dans le chemin mono-prompt, l'indice de LIGNE sert d'indice de ROLLOUT —
     faux dès qu'on mélange les prompts (ligne 12 = rollout 4 du prompt 2)
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

EOS = 99
THINK_CLOSE = 7
FORCE = [5, 6]


class _FakeModel:
    """Capture les arguments du processeur forced-seed et rend des tokens
    déterministes, pour vérifier le CONTRAT sans GPU."""

    device = "cpu"

    def __init__(self):
        self.calls = []

    def generate(self, input_ids, attention_mask=None, max_new_tokens=8, **kw):
        procs = kw.get("logits_processor") or []
        self.calls.append({
            "width": int(input_ids.shape[1]),
            "rows": int(input_ids.shape[0]),
            "procs": list(procs),
        })
        # continue chaque ligne par [EOS] pour terminer immédiatement
        tail = torch.full((input_ids.shape[0], 1), EOS, dtype=torch.long)
        return torch.cat([input_ids, tail], dim=1)


def _groups():
    """2 prompts de LONGUEURS DIFFÉRENTES, 2 rollouts chacun.
    Rollout 0 de chaque prompt finit sur EOS (donc pas de phase-2) ;
    rollout 1 n'a pas fermé le thinking (donc FORCE + phase-2)."""
    pA, pB = [1, 2, 3], [4, 5]          # len 3 et len 2
    return [
        {"prompt_tokens": pA, "prompt_idx": 111,
         "completions": [pA + [10, EOS], pA + [11, 12]]},
        {"prompt_tokens": pB, "prompt_idx": 222,
         "completions": [pB + [20, EOS], pB + [21, 22]]},
    ]


def _run(model):
    from reliquary.miner.bft import bft_rollouts_from_completions_multi

    return bft_rollouts_from_completions_multi(
        _groups(), model=model, think_close_ids={THINK_CLOSE},
        force_ids=FORCE, eos_ids={EOS}, answer_budget=8,
        randomness="ab" * 32, hotkey="unused-v2", checkpoint_hash="ckpt",
    )


def test_resultats_regroupes_par_prompt_dans_l_ordre_d_entree():
    out = _run(_FakeModel())
    assert len(out) == 2
    assert all(len(g) == 2 for g in out)


def test_chaque_rollout_porte_le_plen_de_SON_prompt():
    """Piège 1 : un plen commun décale la frontière prompt/complétion."""
    out = _run(_FakeModel())
    assert [r["prompt_length"] for r in out[0]] == [3, 3]
    assert [r["prompt_length"] for r in out[1]] == [2, 2]


def test_chaque_rollout_repart_du_prefixe_de_SON_prompt():
    """Piège 2 : les tokens doivent commencer par le bon prompt."""
    out = _run(_FakeModel())
    for r in out[0]:
        assert r["tokens"][:3] == [1, 2, 3]
    for r in out[1]:
        assert r["tokens"][:2] == [4, 5]


def test_le_processeur_recoit_le_prompt_idx_de_chaque_ligne():
    """Piège 3 : une ligne sur le mauvais prompt_idx = flux forcé faux."""
    m = _FakeModel()
    _run(m)
    assert len(m.calls) == 1, "la phase-2 doit être UN seul appel batché"
    proc = m.calls[0]["procs"][-1]
    # seuls les rollouts 1 de chaque prompt vont en phase-2
    assert proc.prompt_indices == [111, 222]


def test_le_processeur_recoit_l_indice_de_ROLLOUT_pas_de_LIGNE():
    """Piège 4 : le plus insidieux. Ligne globale 1 = rollout 1 du prompt A ;
    ligne globale 3 = rollout 1 du prompt B — pas rollout 3."""
    m = _FakeModel()
    _run(m)
    proc = m.calls[0]["procs"][-1]
    assert proc.rollout_indices == [1, 1], (
        f"indices de rollout {proc.rollout_indices} : l'indice de ligne globale "
        f"a ete utilise au lieu de la position dans le groupe"
    )


def test_les_lignes_terminees_sur_eos_ne_passent_pas_en_phase_2():
    """Comportement identique au chemin mono-prompt."""
    m = _FakeModel()
    out = _run(m)
    assert m.calls[0]["rows"] == 2, "seuls les rollouts non terminés doivent générer"
    assert out[0][0]["forced"] is False
    assert out[1][0]["forced"] is False


def test_un_groupe_vide_est_rejete():
    from reliquary.miner.bft import bft_rollouts_from_completions_multi

    with pytest.raises(ValueError):
        bft_rollouts_from_completions_multi(
            [], model=_FakeModel(), think_close_ids={THINK_CLOSE},
            force_ids=FORCE, eos_ids={EOS}, answer_budget=8,
            randomness="ab" * 32, hotkey="u", checkpoint_hash="c",
        )
