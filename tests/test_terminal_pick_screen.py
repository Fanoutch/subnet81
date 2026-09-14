"""Garde locale : l'EOS final doit être le pick forced-seed EXACT (V1, #253).

Validateur 1f1cc16 (live 12/09 23:23) : sous v6 ``verify_termination`` exige
``terminal_pick_ok`` — le dernier token, s'il est un stop, doit être
``pick(warp(logits[t-1]), u_at(..., rollout, j))`` recalculé sur SON forward
HF. Nos tokens sortent de vLLM ; un EOS que le forward HF n'aurait pas tiré
donne ``bad_termination``, et le seuil de dette est de 1 échec par hotkey :
toute la fenêtre est perdue (mesuré fen 45883 : 1 bad_termination à +15,6 s,
5 groupes admis sautés ``proof_failure_debt``).
"""
from __future__ import annotations

import math

import pytest
import torch

from reliquary.miner import engine


@pytest.fixture(autouse=True)
def _sampling_v6(monkeypatch):
    # contrat V1 : T=1,0, top_p=1,0, top_k=0 (la suite tourne sous les
    # constantes par défaut du dépôt, pas celles du profil live).
    monkeypatch.setattr(engine, "T_PROTO", 1.0)
    monkeypatch.setattr(engine, "TOP_K_PROTO", 0)
    monkeypatch.setattr(engine, "TOP_P_PROTO", 1.0)

EOS = 7


def _logits_with(p_eos: float, vocab: int = 10) -> torch.Tensor:
    """Logits dont le softmax donne p_eos à EOS et le reste réparti."""
    rest = (1.0 - p_eos) / (vocab - 1)
    probs = torch.full((vocab,), rest)
    probs[EOS] = p_eos
    return torch.log(probs)


def _eos_interval(p_eos: float, vocab: int = 10):
    rest = (1.0 - p_eos) / (vocab - 1)
    lower = rest * EOS
    return lower, lower + p_eos


# --------------------------------------------------------- position terminale
def test_inputs_rollout_termine_sur_eos():
    tokens = [1, 2, 3, 4, EOS]
    assert engine.terminal_pick_inputs(tokens, prompt_length=2, eos_set={EOS}) == (4, 2)


def test_inputs_rollout_sans_eos_final_ne_se_juge_pas():
    assert engine.terminal_pick_inputs([1, 2, 3, 4], 2, {EOS}) is None


def test_inputs_completion_vide():
    assert engine.terminal_pick_inputs([1, EOS], 2, {EOS}) is None


# ------------------------------------------------------------- verdict du pick
def test_pick_exact_au_centre_de_lintervalle():
    lo, hi = _eos_interval(0.6)
    ok, edge = engine.terminal_pick_verdict(_logits_with(0.6), EOS, (lo + hi) / 2,
                                           margin=1e-3)
    assert ok is True
    assert edge > 0.2


def test_pick_hors_intervalle_echoue():
    lo, _ = _eos_interval(0.6)
    ok, edge = engine.terminal_pick_verdict(_logits_with(0.6), EOS, lo / 2, margin=0.0)
    assert ok is False
    assert edge < 0


def test_pick_exact_mais_trop_pres_du_bord_echoue_avec_marge():
    lo, _ = _eos_interval(0.6)
    u = lo + 1e-5
    ok, edge = engine.terminal_pick_verdict(_logits_with(0.6), EOS, u, margin=1e-3)
    assert ok is False
    assert 0 <= edge < 1e-3
    ok0, _ = engine.terminal_pick_verdict(_logits_with(0.6), EOS, u, margin=0.0)
    assert ok0 is True


def test_logits_bf16_sont_upcastes_comme_le_validateur():
    lo, hi = _eos_interval(0.6)
    lg = _logits_with(0.6).to(torch.bfloat16)
    ok, _ = engine.terminal_pick_verdict(lg, EOS, (lo + hi) / 2, margin=1e-3)
    assert ok is True


def test_marge_lue_dans_lenvironnement(monkeypatch):
    monkeypatch.delenv("RELIQUARY_TERMINAL_PICK_MARGIN", raising=False)
    assert math.isclose(engine.terminal_pick_margin(), 5e-4)
    monkeypatch.setenv("RELIQUARY_TERMINAL_PICK_MARGIN", "0.002")
    assert math.isclose(engine.terminal_pick_margin(), 0.002)
    monkeypatch.setenv("RELIQUARY_TERMINAL_PICK_MARGIN", "nope")
    assert math.isclose(engine.terminal_pick_margin(), 5e-4)


# ------------------------------------------------------ branchement de la preuve
def _proof_engine(monkeypatch, logits_for):
    """Moteur partiel dont le forward renvoie des logits choisis par position."""
    import contextlib
    from types import SimpleNamespace

    import reliquary.shared.forward as fwd
    from reliquary.environment import code_grader

    eng = engine.MiningEngine.__new__(engine.MiningEngine)
    eng.hf_model = SimpleNamespace()          # pas de lm_head → chemin legacy
    eng.tokenizer = SimpleNamespace(decode=lambda toks: "x")
    eng._eos_ids = [EOS]
    eng._cached_randomness = "ab" * 32
    eng._local_hash = "c" * 40

    def fake_forward(model, inp, mask, layer, materialize_logits=True):
        toks = inp[0].tolist()
        n = len(toks)
        logits = torch.stack([logits_for(i, toks) for i in range(n)])[None]
        return torch.zeros(1, n, 4), logits

    monkeypatch.setattr(fwd, "forward_single_layer", fake_forward)
    monkeypatch.setattr(code_grader, "fork_gpu_guard", contextlib.nullcontext)
    monkeypatch.setenv("RELIQUARY_TERMINAL_PICK_MARGIN", "0")
    return eng


def test_preuve_marque_le_groupe_si_leos_final_nest_pas_le_pick(monkeypatch):
    from reliquary.environment.forced_sampling import u_at

    prompt_idx, prompt_len = 42, 2
    tokens = [1, 2, 3, EOS]                   # j = 1 pour l'EOS final
    eng = _proof_engine(monkeypatch, lambda i, toks: torch.zeros(10))
    u = u_at(eng._cached_randomness, prompt_idx, eng._local_hash, 0, 1)
    # logits uniformes : EOS=7 couvre [0,7 ; 0,8) — pick exact ssi u y tombe
    expected_ok = 0.7 <= u < 0.8
    out = eng._proof_rollouts(
        [{"tokens": tokens, "prompt_length": prompt_len}], texts=["x"],
        device="cpu", terminal_ctx=eng._terminal_pick_ctx(prompt_idx),
    )
    assert (out[0]["local_screen"] is None) is expected_ok
    if not expected_ok:
        assert out[0]["local_screen"] == "local_terminal_pick"


def test_preuve_garde_le_groupe_quand_leos_est_quasi_certain(monkeypatch):
    def lg(i, toks):
        # chaque position prédit avec certitude le token suivant réel : aucun
        # autre filtre local ne mord, seul le pick terminal est en jeu.
        row = torch.full((10,), -30.0)
        row[toks[i + 1] if i + 1 < len(toks) else EOS] = 30.0
        return row

    eng = _proof_engine(monkeypatch, lg)
    gens = [{"tokens": [1, 2, 3, EOS], "prompt_length": 2} for _ in range(3)]
    out = eng._proof_rollouts(gens, texts=["x"] * 3, device="cpu",
                              terminal_ctx=eng._terminal_pick_ctx(7))
    assert all(r["local_screen"] is None for r in out)


def test_garde_coupee_par_variable(monkeypatch):
    eng = _proof_engine(monkeypatch, lambda i, toks: torch.zeros(10))
    monkeypatch.setenv("RELIQUARY_TERMINAL_PICK_SCREEN", "0")
    assert eng._terminal_pick_ctx(1) is None
