"""Balayage final G3/G4 : miroir mineur du gate v4 « uncertain » + malformed box.

Parité upstream a6456b4 (PR #178) : un rollout tronqué (cap sans EOS) ou
non-boxé (math, MATH_ANSWER_FORMAT=="boxed") est un outcome INCERTAIN — le
groupe n'est admis que si TOUTE réinterprétation sur le lattice reste en zone
(admission.py:843-895) ; une box finale mal formée à reward 0 rejette le
groupe ENTIER (MALFORMED_FINAL_ANSWER, admission.py:907-930).
"""
import importlib

import pytest

from tests.v4helpers import reload_constants

EOS = (151643, 151645)
CAP = 6
OK_TOKS = [1, 2, 3, EOS[0]]          # EOS final propre
TRUNC_TOKS = [1, 2, 3, 4, 5, 6]      # zéro EOS, longueur == cap


def _reload_engine(monkeypatch, version=None):
    reload_constants(monkeypatch, version)
    import reliquary.miner.engine as e

    importlib.reload(e)
    return e


@pytest.fixture(autouse=True)
def _restore(monkeypatch):
    yield
    _reload_engine(monkeypatch)


def _guard(e, rewards, comps, texts, env="openmathinstruct", tt=1):
    return e.v4_uncertain_guard(
        rewards, comps, texts, EOS, env, CAP, total_tests=tt,
    )


def test_malformed_final_box_drops_group(monkeypatch):
    e = _reload_engine(monkeypatch, 4)
    rewards = [1.0, 0.0]
    comps = [OK_TOKS, OK_TOKS]
    texts = ["$\\boxed{7}$", "je coupe en plein \\boxed{12"]
    assert _guard(e, rewards, comps, texts)[0] == "malformed_final_answer"


def test_k15_single_unboxed_zero_is_uncertain_out_of_zone(monkeypatch):
    # k=15/16 dont l'UNIQUE zéro est non-boxé : l'assignation 1 → k'=16, σ=0
    # → utilité 0 → min 0 → le validateur rejettera OUT_OF_ZONE.
    e = _reload_engine(monkeypatch, 4)
    rewards = [1.0] * 15 + [0.0]
    comps = [OK_TOKS] * 16
    texts = ["$\\boxed{7}$"] * 15 + ["pas de box ici"]
    assert _guard(e, rewards, comps, texts)[0] == "uncertain_out_of_zone"


def test_k1_single_correct_truncated_is_uncertain_out_of_zone(monkeypatch):
    # k=1/16 dont l'unique 1 est TRONQUÉ : assignation 0 → k'=0 → utilité 0.
    e = _reload_engine(monkeypatch, 4)
    rewards = [1.0] + [0.0] * 15
    comps = [TRUNC_TOKS] + [OK_TOKS] * 15
    texts = ["$\\boxed{7}$"] + ["$\\boxed{9}$"] * 15
    assert _guard(e, rewards, comps, texts)[0] == "uncertain_out_of_zone"


def test_partially_unboxed_still_in_zone_scores_at_min(monkeypatch):
    # k=8/16, un seul zéro non-boxé parmi 8 : toute réinterprétation reste en
    # zone → gardé, mais valorisé au MIN (< score observé).
    e = _reload_engine(monkeypatch, 4)
    rewards = [1.0] * 8 + [0.0] * 8
    comps = [OK_TOKS] * 16
    texts = ["$\\boxed{7}$"] * 8 + ["$\\boxed{9}$"] * 7 + ["pas de box"]
    reason, robust = _guard(e, rewards, comps, texts)
    assert reason is None
    assert robust is not None and 0.0 < robust < e.auction_score(rewards)


def test_no_uncertain_is_noop(monkeypatch):
    e = _reload_engine(monkeypatch, 4)
    rewards = [1.0] * 8 + [0.0] * 8
    comps = [OK_TOKS] * 16
    texts = ["$\\boxed{7}$"] * 16
    assert _guard(e, rewards, comps, texts) == (None, None)


def test_unboxed_detection_off_in_v3_format(monkeypatch):
    # v3 : MATH_ANSWER_FORMAT="boxed_or_trailing_number" → non-boxé n'est PAS
    # incertain (et le câblage engine n'appelle le guard que sous v4).
    e = _reload_engine(monkeypatch)
    rewards = [1.0] * 4 + [0.0] * 4
    comps = [OK_TOKS] * 8
    texts = ["7"] * 8  # aucun boxed
    assert _guard(e, rewards, comps, texts) == (None, None)


def test_code_env_uses_fractional_lattice(monkeypatch):
    # code : uncertain = tronqués seulement, lattice passed/total_tests.
    e = _reload_engine(monkeypatch, 4)
    rewards = [1.0] * 8 + [0.5] * 7 + [0.0]
    comps = [OK_TOKS] * 15 + [TRUNC_TOKS]
    texts = ["x"] * 16  # pas de détection unboxed en code
    reason, robust = _guard(e, rewards, comps, texts, env="opencodeinstruct", tt=2)
    assert reason is None and robust is not None and robust > 0.0
