"""Port v4 : grader OMI — contrat answer_format (réaligné sur a6456b4).

Upstream a SUPPRIMÉ le canal ``Answer:`` le 2026-08-17 (commit 5954076 : il
contournait le tamper guard — la preuve d'intégrité de réponse n'authentifie
que le span ``\\boxed{}``). Contrat final : ``MATH_ANSWER_FORMAT`` —
``"boxed"`` en v4 (pas de boxed → reward 0, AUCUN fallback),
``"boxed_or_trailing_number"`` en v2/v3 (comportement payé historique,
byte-identique). Le strip des délimiteurs LaTeX ``\\( \\) \\[ \\]`` sous
RAW_COMPLETION_PROMPTS est CONSERVÉ par upstream.
"""
import pytest

from reliquary.environment.openmathinstruct import _compute_omi_reward
from tests.v4helpers import reload_constants


@pytest.fixture(autouse=True)
def _restore_constants(monkeypatch):
    yield
    reload_constants(monkeypatch)


def _reward(monkeypatch, completion, gt="7", v4=True):
    monkeypatch.setattr("reliquary.constants.RAW_COMPLETION_PROMPTS", v4)
    monkeypatch.setattr(
        "reliquary.constants.MATH_ANSWER_FORMAT",
        "boxed" if v4 else "boxed_or_trailing_number",
    )
    return _compute_omi_reward({"ground_truth": gt}, completion)


def test_v4_boxed_scores(monkeypatch):
    assert _reward(monkeypatch, "donc $\\boxed{7}$") == 1.0


def test_v4_unboxed_answer_line_is_zero(monkeypatch):
    # a6456b4 : le canal Answer: n'existe plus — non boxé = 0, tamper guard.
    assert _reward(monkeypatch, "blah\nAnswer: 7\n") == 0.0
    assert _reward(monkeypatch, "**Answer:** 7") == 0.0


def test_v4_trailing_number_is_zero(monkeypatch):
    # v4 : pas de fallback trailing-number non plus.
    assert _reward(monkeypatch, "le résultat est\n7") == 0.0


def test_v4_boxed_with_latex_delims(monkeypatch):
    # le strip \( \) \[ \] sous RAW_COMPLETION_PROMPTS est conservé upstream
    assert _reward(monkeypatch, "donc $\\boxed{\\(7\\)}$") == 1.0


def test_v4_wrong_boxed_is_zero(monkeypatch):
    assert _reward(monkeypatch, "donc $\\boxed{3}$") == 0.0


def test_v3_trailing_number_fallback_unchanged(monkeypatch):
    # v3 byte-identique : boxed sinon trailing-number de la dernière ligne.
    assert _reward(monkeypatch, "donc $\\boxed{7}$", v4=False) == 1.0
    assert _reward(monkeypatch, "explication\n7", v4=False) == 1.0
    assert _reward(monkeypatch, "blah\nAnswer: 7\nfin 3", v4=False) == 0.0


def test_constants_math_answer_format(monkeypatch):
    c = reload_constants(monkeypatch)
    assert c.MATH_ANSWER_FORMAT == "boxed_or_trailing_number"
    c = reload_constants(monkeypatch, 4)
    assert c.MATH_ANSWER_FORMAT == "boxed"
