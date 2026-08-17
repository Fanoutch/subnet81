"""Task 3 du port v4 : grader OMI — ligne ``Answer:`` + strip délimiteurs LaTeX.

Sous v4 (raw-completion, modèle base sans template) la réponse sort via une
ligne ``Answer: X`` au lieu de ``\\boxed{}`` ; les délimiteurs LaTeX inline
``\\( \\)`` / display ``\\[ \\]`` sont retirés à la normalisation. Gaté sur
RAW_COMPLETION_PROMPTS pour que le reward v3 (la quantité PAYÉE) reste
byte-identique. Parité upstream 8c38992 (commit c54d716).
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
    return _compute_omi_reward({"ground_truth": gt}, completion)


def test_v4_answer_line(monkeypatch):
    assert _reward(monkeypatch, "blah\nAnswer: 7\n") == 1.0


def test_v4_answer_line_markdown_emphasis(monkeypatch):
    assert _reward(monkeypatch, "**Answer:** 7") == 1.0


def test_v4_last_answer_line_wins(monkeypatch):
    assert _reward(monkeypatch, "Answer: 3\nAnswer: 7") == 1.0


def test_v4_inline_latex_delims_stripped(monkeypatch):
    assert _reward(monkeypatch, r"Answer: \(7\)") == 1.0
    assert _reward(monkeypatch, r"Answer: \[7\]") == 1.0


def test_v4_boxed_still_takes_precedence(monkeypatch):
    assert _reward(monkeypatch, "Answer: 3\ndonc $\\boxed{7}$") == 1.0


def test_v4_wrong_answer_line_is_zero(monkeypatch):
    assert _reward(monkeypatch, "Answer: 3") == 0.0


def test_v3_answer_line_ignored(monkeypatch):
    # v3 : la branche Answer: n'existe pas — fallback trailing number sur la
    # dernière ligne ("fin 3" ne matche pas le pattern numérique → 0.0).
    assert _reward(monkeypatch, "blah\nAnswer: 7\nfin 3", v4=False) == 0.0


def test_v3_boxed_unchanged(monkeypatch):
    assert _reward(monkeypatch, "donc $\\boxed{7}$", v4=False) == 1.0
