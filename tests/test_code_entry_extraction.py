"""Sélection du bloc fencé NOTÉ, par point d'entrée du contrat.

PORT des PR upstream #202 / #203 (image validateur `7b4a407`, live depuis le
25/08 ~21:30 UTC).

Protocole v2-v4 : on note ``matches[-1]``, le DERNIER bloc fencé, avec repli sur
la complétion brute quand il n'y a pas de bloc. Cette règle date d'un modèle
« chat » qui terminait toujours par son implémentation.

Sous le prompt de raisonnement v5, le modèle termine souvent par une
démonstration d'usage, un listing de sortie attendue ou un bloc de tests : le
span noté n'est plus l'implémentation, et un rollout correct est noté 0.

Depuis v5 : le bloc noté est le dernier qui *définit la fonction d'entrée du
contrat*, avec repli sur ``matches[-1]`` si aucun ne la définit, et "" s'il n'y
a aucun bloc fencé.

⚠️ La borne s'arrête à v5 pour que v2-v4 restent BYTE-EXACTS : c'est notre
repli (`RELIQUARY_PROTOCOL_VERSION=4`). Les rewards code sont
``validator_authoritative_reward = True`` — aucun `reward_mismatch` ne peut
naître de ce changement ; il ne corrige que NOTRE sélection.
"""

import hashlib
import json

import pytest


ENTRY = "is_balanced_parentheses"

IMPLEMENTATION = (
    "def is_balanced_parentheses(s):\n"
    "    counter = 0\n"
    "    for ch in s:\n"
    "        if ch == '(':\n"
    "            counter += 1\n"
    "        elif ch == ')':\n"
    "            counter -= 1\n"
    "            if counter < 0:\n"
    "                return False\n"
    "    return counter == 0"
)

USAGE_DEMO = (
    "print(is_balanced_parentheses('(())'))\n"
    "print(is_balanced_parentheses('(()'))"
)


def _fence(body: str) -> str:
    return f"```python\n{body}\n```"


@pytest.fixture
def v5(monkeypatch):
    """Épingle le protocole live ; la règle d'entrée s'applique à partir de v5."""
    import reliquary.constants as constants

    monkeypatch.setattr(constants, "PROTOCOL_VERSION", 5, raising=False)


@pytest.fixture
def v4(monkeypatch):
    """Épingle le protocole de repli ; comportement historique attendu."""
    import reliquary.constants as constants

    monkeypatch.setattr(constants, "PROTOCOL_VERSION", 4, raising=False)


# ---------------------------------------------------------------------------
# _entry_function_name : le point d'entrée noté, lu dans les cases.
# ---------------------------------------------------------------------------

def test_entry_function_name_reads_the_function_contract():
    from reliquary.environment.opencodeinstruct import _entry_function_name

    cases = [{"entry": {"kind": "function", "name": ENTRY}, "args": ["()"]}]
    assert _entry_function_name(cases) == ENTRY


def test_entry_function_name_is_none_for_a_method_entry():
    from reliquary.environment.opencodeinstruct import _entry_function_name

    cases = [{"entry": {"kind": "method", "name": "solve", "class": "S"}}]
    assert _entry_function_name(cases) is None


def test_entry_function_name_is_none_for_empty_cases():
    from reliquary.environment.opencodeinstruct import _entry_function_name

    assert _entry_function_name([]) is None
    assert _entry_function_name(None) is None


# ---------------------------------------------------------------------------
# v5 : le bloc qui DÉFINIT l'entrée gagne, quelle que soit sa position.
# ---------------------------------------------------------------------------

def test_v5_skips_a_trailing_usage_demo(v5):
    from reliquary.environment.opencodeinstruct import _extract_python

    text = f"Voici la solution.\n\n{_fence(IMPLEMENTATION)}\n\nExemple :\n\n{_fence(USAGE_DEMO)}\n"
    assert _extract_python(text, entry_name=ENTRY) == IMPLEMENTATION


def test_v5_skips_a_trailing_expected_output_listing(v5):
    from reliquary.environment.opencodeinstruct import _extract_python

    text = f"{_fence(IMPLEMENTATION)}\n\nSortie attendue :\n\n```\nTrue\nFalse\n```\n"
    assert _extract_python(text, entry_name=ENTRY) == IMPLEMENTATION


def test_v5_takes_the_last_block_that_defines_the_entry(v5):
    """Deux blocs définissent l'entrée : le DERNIER gagne (le brouillon perd)."""
    from reliquary.environment.opencodeinstruct import _extract_python

    draft = "def is_balanced_parentheses(s):\n    return None"
    text = f"{_fence(draft)}\nCorrection :\n{_fence(IMPLEMENTATION)}\n{_fence(USAGE_DEMO)}"
    assert _extract_python(text, entry_name=ENTRY) == IMPLEMENTATION


def test_v5_falls_back_to_last_block_when_none_defines_the_entry(v5):
    from reliquary.environment.opencodeinstruct import _extract_python

    a = "def helper(x):\n    return x"
    b = "def other(x):\n    return x + 1"
    assert _extract_python(f"{_fence(a)}\n{_fence(b)}", entry_name=ENTRY) == b


def test_v5_without_entry_name_keeps_the_last_block(v5):
    """Entrée « méthode » (entry_name None) : règle positionnelle inchangée."""
    from reliquary.environment.opencodeinstruct import _extract_python

    a = "def is_balanced_parentheses(s):\n    return True"
    b = "print('demo')"
    assert _extract_python(f"{_fence(a)}\n{_fence(b)}") == b


def test_v5_returns_empty_when_there_is_no_fenced_block(v5):
    from reliquary.environment.opencodeinstruct import _extract_python

    prose = "Il faut compter les parenthèses ouvrantes et fermantes."
    assert _extract_python(prose, entry_name=ENTRY) == ""
    assert _extract_python(prose) == ""


def test_v5_unparsable_block_is_skipped_not_crashing(v5):
    """Un bloc au code non parsable ne peut pas définir l'entrée — et ne lève pas."""
    from reliquary.environment.opencodeinstruct import _extract_python

    broken = "def is_balanced_parentheses(s:\n    return ((("
    text = f"{_fence(IMPLEMENTATION)}\n{_fence(broken)}"
    assert _extract_python(text, entry_name=ENTRY) == IMPLEMENTATION


def test_v5_all_blocks_unparsable_falls_back_to_last(v5):
    from reliquary.environment.opencodeinstruct import _extract_python

    broken1 = "def is_balanced_parentheses(s:\n    ???"
    broken2 = "!!! not python"
    text = f"{_fence(broken1)}\n{_fence(broken2)}"
    assert _extract_python(text, entry_name=ENTRY) == broken2


def test_v5_nested_definition_does_not_select_the_block(v5):
    """Un `def` imbriqué n'est pas résolu par le grader — il ne doit pas gagner."""
    from reliquary.environment.opencodeinstruct import _extract_python

    nested = (
        "def wrapper():\n"
        "    def is_balanced_parentheses(s):\n"
        "        return True\n"
        "    return is_balanced_parentheses"
    )
    text = f"{_fence(IMPLEMENTATION)}\n{_fence(nested)}"
    assert _extract_python(text, entry_name=ENTRY) == IMPLEMENTATION


def test_v5_accepts_an_async_definition(v5):
    from reliquary.environment.opencodeinstruct import _extract_python

    impl = "async def is_balanced_parentheses(s):\n    return True"
    text = f"{_fence(impl)}\n{_fence(USAGE_DEMO)}"
    assert _extract_python(text, entry_name=ENTRY) == impl


def test_v5_prefix_name_does_not_select_the_block(v5):
    from reliquary.environment.opencodeinstruct import _extract_python

    almost = "def is_balanced_parentheses_v2(s):\n    return True"
    text = f"{_fence(IMPLEMENTATION)}\n{_fence(almost)}"
    assert _extract_python(text, entry_name=ENTRY) == IMPLEMENTATION


def test_v5_span_offsets_point_at_the_selected_bytes(v5):
    from reliquary.environment.opencodeinstruct import _select_python_span

    text = f"intro\n{_fence(IMPLEMENTATION)}\ntail\n{_fence(USAGE_DEMO)}\n"
    body, start, end = _select_python_span(text, entry_name=ENTRY)
    assert body == IMPLEMENTATION
    assert text[start:end] == IMPLEMENTATION


# ---------------------------------------------------------------------------
# v2-v4 : BYTE-EXACT, c'est le repli. Aucune de ces valeurs ne doit bouger.
# ---------------------------------------------------------------------------

def test_v4_still_grades_the_last_block_even_a_usage_demo(v4):
    from reliquary.environment.opencodeinstruct import _extract_python

    text = f"{_fence(IMPLEMENTATION)}\n{_fence(USAGE_DEMO)}"
    assert _extract_python(text, entry_name=ENTRY) == USAGE_DEMO
    assert _extract_python(text) == USAGE_DEMO


def test_v4_still_falls_back_to_the_raw_completion(v4):
    from reliquary.environment.opencodeinstruct import _extract_python

    prose = "def g(): return 2"
    assert _extract_python(prose, entry_name=ENTRY) == prose
    assert _extract_python(prose) == prose


def test_v4_empty_completion_is_empty(v4):
    from reliquary.environment.opencodeinstruct import _extract_python

    assert _extract_python("") == ""
    assert _extract_python("", entry_name=ENTRY) == ""


def test_v5_empty_completion_is_empty(v5):
    from reliquary.environment.opencodeinstruct import _extract_python

    assert _extract_python("") == ""


def test_v4_tilde_fence_and_bare_fence_unchanged(v4):
    from reliquary.environment.opencodeinstruct import _extract_python

    body = "def f():\n    return 1"
    assert _extract_python(f"~~~py\n{body}\n~~~") == body
    assert _extract_python(f"```\n{body}\n```") == body


# ---------------------------------------------------------------------------
# Le PROMPT ne bouge pas : les tokens fixent le forced-seed.
# ---------------------------------------------------------------------------

_CASES = [{"entry": {"kind": "function", "name": "solve"}, "args": [1], "expected": 2}]


class _FakeDataset:
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {"input": "Do the thing.", "structured_cases": json.dumps(_CASES)}


@pytest.mark.parametrize(
    "version,expected_sha",
    [
        (4, "49af28299032875403f7a71650272fdb7aa9cdcd158d79a0738f927747d57db4"),
        (5, "0265f2624020d29ac5c91664793a94cb25a0ab48993ed74dda167f5c3685278d"),
    ],
)
def test_prompt_bytes_are_unchanged_by_the_graded_span_port(
    monkeypatch, version, expected_sha
):
    """Empreintes relevées AVANT le port. Un octet d'écart = 100 % de rejet."""
    import reliquary.constants as constants
    from reliquary.environment import opencodeinstruct as module

    monkeypatch.setattr(constants, "PROTOCOL_VERSION", version, raising=False)
    monkeypatch.setattr(
        module.OpenCodeInstructEnvironment, "_dataset_cache", _FakeDataset()
    )
    problem = module.OpenCodeInstructEnvironment().get_problem(0)
    assert hashlib.sha256(problem["prompt"].encode()).hexdigest() == expected_sha


def test_compute_reward_grades_the_entry_block(monkeypatch, v5):
    """Le câblage : compute_reward passe bien entry_name à l'extracteur."""
    from reliquary.environment import opencodeinstruct as module

    monkeypatch.setattr(
        module.OpenCodeInstructEnvironment, "_dataset_cache", _FakeDataset()
    )
    env = module.OpenCodeInstructEnvironment()
    problem = env.get_problem(0)

    graded = {}

    def _fake_grade(code, cases, timeout_s=None):
        graded["code"] = code
        return 1.0

    monkeypatch.setattr(module, "grade_structured_cases", _fake_grade)

    impl = "def solve(x):\n    return x + 1"
    demo = "print(solve(1))"
    completion = f"{_fence(impl)}\n{_fence(demo)}"
    assert env.compute_reward(problem, completion) == 1.0
    assert graded["code"] == impl
