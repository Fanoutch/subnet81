"""Prompt math rendu par le template versionné du protocole (v5+).

Le validateur live (profil qwen3-4b-base-dapo-reliquary-v1, protocole 6)
rend le prompt math par ``openmathinstruct-step-by-step-v1``. Un prompt
différent d'un seul caractère change les tokens, donc le forced-seed : rejet
``prompt_mismatch`` à 100 %, et le disjoncteur coupe la COLDKEY entière.
Le chemin legacy (protocole < 5) doit rester byte-exact.
"""
import pytest

from reliquary.environment.openmathinstruct import OpenMathInstructEnvironment

QUESTION = "Compute $2 + 2$. What is $x$ if $x^2 = 4$ and $x > 0$?"


def _env(rows):
    env = OpenMathInstructEnvironment.__new__(OpenMathInstructEnvironment)
    env._dataset = rows
    env._eligible = None
    return env


@pytest.mark.parametrize("pv", [5, 6])
def test_prompt_uses_step_by_step_template(monkeypatch, pv):
    monkeypatch.setattr("reliquary.constants.PROTOCOL_VERSION", pv)
    env = _env([{"problem": QUESTION, "expected_answer": "2"}])
    p = env.get_problem(0)
    assert p["prompt"] == (
        "Solve the following math problem step by step.\n\n"
        + QUESTION
        + "\n\nPut your final answer within \\boxed{}."
    )
    assert p["ground_truth"] == "2"


def test_legacy_prompt_unchanged_below_v5(monkeypatch):
    monkeypatch.setattr("reliquary.constants.PROTOCOL_VERSION", 4)
    env = _env([{"problem": QUESTION, "expected_answer": "2"}])
    assert env.get_problem(0)["prompt"] == (
        QUESTION + "\n\nPut your final answer within \\boxed{}."
    )


def test_id_is_hash_of_question_not_prompt(monkeypatch):
    # Comme upstream : l'id reste celui de l'énoncé brut.
    import hashlib

    monkeypatch.setattr("reliquary.constants.PROTOCOL_VERSION", 6)
    env = _env([{"problem": QUESTION, "expected_answer": "2"}])
    assert env.get_problem(0)["id"] == hashlib.sha256(
        QUESTION.encode()
    ).hexdigest()[:16]


def test_problem_source_is_returned_for_the_prior(monkeypatch):
    # prior math v2 : la source OMI sert de trait ; le prompt et la réponse
    # attendue (consensus) ne changent pas.
    monkeypatch.setattr("reliquary.constants.PROTOCOL_VERSION", 6)
    env = _env([{"problem": QUESTION, "expected_answer": "2",
                 "problem_source": "augmented_math"}])
    p = env.get_problem(0)
    assert p["source"] == "augmented_math"
    assert p["ground_truth"] == "2"
    assert _env([{"problem": QUESTION, "expected_answer": "2"}]).get_problem(0)["source"] is None


def test_loader_reads_problem_source_column(monkeypatch):
    monkeypatch.setattr("reliquary.constants.OMI_TRAIN_SHARDS_ONLY", True)
    captured = {}

    class _FakeVP:
        def __init__(self, repo, revision, **kwargs):
            captured.update(kwargs)

    from reliquary.environment import virtual_parquet as vp_mod
    monkeypatch.setattr(vp_mod, "VirtualParquetDataset", _FakeVP)
    from reliquary.environment.openmathinstruct import _load_dataset
    _load_dataset("nvidia/OpenMathInstruct-2", "rev")
    assert captured["columns"][:2] == ["problem", "expected_answer"]
    assert "problem_source" in captured["columns"]
