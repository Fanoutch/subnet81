from reliquary.environment.opencodeinstruct import (
    OpenCodeInstructEnvironment,  # noqa: F401  (import-sanity for the env module)
    _extract_python,
)


def test_extract_python_strips_fences():
    md = "Voici:\n```python\ndef f(): return 1\n```\nfin"
    assert "def f(): return 1" in _extract_python(md)


def test_extract_python_falls_back_to_raw():
    assert _extract_python("def g(): return 2") == "def g(): return 2"


# NOTE: compute_reward + structured-case grading + curated dataset behaviour is
# covered by tests/test_opencode_curated_env.py. The previous local-assertion /
# RELIQUARY_OCI_CASES_PATH tests were removed when the env switched to the
# validator's curated structured_cases (no more nvidia-join _load_cases).
