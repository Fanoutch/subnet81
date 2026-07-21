import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from difficulty_probe import (  # noqa: E402
    extract_features_code, CODE_HAND_FEATURES, _code_in_zone,
)


_PROMPT = (
    "Write a function that returns the sum of a list of integers.\n"
    "Example:\nInput: [1, 2, 3]\nOutput: 6\n\n"
    "Write your solution as a Python function named `add_list` that "
    "takes 1 argument and returns the result; do not read from stdin or print."
)


def test_features_are_all_present_and_numeric():
    f = extract_features_code(_PROMPT)
    for name in CODE_HAND_FEATURES:
        assert name in f, f"missing feature {name}"
        assert isinstance(f[name], (int, float))


def test_detects_function_keyword_and_examples():
    f = extract_features_code(_PROMPT)
    assert f["kw_function"] == 1
    assert f["kw_return"] == 1
    assert f["has_example"] == 1          # "Example:" / Input:/Output:
    assert f["n_args"] == 1               # from "takes 1 argument"


def test_plurals_and_absence():
    f = extract_features_code("Reverse the string. It takes 2 arguments.")
    assert f["n_args"] == 2
    assert f["kw_function"] == 0          # no "function" word
    assert f["has_example"] == 0


def test_empty_prompt_safe():
    f = extract_features_code("")
    assert f["n_char"] == 0 and f["n_args"] == 0


# ── continuous in-zone labelling (σ ≥ 0.43) ──
def test_code_in_zone_bimodal_is_in_zone():
    sigma, iz = _code_in_zone([1.0, 0.9, 1.0, 0.8, 0.0, 0.1, 0.0, 0.2])
    assert sigma >= 0.43 and iz == 1


def test_code_in_zone_all_pass_is_out():
    sigma, iz = _code_in_zone([1.0] * 8)
    assert sigma == 0.0 and iz == 0


def test_code_in_zone_middling_is_out():
    sigma, iz = _code_in_zone([0.5, 0.5, 0.6, 0.4, 0.5, 0.55, 0.45, 0.5])
    assert iz == 0


def test_code_in_zone_empty_is_out():
    assert _code_in_zone([]) == (0.0, 0)
