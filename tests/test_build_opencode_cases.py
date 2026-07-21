import json

from scripts.build_opencode_cases import cases_from_unit_tests


def test_cases_from_unit_tests_parses_assertions():
    raw = json.dumps(["\nassert factorial(5) == 120\n", "\nassert factorial(0) == 1\n"])
    cases = cases_from_unit_tests(raw)
    assert cases == ["assert factorial(5) == 120", "assert factorial(0) == 1"]


def test_cases_from_unit_tests_handles_garbage():
    assert cases_from_unit_tests("not json") == []
    assert cases_from_unit_tests(json.dumps([])) == []
    assert cases_from_unit_tests(json.dumps("x")) == []
    assert cases_from_unit_tests(None) == []
