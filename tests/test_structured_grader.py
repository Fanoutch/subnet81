from reliquary.environment.code_grader import grade_structured_cases

ADD = "def add(a, b):\n    return a + b\n"


def _case(name, args, expected):
    return {"entry": {"kind": "function", "name": name}, "args": args,
            "kwargs": {}, "expected": expected, "compare": "exact"}


def test_all_pass_is_one():
    cases = [_case("add", [1, 2], 3), _case("add", [0, 0], 0), _case("add", [-1, 5], 4)]
    assert grade_structured_cases(ADD, cases) == 1.0


def test_all_fail_is_zero():
    cases = [_case("add", [1, 2], 99), _case("add", [0, 0], 99)]
    assert grade_structured_cases(ADD, cases) == 0.0


def test_partial_is_fraction():
    cases = [_case("add", [1, 2], 3), _case("add", [1, 1], 99), _case("add", [2, 2], 4),
             _case("add", [5, 5], 99)]
    assert grade_structured_cases(ADD, cases) == 0.5


def test_list_value_comparison():
    code = "def rev(x):\n    return list(reversed(x))\n"
    assert grade_structured_cases(code, [_case("rev", [[1, 2, 3]], [3, 2, 1])]) == 1.0


def test_float_isclose():
    code = "def half(x):\n    return x / 2\n"
    assert grade_structured_cases(code, [_case("half", [1], 0.5)]) == 1.0


def test_bool_strict_not_int():
    # _json_equal: True != 1 (type strict). Une fn renvoyant 1 où on attend True échoue.
    code = "def f():\n    return 1\n"
    c = {"entry": {"kind": "function", "name": "f"}, "args": [], "kwargs": {},
         "expected": True, "compare": "exact"}
    assert grade_structured_cases(code, [c]) == 0.0


def test_crash_returns_zero_not_raise():
    code = "def add(a, b):\n    raise RuntimeError('boom')\n"
    assert grade_structured_cases(code, [_case("add", [1, 2], 3)]) == 0.0


def test_no_cases_zero():
    assert grade_structured_cases(ADD, []) == 0.0
