from reliquary.environment.code_grader import grade_completion

CASES = [
    "assert factorial(0) == 1",
    "assert factorial(1) == 1",
    "assert factorial(2) == 2",
    "assert factorial(3) == 6",
    "assert factorial(4) == 24",
    "assert factorial(5) == 120",
]
CORRECT = "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)"
BUGGY = "def factorial(n):\n    return n"


def test_correct_completion_scores_one():
    assert grade_completion(CORRECT, CASES, timeout_s=5) == 1.0


def test_buggy_completion_scores_partial():
    r = grade_completion(BUGGY, CASES, timeout_s=5)
    assert 0.0 < r < 1.0


def test_crashing_completion_scores_zero():
    code = "def factorial(n):\n    raise ValueError()"
    assert grade_completion(code, CASES, timeout_s=5) == 0.0


def test_empty_cases_scores_zero():
    assert grade_completion(CORRECT, [], timeout_s=5) == 0.0


def test_never_raises_on_garbage():
    assert grade_completion("this is not python !!!", CASES, timeout_s=5) == 0.0
