from reliquary.environment.opencodeinstruct import OpenCodeInstructEnvironment

_ROW = {"input": "Add two numbers.",
        "structured_cases": [{"entry": {"kind": "function", "name": "add"},
                              "args": [1, 2], "kwargs": {}, "expected": 3, "compare": "exact"}]}


class _FakeDS:
    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, i):
        return self._rows[i % len(self._rows)]


def _env(monkeypatch, rows):
    monkeypatch.setattr(OpenCodeInstructEnvironment, "_dataset_cache", _FakeDS(rows))
    return OpenCodeInstructEnvironment()


def test_env_is_continuous():
    assert OpenCodeInstructEnvironment.continuous_reward is True


def test_get_problem_appends_contract(monkeypatch):
    env = _env(monkeypatch, [_ROW])
    p = env.get_problem(0)
    assert "function named `add`" in p["prompt"]          # contract appliqué
    assert "takes 2 arguments and returns" in p["prompt"]
    assert isinstance(p["ground_truth"], str) and p["ground_truth"]  # case_id stocké


def test_compute_reward_uses_structured_cases(monkeypatch):
    env = _env(monkeypatch, [_ROW])
    p = env.get_problem(0)
    good = "```python\ndef add(a, b):\n    return a + b\n```"
    bad = "```python\ndef add(a, b):\n    return 99\n```"
    assert env.compute_reward(p, good) == 1.0
    assert env.compute_reward(p, bad) == 0.0
