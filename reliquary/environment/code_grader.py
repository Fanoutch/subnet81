"""Local exact grader for opencode completions.

Runs a generated completion against recovered assertion-cases in an isolated
subprocess (timeout + memory rlimit). Returns the fraction of cases passed.

This is NOT a security sandbox against adversaries: it runs OUR OWN model's
generated code, so an isolated subprocess with resource limits is sufficient
(the validator needs gVisor because it runs unknown miners' code; we do not).
"""
from __future__ import annotations

import json
import os
import resource
import subprocess
import sys

_MEM_LIMIT_BYTES = 512 * 1024 * 1024  # 512 MB per grading process


def _limit() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (_MEM_LIMIT_BYTES, _MEM_LIMIT_BYTES))


def grade_structured_cases(code: str, cases: list[dict], timeout_s: float = 5.0) -> float:
    """Fraction passed/total via la sémantique EXACTE du validateur (sandbox +
    entry-resolve + _json_equal), dans un subprocess isolé. Never raises; 0.0 si
    crash/timeout/no-case. ``cases`` = liste de dicts ``{entry, args, kwargs,
    expected, compare}`` (format curated structured_cases)."""
    if not cases:
        return 0.0
    driver = os.path.join(os.path.dirname(__file__), "code_grader_driver.py")
    payload = json.dumps({"code": code or "", "cases": cases})
    try:
        result = subprocess.run(
            [sys.executable, "-I", driver],
            input=payload, capture_output=True, text=True,
            timeout=timeout_s, preexec_fn=_limit,
        )
        out = json.loads(result.stdout.strip().splitlines()[-1])
        total = int(out["total"])
        return (int(out["passed"]) / total) if total > 0 else 0.0
    except Exception:
        return 0.0


def grade_completion(completion: str, cases: list[str], timeout_s: float = 5.0) -> float:
    """Fraction of assertion-cases the completion passes. Never raises.

    All assertions run in ONE isolated subprocess: the completion defines the
    function(s) once, then each assertion is guarded by try/except and prints a
    pass/fail marker. Reward = passed / total. Returns 0.0 on timeout/crash/no
    cases.
    """
    if not cases:
        return 0.0
    guarded = "\n".join(
        "try:\n    {c}\n    print('P')\nexcept Exception:\n    print('F')".format(c=c.strip())
        for c in cases
    )
    body = (completion or "") + "\n" + guarded + "\n"
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-c", body],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            preexec_fn=_limit,
        )
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return 0.0
    passed = result.stdout.count("P")
    return passed / len(cases)
