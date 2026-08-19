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
import threading

# Verrou fork∥forward (2026-08-19 soir) : un fork() pendant qu'un forward
# CUDA de preuve est en vol dans un autre thread corrompt subtilement les
# calculs (mesuré : logprob_mismatch/token_tampered aux rangs 11-12, 3/32
# verdicts, TOUS pendant la fenêtre de recouvrement). Tenu uniquement
# pendant le fork+exec (~15 ms via Popen), jamais pendant l'exécution des
# tests de l'enfant. L'autre côté : _proof_rollouts (engine) le tient
# pendant chaque forward (~40 ms).
FORK_GPU_LOCK = threading.Lock()

_MEM_LIMIT_BYTES = 512 * 1024 * 1024  # 512 MB per grading process


def _limit() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (_MEM_LIMIT_BYTES, _MEM_LIMIT_BYTES))


# Fix fork 2026-08-19 : ``preexec_fn`` interdit à CPython le chemin rapide
# vfork/posix_spawn ET tient le GIL pendant le fork — sur le process mineur
# (CUDA, ~8 Go résidents, 80 threads), chaque lancement coûtait ~180 ms de
# gel Python, ×16 par groupe = ~3 s de grading plat mesuré (timeline B6 +
# rapport agent). La limite mémoire est posée PAR L'ENFANT lui-même, avant
# toute exécution du code généré — protection identique, fork rapide.
_CHILD_PRELUDE = (
    "import resource as _r; "
    f"_r.setrlimit(_r.RLIMIT_AS, ({_MEM_LIMIT_BYTES}, {_MEM_LIMIT_BYTES}))\n"
)


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
        with FORK_GPU_LOCK:
            proc = subprocess.Popen(
                [sys.executable, "-I", driver],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True,
            )
        stdout, _ = proc.communicate(input=payload, timeout=timeout_s)
        out = json.loads(stdout.strip().splitlines()[-1])
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
    body = _CHILD_PRELUDE + (completion or "") + "\n" + guarded + "\n"
    try:
        with FORK_GPU_LOCK:
            proc = subprocess.Popen(
                [sys.executable, "-I", "-c", body],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
        stdout, _ = proc.communicate(timeout=timeout_s)
    except (subprocess.TimeoutExpired, OSError, ValueError):
        try: proc.kill()
        except Exception: pass
        return 0.0
    passed = stdout.count("P")
    return passed / len(cases)
