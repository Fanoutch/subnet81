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

# Interrupteur (20/08) : le verrou sérialise TOUS les forwards de preuve —
# mesuré avec 8 groupes concurrents, la queue gen→POST passe de 3,2 à 5,8 s
# et nos POSTs de +11 à +16 s, soit APRÈS la fermeture du batch (~+12-15 s).
# Sa raison d'être (corruption fork×CUDA) n'a jamais été prouvée : les échecs
# ont continué spec off, verrou posé ET pipeline sériel. Le filtre local
# (local_verif_screen) couvre le risque résiduel. Défaut OFF ; remettre
# RELIQUARY_FORK_GPU_LOCK=1 si les token_tampered reviennent.
_FORK_LOCK_ON = os.environ.get("RELIQUARY_FORK_GPU_LOCK", "0") == "1"


def fork_gpu_guard():
    """Contexte du verrou fork∥forward, ou no-op si désactivé."""
    import contextlib
    return FORK_GPU_LOCK if _FORK_LOCK_ON else contextlib.nullcontext()

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
    proc = None
    try:
        with fork_gpu_guard():
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
    finally:
        # RÉGRESSION 19/08 → CORRIGÉE 20/08 : le passage de subprocess.run à
        # Popen (fix fork) avait supprimé le kill AUTOMATIQUE que run() fait au
        # timeout. Résultat : chaque code généré qui boucle laissait un
        # processus à 100 % CPU, DÉFINITIVEMENT — 1,23/min, 590 en 8 h, quota
        # conteneur (24 CPU) saturé, vLLM affamé à 0,01 CPU → la nuit à zéro
        # acceptée et la dégradation ×2,5/45 min. Le kill est obligatoire.
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
                proc.communicate(timeout=1)   # ferme les pipes + reap
            except Exception:
                pass


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
        with fork_gpu_guard():
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
