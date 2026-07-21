# Plan A — Grader code local + env opencode — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> **NOTE:** `reliquary-miner-priv` is NOT a git repo (plain dir synced by rsync). Replace every "commit" step with a **review checkpoint** — pause for the user to review the diff. Do not run `git`.

**Goal:** Donner au miner un grading EXACT et local de l'env opencode, en reconstruisant les cas de test cachés (join miroir public ↔ `nvidia/OpenCodeInstruct` par `id`), pour permettre la sélection σ-zone sur le code comme sur le math.

**Architecture :** Un builder one-time produit un artefact compact `id → cases` (assertions) à partir du dataset NVIDIA public. Un module grader exécute une complétion contre ces cas dans une sandbox légère (subprocess + timeout + RLIMIT) et renvoie le taux de cas passés. L'env `opencodeinstruct` (mode prompt_only) charge l'artefact et utilise le grader dans `compute_reward`.

**Tech Stack :** Python 3.12, `datasets`/`pyarrow` (build only), subprocess sandbox (runtime). Pas de GPU, pas de gVisor (on exécute le code de notre propre modèle, pas adversaire).

**Pré-requis vérifié (PoC 2026-06-11) :** join id confirmé (4/4 matches), `unit_tests` = assertions args+expected, grading local discrimine correct (1.0) vs buggé (0.2).

---

### Task 1 : Builder de l'artefact `id → cases`

**Files:**
- Create: `scripts/build_opencode_cases.py`
- Create: `tests/test_build_opencode_cases.py`

Produit `data/opencode_cases.json` : `{ id: [assertion_str, ...] }` pour tous les `id` du miroir public, en streamant NVIDIA (pas de download complet).

- [ ] **Step 1 : Écrire le test (fonction de parsing pure, sans réseau)**

```python
# tests/test_build_opencode_cases.py
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
```

- [ ] **Step 2 : Lancer le test, vérifier l'échec**

Run: `python3 -m pytest tests/test_build_opencode_cases.py -v`
Expected: FAIL (`ModuleNotFoundError: scripts.build_opencode_cases`)

- [ ] **Step 3 : Implémenter le builder**

```python
# scripts/build_opencode_cases.py
"""One-time build of the id->cases artifact for local opencode grading.

Joins the public miner mirror (R0mAI/opencodeinstruct-prompts) to the public
source nvidia/OpenCodeInstruct by `id`, extracting the unit_tests (assertion
strings = args+expected). Writes a compact {id: [assertion,...]} JSON.
The full NVIDIA dataset is streamed (never fully downloaded).
"""
from __future__ import annotations
import json, os, sys

MIRROR_REPO = "R0mAI/opencodeinstruct-prompts"
SOURCE_REPO = "nvidia/OpenCodeInstruct"
OUT_PATH = os.environ.get("RELIQUARY_OCI_CASES_PATH", "data/opencode_cases.json")

def cases_from_unit_tests(raw) -> list[str]:
    """Parse the string-encoded unit_tests list into clean assertion strings."""
    if not isinstance(raw, str):
        return []
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [s.strip() for s in parsed if isinstance(s, str) and s.strip()]

def build(out_path: str = OUT_PATH, scan_cap: int = 5_000_000) -> dict[str, list[str]]:
    from datasets import load_dataset
    mirror = load_dataset(MIRROR_REPO, split="train")
    want = {r["id"] for r in mirror}
    print(f"mirror: {len(want)} ids", file=sys.stderr)
    src = load_dataset(SOURCE_REPO, split="train", streaming=True)
    out: dict[str, list[str]] = {}
    scanned = 0
    for row in src:
        scanned += 1
        rid = row.get("id")
        if rid in want and rid not in out:
            cases = cases_from_unit_tests(row.get("unit_tests"))
            if cases:
                out[rid] = cases
        if len(out) >= len(want) or scanned >= scan_cap:
            break
    print(f"scanned {scanned}, recovered {len(out)}/{len(want)}", file=sys.stderr)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f)
    return out

if __name__ == "__main__":
    build()
```

- [ ] **Step 4 : Lancer le test, vérifier le succès**

Run: `python3 -m pytest tests/test_build_opencode_cases.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5 : Checkpoint de revue** — montrer `scripts/build_opencode_cases.py` + résultat des tests. (Le run réel `build()` qui streame NVIDIA se fera en Task 4 / sur GPU au build prod.)

---

### Task 2 : Grader local (sandbox légère)

**Files:**
- Create: `reliquary/environment/code_grader.py`
- Create: `tests/test_code_grader.py`

`grade_completion(completion: str, cases: list[str], timeout_s: float) -> float` = taux de cas passés, exécution isolée. Reward 0.0 si tout échoue/crash. Ne lève jamais.

- [ ] **Step 1 : Écrire le test**

```python
# tests/test_code_grader.py
from reliquary.environment.code_grader import grade_completion

CASES = [
    "assert factorial(0) == 1", "assert factorial(1) == 1",
    "assert factorial(2) == 2", "assert factorial(3) == 6",
    "assert factorial(4) == 24", "assert factorial(5) == 120",
]
CORRECT = "def factorial(n):\n    return 1 if n<=1 else n*factorial(n-1)"
BUGGY   = "def factorial(n):\n    return n"

def test_correct_completion_scores_one():
    assert grade_completion(CORRECT, CASES, timeout_s=5) == 1.0

def test_buggy_completion_scores_partial():
    r = grade_completion(BUGGY, CASES, timeout_s=5)
    assert 0.0 < r < 1.0

def test_crashing_completion_scores_zero():
    assert grade_completion("def factorial(n):\n    raise ValueError()", CASES, 5) == 0.0

def test_empty_cases_scores_zero():
    assert grade_completion(CORRECT, [], timeout_s=5) == 0.0

def test_never_raises_on_garbage():
    assert grade_completion("this is not python !!!", CASES, 5) == 0.0
```

- [ ] **Step 2 : Lancer le test, vérifier l'échec**

Run: `python3 -m pytest tests/test_code_grader.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3 : Implémenter le grader**

```python
# reliquary/environment/code_grader.py
"""Local exact grader for opencode completions.

Runs a generated completion against recovered assertion-cases in an isolated
subprocess (timeout + memory rlimit). Returns the fraction of cases passed.
NOT a security sandbox against adversaries — it runs OUR OWN model's code, so a
subprocess with limits is sufficient (the validator needs gVisor; we do not).
"""
from __future__ import annotations
import subprocess, sys, resource

_MEM_LIMIT_BYTES = 512 * 1024 * 1024  # 512 MB per case process

def _limit():
    resource.setrlimit(resource.RLIMIT_AS, (_MEM_LIMIT_BYTES, _MEM_LIMIT_BYTES))

def grade_completion(completion: str, cases: list[str], timeout_s: float = 5.0) -> float:
    if not cases:
        return 0.0
    # Run all assertions in ONE process: define the function once, then assert.
    body = completion + "\n" + "\n".join(
        f"try:\n    {c.strip()}\n    print('P')\nexcept Exception:\n    print('F')"
        for c in cases
    )
    try:
        r = subprocess.run(
            [sys.executable, "-I", "-c", body],
            capture_output=True, text=True, timeout=timeout_s,
            preexec_fn=_limit,
        )
    except (subprocess.TimeoutExpired, Exception):
        return 0.0
    passed = r.stdout.count("P")
    return passed / len(cases)
```

- [ ] **Step 4 : Lancer le test, vérifier le succès**

Run: `python3 -m pytest tests/test_code_grader.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5 : Checkpoint de revue** — montrer le module + tests verts.

---

### Task 3 : Env opencode (prompt_only) câblé sur le grader local

**Files:**
- Create: `reliquary/environment/opencodeinstruct.py` (porter depuis `origin/main`, adapter `compute_reward`)
- Modify: `reliquary/environment/__init__.py` (enregistrer l'env dans `load_environment`/le registre — suivre le pattern de `openmathinstruct`)
- Create: `tests/test_opencodeinstruct_env.py`

L'env charge le miroir prompts (prompt_only) + l'artefact `id→cases`, et `compute_reward(problem, completion)` = `grade_completion(extract_python(completion), cases_by_id[problem['id']])`.

- [ ] **Step 1 : Écrire le test (avec dataset + cases factices, pas de réseau)**

```python
# tests/test_opencodeinstruct_env.py
from reliquary.environment.opencodeinstruct import OpenCodeInstructEnvironment, _extract_python

def test_extract_python_strips_fences():
    md = "Voici:\n```python\ndef f(): return 1\n```\nfin"
    assert "def f(): return 1" in _extract_python(md)

def test_compute_reward_uses_local_cases(monkeypatch, tmp_path):
    import json
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps({"abc": ["assert add(2,3)==5", "assert add(0,0)==0"]}))
    monkeypatch.setenv("RELIQUARY_OCI_CASES_PATH", str(cases))
    env = OpenCodeInstructEnvironment.__new__(OpenCodeInstructEnvironment)
    env._load_cases()  # charge l'artefact sans toucher au dataset
    problem = {"id": "abc", "prompt": "écris add(a,b)"}
    good = "```python\ndef add(a,b): return a+b\n```"
    bad  = "```python\ndef add(a,b): return a-b\n```"
    assert env.compute_reward(problem, good) == 1.0
    assert env.compute_reward(problem, bad) == 0.5  # add(0,0)==0 passe, add(2,3) non
```

- [ ] **Step 2 : Lancer le test, vérifier l'échec**

Run: `python3 -m pytest tests/test_opencodeinstruct_env.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3 : Porter l'env upstream et adapter `compute_reward`**

Récupérer la base verbatim : `git -C /root/subnet81/reliquary show origin/main:reliquary/environment/opencodeinstruct.py` → copier dans `reliquary/environment/opencodeinstruct.py`. Puis remplacer le bloc grader-distant par le grader LOCAL :

```python
# en tête du fichier
import json, os, re
from reliquary.environment.code_grader import grade_completion
from reliquary.constants import GRADER_EVAL_TIMEOUT_SECONDS  # = 5 (si absent, fixer 5.0)

_FENCE_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)

def _extract_python(text: str) -> str:
    if not text:
        return ""
    m = _FENCE_RE.findall(text)
    return m[-1] if m else text

# dans la classe OpenCodeInstructEnvironment :
    _DEFAULT_CASES_PATH: str = "data/opencode_cases.json"

    def _load_cases(self) -> None:
        path = os.environ.get("RELIQUARY_OCI_CASES_PATH", self._DEFAULT_CASES_PATH)
        try:
            with open(path) as f:
                self._cases_by_id = json.load(f)
        except (OSError, json.JSONDecodeError):
            self._cases_by_id = {}

    def compute_reward(self, problem: dict, completion: str) -> float:
        cases = self._cases_by_id.get(problem.get("id", ""))
        if not cases:
            return 0.0
        code = _extract_python(completion or "")
        return grade_completion(code, cases, timeout_s=float(GRADER_EVAL_TIMEOUT_SECONDS))
```

Garder de l'upstream : le chargement du miroir prompts en `_prompt_only` (révision `f50bef12…`), `get_problem` (qui doit exposer `id` dans le dict retourné), `name = "opencodeinstruct"`. Supprimer la dépendance `GraderClient`/socket. `__init__` appelle `self._load_cases()`.

- [ ] **Step 4 : Lancer le test, vérifier le succès**

Run: `python3 -m pytest tests/test_opencodeinstruct_env.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5 : Checkpoint de revue** — montrer l'env porté + le diff `compute_reward` + tests verts.

---

### Task 4 : Test d'intégration bout-en-bout (vraies données, sans GPU)

**Files:**
- Create: `tests/test_opencode_integration.py` (marqué `@pytest.mark.network`)

Prouve la chaîne complète sur de vraies données : build d'un mini-artefact (quelques ids) → grader → reward correct/buggé.

- [ ] **Step 1 : Écrire le test d'intégration**

```python
# tests/test_opencode_integration.py
import pytest
from scripts.build_opencode_cases import build
from reliquary.environment.code_grader import grade_completion

@pytest.mark.network
def test_real_join_and_grade(tmp_path):
    out = tmp_path / "cases.json"
    # build complet (streame NVIDIA jusqu'à couvrir le miroir) — long; en CI réduire via fixture
    cases_map = build(out_path=str(out))
    assert len(cases_map) > 1000  # on a récupéré une large part du miroir
    # un cas connu (factorial) doit grader correctement
    fid = "e7ca4436b5c004b2c07534b50b1e4c83"
    if fid in cases_map:
        good = "def factorial(n):\n    return 1 if n<=1 else n*factorial(n-1)"
        assert grade_completion(good, cases_map[fid], 5) == 1.0
```

- [ ] **Step 2 : Lancer (réseau requis)**

Run: `python3 -m pytest tests/test_opencode_integration.py -v -m network`
Expected: PASS (peut prendre plusieurs minutes — streaming NVIDIA).

- [ ] **Step 3 : Checkpoint de revue final** — confirmer : artefact construit, taille cohérente (~50k ids), grading exact sur cas réel. Le grader local opencode est prêt à être branché (Plan C).

---

## Self-Review (couverture spec)

- C1 (cases reconstructibles) → Tasks 1 + 4 ✅
- §7.2 (build id→cases, reuse parse) → Task 1 ✅
- §7.3 (sandbox légère, pas gVisor) → Task 2 ✅
- §7.1 (env prompt_only, révision) → Task 3 ✅
- §7.4 (reward = taux passés → σ-zone) → Task 3 (reward) ; σ-zone = Plan C ✅
- Fidélité validateur (parse structured/résolution #85) : **raffinement différé** — la v1 exécute les assertions brutes (prouvé PoC) ; aligner sur la résolution de fonction du `worker.py` upstream est un raffinement post-v1 (à ajouter aux TODOs si σ diverge en prod).

Hors-scope de ce plan (Plans B/C) : plomberie multi-env, MixController, activation du mix, génération vLLM.
