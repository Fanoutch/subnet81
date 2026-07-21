# Alignement opencode curated + sélection σ-continue — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (recommended) ou subagent-driven-development. Steps en checkbox (`- [ ]`).
> **NOTE:** `reliquary-miner-priv` n'est PAS git → « commit » = **checkpoint de revue**. Tests : `cd /root/subnet81/reliquary-miner-priv && PYTHONPATH=. python3 -m pytest <chemin> -v`.

**Goal :** rendre l'env opencode du mineur capable de soumettre du code in-zone — aligné byte-pour-byte sur le validateur courant (curated dataset + prompt-contract + comparaison par valeur), avec une sélection σ-continue pour le reward continu.

**Architecture :** on PORTE les pièces déterministes du validateur (`VirtualParquetDataset`, `_contract_instruction`, l'exécution worker + comparaison `_json_equal`) → parité par construction ; on AJOUTE une branche σ-continue à `_try_select` ; le math binaire reste intact (dispatch par flag d'env).

**Tech Stack :** Python 3.12, pyarrow, huggingface_hub, subprocess, pytest. Pas de GPU. Pas de gVisor (on grade NOTRE code).

## Global Constraints

- **Source de vérité = validateur à jour** : clone `/root/subnet81/reliquary` sur `origin/main` (`196e275`). Copier les blocs **verbatim** depuis les fichiers cités (parité).
- **Curated pinné** : `R0mAI/opencodeinstruct-curated` @ `d3caaefc3b46f8642b251f9efaeccf0d1e95b0a7` (overridable `RELIQUARY_OCI_REPO`/`RELIQUARY_OCI_REVISION`).
- **Math (OMI) : NE PAS TOUCHER** — binaire, déjà aligné. Toute la nouveauté est gatée par un flag d'env.
- **`validator_authoritative_reward=True` pour le code** : le reward soumis n'est pas re-vérifié en valeur (pas de `REWARD_MISMATCH`), MAIS le filtre σ s'applique → notre grade local sert à calculer σ et **pré-sélectionner l'in-zone** (on ne « skip » pas le grade local).
- **`SIGMA_MIN=0.43`** (constants), marge code `RELIQUARY_CODE_SIGMA_MARGIN` **défaut 0.03** (cible std≥0.46).
- Tests réseau (chargement HF réel) marqués/optionnels ; toute la logique est testable CPU avec des datasets/fakes.

---

### Task 1 : Porter `VirtualParquetDataset` dans le mineur

**Files:**
- Create: `reliquary/environment/virtual_parquet.py` (copie verbatim du validateur)
- Test: `tests/test_virtual_parquet_import.py`

**Interfaces:**
- Produces: `VirtualParquetDataset(repo, revision, *, columns=None, data_dir="data", cache_row_groups=64, fs=None)` avec `__len__`, `__getitem__(idx)`, `get_row(idx)`.

- [ ] **Step 1 : Copier le fichier verbatim**

Copier intégralement `/root/subnet81/reliquary/reliquary/environment/virtual_parquet.py` →
`/root/subnet81/reliquary-miner-priv/reliquary/environment/virtual_parquet.py` (aucune modif :
deps = `pyarrow.parquet`, `huggingface_hub.HfFileSystem`, `bisect`, `threading`, `collections.OrderedDict` — toutes standalone).

```bash
cp /root/subnet81/reliquary/reliquary/environment/virtual_parquet.py \
   /root/subnet81/reliquary-miner-priv/reliquary/environment/virtual_parquet.py
```

- [ ] **Step 2 : Test d'import + injection fs (sans réseau)**

```python
# tests/test_virtual_parquet_import.py
def test_virtual_parquet_imports_and_constructs():
    from reliquary.environment.virtual_parquet import VirtualParquetDataset
    # Construction pure (pas de réseau tant qu'on n'appelle pas len/getitem).
    ds = VirtualParquetDataset("owner/repo", "rev123", columns=["input", "structured_cases"])
    assert ds._repo == "owner/repo" and ds._revision == "rev123"
    assert ds._columns == ["input", "structured_cases"]
```

- [ ] **Step 3 : Lancer**

Run: `cd /root/subnet81/reliquary-miner-priv && PYTHONPATH=. python3 -m pytest tests/test_virtual_parquet_import.py -v`
Expected: PASS (si `pyarrow`/`huggingface_hub` absents → `pip install pyarrow huggingface_hub`, déjà deps du mineur).

- [ ] **Step 4 : Checkpoint de revue.**

---

### Task 2 : Grader structuré en parité (`grade_structured_cases`)

**Files:**
- Create: `reliquary/environment/code_grader_driver.py` (driver subprocess, embarque l'exécution + comparaison du validateur)
- Modify: `reliquary/environment/code_grader.py` (ajouter `grade_structured_cases`, garder `grade_completion` legacy)
- Test: `tests/test_structured_grader.py`

**Interfaces:**
- Produces: `grade_structured_cases(code: str, cases: list[dict], timeout_s: float = 5.0) -> float` → fraction `passed/total` ∈ [0,1], never raises. Chaque case = `{"entry": {"kind","name",...}, "args": [...], "kwargs": {...}, "expected": <val>, "compare": "exact"}`.

Le driver réplique la sémantique du validateur (parité). On porte **verbatim** ses fonctions :
- de `/root/subnet81/reliquary/reliquary/environment/grader/worker.py` : la résolution d'entry
  `_resolve_function` (l.201-229) + ses helpers AST `_defined_functions_in_order`,
  `_accepts_arity`, `_returns_a_value`, `_call_graph_roots`, l'appel de la fonction
  (kind function/method) et `_json_safe` (sanitize la sortie).
- de `/root/subnet81/reliquary/reliquary/environment/grader/server.py` : `_json_equal` (l.551-567)
  + `_outputs_match` (l.545-549).

- [ ] **Step 1 : Écrire le test (public API, end-to-end subprocess)**

```python
# tests/test_structured_grader.py
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
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `PYTHONPATH=. python3 -m pytest tests/test_structured_grader.py -v`
Expected: FAIL (`grade_structured_cases` n'existe pas)

- [ ] **Step 3 : Créer le driver subprocess (parité verbatim)**

Créer `reliquary/environment/code_grader_driver.py` : lit `{"code","cases"}` en JSON sur stdin,
exec le code dans un namespace isolé, et pour chaque case : résout l'entry (par `name` si présent,
sinon `_resolve_function`), appelle avec `args`/`kwargs`, `_json_safe(output)`, compare via
`_outputs_match(output, expected, compare)`, compte les passés ; imprime `{"passed": N, "total": M}`.
**Coller verbatim** dans ce fichier les helpers cités (worker `_resolve_function` + AST + appel +
`_json_safe` ; server `_json_equal` + `_outputs_match`). Squelette du `__main__` :

```python
# reliquary/environment/code_grader_driver.py  (exécuté: python -I code_grader_driver.py)
import json, sys, math, ast  # math/ast requis par les helpers collés
# <<< COLLER VERBATIM ICI: _json_safe, _resolve_function, _defined_functions_in_order,
#     _accepts_arity, _returns_a_value, _call_graph_roots (worker.py) ;
#     _json_equal, _outputs_match (server.py) >>>

def _call(ns, entry, args, kwargs):
    name = (entry or {}).get("name")
    fn = ns.get(name) if name and callable(ns.get(name)) else _resolve_function(ns, _CODE, len(args))
    if fn is None:
        raise RuntimeError("no entry")
    return fn(*args, **(kwargs or {}))

def main():
    req = json.loads(sys.stdin.read())
    code, cases = req["code"], req["cases"]
    ns = {}
    passed = 0
    try:
        exec(code, ns)
    except Exception:
        print(json.dumps({"passed": 0, "total": len(cases)})); return
    global _CODE; _CODE = code
    for c in cases:
        try:
            out = _json_safe(_call(ns, c.get("entry"), c.get("args", []), c.get("kwargs", {})))
            if _outputs_match(out, c.get("expected"), c.get("compare", "exact")):
                passed += 1
        except Exception:
            pass
    print(json.dumps({"passed": passed, "total": len(cases)}))

if __name__ == "__main__":
    main()
```

- [ ] **Step 4 : Ajouter `grade_structured_cases` dans `code_grader.py`**

```python
# en tête de reliquary/environment/code_grader.py
import json, os

def grade_structured_cases(code: str, cases: list[dict], timeout_s: float = 5.0) -> float:
    """Fraction passed/total via la sémantique EXACTE du validateur (entry-resolve
    + _json_equal), dans un subprocess isolé. Never raises; 0.0 si crash/timeout/no-case."""
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
```

- [ ] **Step 5 : Lancer, vérifier le succès**

Run: `PYTHONPATH=. python3 -m pytest tests/test_structured_grader.py -v`
Expected: PASS (8 tests)

- [ ] **Step 6 : Non-régression du grader legacy** — `PYTHONPATH=. python3 -m pytest tests/ -k grader -v`. Checkpoint de revue.

---

### Task 3 : Repointer l'env opencode sur le curated (+ contract + cases structurées + flag continu)

**Files:**
- Modify: `reliquary/environment/opencodeinstruct.py` (réécriture alignée validateur)
- Test: `tests/test_opencode_curated_env.py`

**Interfaces:**
- Consumes: `VirtualParquetDataset` (Task 1), `grade_structured_cases` (Task 2).
- Produces: `OpenCodeInstructEnvironment` avec `name="opencodeinstruct"`,
  `continuous_reward = True` (nouveau flag), `__len__`, `get_problem(idx) -> {"prompt","ground_truth","id"}`,
  `compute_reward(problem, completion) -> float`.

- [ ] **Step 1 : Écrire le test (avec dataset factice, sans réseau)**

```python
# tests/test_opencode_curated_env.py
from reliquary.environment.opencodeinstruct import OpenCodeInstructEnvironment

_ROW = {"input": "Add two numbers.",
        "structured_cases": [{"entry": {"kind": "function", "name": "add"},
                              "args": [1, 2], "kwargs": {}, "expected": 3, "compare": "exact"}]}

class _FakeDS:
    def __init__(self, rows): self._rows = rows
    def __len__(self): return len(self._rows)
    def __getitem__(self, i): return self._rows[i % len(self._rows)]

def _env(monkeypatch, rows):
    monkeypatch.setattr(OpenCodeInstructEnvironment, "_dataset_cache", _FakeDS(rows))
    return OpenCodeInstructEnvironment()

def test_env_is_continuous(monkeypatch):
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
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `PYTHONPATH=. python3 -m pytest tests/test_opencode_curated_env.py -v`
Expected: FAIL (`continuous_reward` absent / prompt sans contract / mauvais grading)

- [ ] **Step 3 : Réécrire `opencodeinstruct.py` aligné validateur**

Remplacer le contenu par l'alignement suivant (copier `_extract_python`, `_load_dataset`,
`_contract_instruction` **verbatim** depuis le validateur `opencodeinstruct.py:34-88`) :

```python
from __future__ import annotations
import hashlib, json, os, re
from pathlib import Path
from typing import ClassVar
from reliquary.constants import GRADER_EVAL_TIMEOUT_SECONDS
from reliquary.environment.code_grader import grade_structured_cases

# <<< COLLER VERBATIM: _FENCE_RE, _extract_python, _load_dataset, _contract_instruction
#     depuis /root/subnet81/reliquary/reliquary/environment/opencodeinstruct.py:34-88 >>>

class OpenCodeInstructEnvironment:
    name: str = "opencodeinstruct"
    validator_authoritative_reward: ClassVar[bool] = True
    continuous_reward: ClassVar[bool] = True          # <-- dispatch sélection (Task 4)

    _dataset_cache: ClassVar = {}
    _CURATED_REPO: ClassVar[str] = "R0mAI/opencodeinstruct-curated"
    _CURATED_REVISION: ClassVar[str] = "d3caaefc3b46f8642b251f9efaeccf0d1e95b0a7"

    def __init__(self) -> None:
        repo = os.environ.get("RELIQUARY_OCI_REPO", self._CURATED_REPO)
        revision = os.environ.get("RELIQUARY_OCI_REVISION", self._CURATED_REVISION)
        cache = OpenCodeInstructEnvironment._dataset_cache
        if isinstance(cache, dict):
            key = (repo, revision)
            if key not in cache:
                cache[key] = _load_dataset(repo, revision)
            self._dataset = cache[key]
        else:
            self._dataset = cache
        self._cases_by_id: dict[str, list[dict]] = {}

    def __len__(self) -> int:
        return len(self._dataset)

    def get_problem(self, index: int) -> dict:
        idx = index % len(self._dataset)
        row = self._dataset[idx]
        prompt: str = row["input"]
        cases = self._row_cases(row)
        prompt = prompt + _contract_instruction(cases)
        problem_id = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        case_id = hashlib.sha256(
            (problem_id + json.dumps(cases, sort_keys=True, separators=(",", ":"))).encode()
        ).hexdigest()[:16]
        self._cases_by_id[case_id] = cases
        return {"prompt": prompt, "ground_truth": case_id, "id": problem_id}

    def compute_reward(self, problem: dict, completion: str) -> float:
        case_id = problem.get("ground_truth", "")
        if not isinstance(case_id, str):
            return 0.0
        cases = self._cases_by_id.get(case_id)
        if not cases:
            return 0.0
        code = _extract_python(completion or "")
        return grade_structured_cases(code, cases, timeout_s=float(GRADER_EVAL_TIMEOUT_SECONDS))

    @staticmethod
    def _row_cases(row) -> list[dict]:
        raw = row.get("structured_cases", [])
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                return []
        if not isinstance(raw, list):
            return []
        return [dict(c) for c in raw if isinstance(c, dict)]
```

Note : déclarer aussi `continuous_reward = False` sur l'env math (`openmathinstruct.py`,
attribut de classe) pour que le dispatch Task 4 soit explicite.

- [ ] **Step 4 : Lancer, vérifier le succès**

Run: `PYTHONPATH=. python3 -m pytest tests/test_opencode_curated_env.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5 : Sanity import + registration** — `PYTHONPATH=. python3 -c "from reliquary.environment import load_environment; print(load_environment('opencodeinstruct').continuous_reward)"` → `True`. Checkpoint.

---

### Task 4 : `_try_select` env-aware + branche σ-continue

**Files:**
- Modify: `reliquary/miner/engine.py` (`_try_select` + ses 2 appels) + `reliquary/miner/zone.py` (helper variance, optionnel)
- Test: `tests/test_try_select_continuous.py`

**Interfaces:**
- Consumes: `env.continuous_reward` (Task 3), `SIGMA_MIN` (constants), `_passes_local_dist`/dédup existants.
- Produces: `_try_select(self, rollouts, env)` — dispatch ; renvoie `(subset, k|None)`. Math inchangé ;
  code → subset de M_ROLLOUTS dont `std(rewards) >= SIGMA_MIN + CODE_SIGMA_MARGIN`.

- [ ] **Step 1 : Écrire le test (pure CPU, sans GPU)**

```python
# tests/test_try_select_continuous.py
from reliquary.miner.engine import MiningEngine, _select_continuous_subset

def _r(reward, bt=True):
    # rollout minimal : tokens uniques (anti-dédup), reward, bt_ok, q10/median hauts
    _r.n = getattr(_r, "n", 0) + 1
    return {"all_tokens": [_r.n, _r.n + 1000], "reward": reward, "bt_ok": bt,
            "q10_local": 0.9, "median_local": 0.9, "p_stop_local": 0.9, "in_eos": True}

def test_continuous_bimodal_accepts():
    # 4×~1.0 + 4×~0.0 → std ~0.5 ≥ 0.46
    rolls = [_r(1.0), _r(0.95), _r(0.9), _r(1.0), _r(0.0), _r(0.05), _r(0.1), _r(0.0)]
    subset = _select_continuous_subset(rolls, size=8, sigma_target=0.46)
    assert subset is not None and len(subset) == 8

def test_continuous_all_pass_rejects():
    rolls = [_r(1.0) for _ in range(8)]            # σ=0
    assert _select_continuous_subset(rolls, size=8, sigma_target=0.46) is None

def test_continuous_middling_rejects():
    rolls = [_r(0.5) for _ in range(8)]            # σ≈0
    assert _select_continuous_subset(rolls, size=8, sigma_target=0.46) is None

def test_continuous_picks_extremes_from_pool():
    # pool de 12 : doit composer 8 max-variance qui franchit le seuil
    rolls = ([_r(1.0), _r(0.98), _r(0.95), _r(0.92)] + [_r(0.5)] * 4 +
             [_r(0.05), _r(0.02), _r(0.0), _r(0.0)])
    subset = _select_continuous_subset(rolls, size=8, sigma_target=0.46)
    assert subset is not None

def test_dispatch_binary_unchanged():
    eng = object.__new__(MiningEngine)
    class _Math: continuous_reward = False
    rolls = [_r(1.0) for _ in range(3)] + [_r(0.0) for _ in range(5)]
    subset, k = eng._try_select(rolls, _Math())   # voie binaire (k-band) intacte
    assert subset is not None and k == 3
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `PYTHONPATH=. python3 -m pytest tests/test_try_select_continuous.py -v`
Expected: FAIL (`_select_continuous_subset` absent ; `_try_select` ne prend pas `env`)

- [ ] **Step 3 : Ajouter le helper variance + le dispatch**

Ajouter en module dans `engine.py` (près de `_skip_for_out_of_zone`) :

```python
def _std(xs: list[float]) -> float:
    n = len(xs)
    if n == 0:
        return 0.0
    m = sum(xs) / n
    return (sum((x - m) ** 2 for x in xs) / n) ** 0.5

def _select_continuous_subset(rollouts, size, sigma_target):
    """Sous-ensemble de `size` rollouts maximisant la dispersion des rewards
    continus, retourné seulement si std >= sigma_target. Heuristique : trier par
    reward et prendre les extrêmes (moitié haute + moitié basse) — la composition
    de variance maximale pour une taille fixe. None si le seuil n'est pas atteint."""
    if len(rollouts) < size:
        return None
    ordered = sorted(rollouts, key=lambda r: r["reward"])
    lo = size // 2
    hi = size - lo
    subset = ordered[:lo] + ordered[len(ordered) - hi:]
    if _std([r["reward"] for r in subset]) >= sigma_target:
        return subset
    return None
```

Modifier la signature `_try_select(self, rollouts, env)`. Après le bloc commun (dédup +
`_passes_local_dist` → `kept`, et le calcul `bt_ok_rollouts`/`non_bt_ok`), insérer le dispatch
AVANT la boucle k binaire :

```python
        # Multi-env: voie σ-continue pour les envs à reward continu (code).
        if getattr(env, "continuous_reward", False):
            import os as _os
            margin = float(_os.environ.get("RELIQUARY_CODE_SIGMA_MARGIN", "0.03"))
            # Préférer bt_ok ; compléter avec non-bt_ok dans la limite du budget.
            pool = bt_ok_rollouts if len(bt_ok_rollouts) >= M_ROLLOUTS else kept
            subset = _select_continuous_subset(pool, M_ROLLOUTS, SIGMA_MIN + margin)
            if subset is None:
                return None, None
            n_non_bt = sum(1 for r in subset if not r["bt_ok"])
            if n_non_bt > MAX_NON_BTOK_IN_SUBMISSION:
                return None, None
            return subset, None
        # ... (voie binaire math existante inchangée ci-dessous) ...
```

- [ ] **Step 4 : Câbler les 2 appelants pour passer `env`**

Dans `_pre_bake_batch` (l.~2260) : `subset, k = self._try_select(rollouts, env)` (`env` déjà local
depuis Task 6). Dans `_process_one_completion` (l.~2662) : `subset, k = self._try_select(rollouts, env)`
(résoudre `env = self.envs[env_name] if env_name else self.env` en tête de la méthode si pas déjà fait).

- [ ] **Step 5 : Lancer, vérifier le succès**

Run: `PYTHONPATH=. python3 -m pytest tests/test_try_select_continuous.py -v`
Expected: PASS (5 tests)

- [ ] **Step 6 : Non-régression globale** — `PYTHONPATH=. python3 -c "import reliquary.miner.engine"` puis la suite filtrée (mêmes ignores que la session précédente) : le compte de fails pré-existants ne doit pas augmenter, +N nouveaux tests verts. Checkpoint de revue.

---

## Self-Review

**Couverture spec (`2026-06-30-miner-multienv-management-design.md`) :**
- A (env curated + VirtualParquetDataset) → Task 1 + Task 3 ✅
- B (grader parité `passed/total`, format `{entry,args,expected}`, comparaison par valeur) → Task 2 ✅
- C (sélection σ-continue, dispatch par env) → Task 4 ✅
- D (pré-filtre local + allocation MixController) → câblé par compute_reward (Task 3) + σ-zone existant + MixController déjà en place (Task 6) ✅
- Anti-`PROMPT_MISMATCH` (contract instruction) → Task 3 Step 3 (`_contract_instruction` verbatim) ✅

**Placeholders :** les blocs « COLLER VERBATIM » référencent des fichiers/lignes EXACTS du
validateur (action pleinement définie : copier ce code précis), pas du TBD. Tests = code complet.

**Cohérence des types :** `grade_structured_cases(code, cases, timeout_s)->float`,
`_select_continuous_subset(rollouts, size, sigma_target)->list|None`,
`_try_select(self, rollouts, env)->(subset, k|None)`, `env.continuous_reward:bool` — identiques
test↔impl↔appelants.

**Hors-scope (rappel) :** prédicteur TF-IDF (Couche 2), validation GPU/activation Phase 2,
fast-forward du clone upstream. Le math (OMI) n'est pas touché (dispatch gated par `continuous_reward`).

**À valider GPU après ces 4 tâches :** chargement réel du curated (réseau), bake code end-to-end,
parité σ local↔validateur sur quelques prompts réels, puis activer `RELIQUARY_ACTIVE_ENVS=
openmathinstruct,opencodeinstruct`.
