# Plan B — MixController + signal verdicts + submitter par-env — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement task-by-task. Steps use checkbox (`- [ ]`).
> **NOTE:** `reliquary-miner-priv` is NOT git — replace "commit" by a **review checkpoint**. Run pytest with `cd /root/subnet81/reliquary-miner-priv && PYTHONPATH=. python3 -m pytest ...`.

**Goal:** Fournir le "cerveau" du multi-env (MixController qui alloue les 8 slots/window par env selon notre rendement réel), le signal qui l'alimente (champs d'observabilité du Verdict), et le param `?env=` du submitter — toutes pièces pures, CPU-testables, sans GPU.

**Architecture :** MixController = module pur sans dépendance (entrée: `record_outcome(env, rewarded)`, sortie: `target_slots() -> {env: int}` sommant à 8, plancher ≥1, EMA). Le Verdict gagne les champs optionnels d'observabilité (dont `rewarded`) pour que l'engine puisse lire l'issue réelle de chaque submission. Le submitter accepte `env=` pour lire le cooldown par-env. Le câblage dans engine.py (mapper verdict→env, piloter le bake) = Plan C.

**Tech Stack :** Python 3.12, pydantic v2, pytest. Pas de GPU.

---

### Task 1 : Champs d'observabilité du Verdict (signal + anti-crash)

**Files:**
- Modify: `reliquary/protocol/submission.py` (classe `Verdict`, ~L181-187)
- Create: `tests/test_verdict_observability.py`

Le `Verdict` a `extra="forbid"` et n'a pas les champs enrichis → parsing strict casserait sur les verdicts du validateur live. Ajouter les champs optionnels (default None), alignés sur upstream.

- [ ] **Step 1 : Écrire le test**

```python
# tests/test_verdict_observability.py
from reliquary.protocol.submission import Verdict, VerdictsResponse, RejectReason

def test_verdict_accepts_observability_fields():
    v = Verdict(merkle_root="a"*64, accepted=True, reason=RejectReason.ACCEPTED,
                ts=1.0, rewarded=True, selected_for_batch=True, accepted_into_pool=True)
    assert v.rewarded is True and v.selected_for_batch is True

def test_verdict_back_compat_without_fields():
    v = Verdict(merkle_root="b"*64, accepted=False, reason=RejectReason.GRAIL_FAIL, ts=2.0)
    assert v.rewarded is None and v.window_n is None

def test_verdicts_response_parses_enriched_payload():
    payload = {"verdicts": [{"merkle_root": "c"*64, "accepted": True,
               "reason": "accepted", "ts": 3.0, "rewarded": True,
               "selected_for_batch": False, "queue_wait_ms": 12.5}]}
    resp = VerdictsResponse.model_validate(payload)
    assert resp.verdicts[0].rewarded is True
    assert resp.verdicts[0].selected_for_batch is False
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `cd /root/subnet81/reliquary-miner-priv && PYTHONPATH=. python3 -m pytest tests/test_verdict_observability.py -v`
Expected: FAIL (`Verdict` rejette `rewarded` — extra forbidden)

- [ ] **Step 3 : Ajouter les champs (aligné upstream)**

Dans `reliquary/protocol/submission.py`, classe `Verdict`, juste après la ligne `ts: float = Field(...)` :

```python
    # Optional observability fields (validator enriches recent verdicts; older
    # records omit them). Mirrors upstream so strict parsing never breaks.
    arrival_ts: float | None = None
    decision_ts: float | None = None
    submitted_drand_round: int | None = None
    arrival_drand_round: int | None = None
    drand_delta: int | None = None
    seal_trigger_round: int | None = None
    prompt_hash_lead: str | None = None
    canonical_rank: int | None = None
    accepted_into_pool: bool | None = None
    selected_for_batch: bool | None = None
    rewarded: bool | None = None
    reject_stage: str | None = None
    reject_reason: str | None = None
    queue_wait_ms: float | None = None
    verify_ms: float | None = None
    total_ms: float | None = None
```

- [ ] **Step 4 : Lancer, vérifier le succès**

Run: `cd /root/subnet81/reliquary-miner-priv && PYTHONPATH=. python3 -m pytest tests/test_verdict_observability.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5 : Non-régression** — `PYTHONPATH=. python3 -m pytest tests/unit/test_verdicts_endpoint.py -v` (si présent) Expected: PASS. Puis checkpoint de revue.

---

### Task 2 : MixController (module pur)

**Files:**
- Create: `reliquary/miner/mix_controller.py`
- Create: `tests/test_mix_controller.py`

Alloue `total_slots` (=8) entre `envs`, proportionnel au rendement EMA `rewarded/invested`, plancher ≥1/env, somme exacte = total. Cold start neutre.

- [ ] **Step 1 : Écrire le test**

```python
# tests/test_mix_controller.py
from reliquary.miner.mix_controller import MixController

def test_cold_start_is_even_with_floor():
    mc = MixController(["math", "code"], total_slots=8, slot_floor=1)
    slots = mc.target_slots()
    assert sum(slots.values()) == 8
    assert slots["math"] >= 1 and slots["code"] >= 1
    assert abs(slots["math"] - slots["code"]) <= 1  # neutre ~ équilibré

def test_high_yield_env_gets_more_but_floor_respected():
    mc = MixController(["math", "code"], total_slots=8, slot_floor=1, alpha=1.0)
    for _ in range(20):
        mc.record_outcome("code", rewarded=True)
        mc.record_outcome("math", rewarded=False)
    slots = mc.target_slots()
    assert sum(slots.values()) == 8
    assert slots["code"] > slots["math"]
    assert slots["math"] >= 1  # plancher: math jamais à 0

def test_sum_always_equals_total():
    mc = MixController(["math", "code"], total_slots=8, slot_floor=1, alpha=0.3)
    import random
    rng = random.Random(0)
    for _ in range(200):
        mc.record_outcome(rng.choice(["math", "code"]), rewarded=rng.random() < 0.4)
        assert sum(mc.target_slots().values()) == 8

def test_unknown_env_outcome_ignored():
    mc = MixController(["math"], total_slots=8, slot_floor=1)
    mc.record_outcome("nope", rewarded=True)  # ne lève pas
    assert mc.target_slots() == {"math": 8}
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `cd /root/subnet81/reliquary-miner-priv && PYTHONPATH=. python3 -m pytest tests/test_mix_controller.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3 : Implémenter le MixController**

```python
# reliquary/miner/mix_controller.py
"""Adaptive slot allocator across envs, driven by our own /verdicts outcomes.

Pure logic, no I/O. The engine feeds outcomes via record_outcome(env, rewarded)
(mapping each verdict back to the env it submitted), and reads target_slots()
at window open to decide how many of the 8 global slots go to each env.
Allocation = proportional to EMA reward-rate per env, with an exploration floor
so no env is ever starved (we keep measuring it).
"""
from __future__ import annotations

_NEUTRAL_YIELD = 0.5  # cold-start prior so unseen envs get a fair share


class MixController:
    def __init__(self, envs, total_slots: int = 8, slot_floor: int = 1,
                 alpha: float = 0.1) -> None:
        self.envs = list(envs)
        self.total = int(total_slots)
        self.floor = int(slot_floor)
        self.alpha = float(alpha)
        self._yield: dict[str, float | None] = {e: None for e in self.envs}

    def record_outcome(self, env: str, rewarded: bool) -> None:
        if env not in self._yield:
            return
        x = 1.0 if rewarded else 0.0
        prev = self._yield[env]
        self._yield[env] = x if prev is None else (1 - self.alpha) * prev + self.alpha * x

    def _effective_yield(self, env: str) -> float:
        y = self._yield[env]
        return _NEUTRAL_YIELD if y is None else y

    def target_slots(self) -> dict[str, int]:
        n = len(self.envs)
        alloc = {e: self.floor for e in self.envs}
        remaining = self.total - self.floor * n
        if remaining <= 0:
            # floors consume the budget; clamp proportionally if over-subscribed
            return {e: max(0, self.total // n) for e in self.envs} if remaining < 0 else alloc
        weights = {e: max(self._effective_yield(e), 1e-9) for e in self.envs}
        wsum = sum(weights.values())
        quota = {e: remaining * weights[e] / wsum for e in self.envs}
        base = {e: int(quota[e]) for e in self.envs}
        for e in self.envs:
            alloc[e] += base[e]
        leftover = remaining - sum(base.values())
        # largest fractional remainder gets the leftover units
        order = sorted(self.envs, key=lambda e: quota[e] - base[e], reverse=True)
        for e in order[:leftover]:
            alloc[e] += 1
        return alloc
```

- [ ] **Step 4 : Lancer, vérifier le succès**

Run: `cd /root/subnet81/reliquary-miner-priv && PYTHONPATH=. python3 -m pytest tests/test_mix_controller.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5 : Checkpoint de revue.**

---

### Task 3 : Param `?env=` du submitter (cooldown par-env)

**Files:**
- Modify: `reliquary/miner/submitter.py` (`get_window_state_v2` ~L200-210 ; `get_window_state_v2_with_resp` ~L213-241)
- Create: `tests/test_submitter_env_param.py`

Ajouter un param optionnel `env: str | None = None` qui, quand fourni, appende `?env=<url-encoded>` à l'URL `/state`. Défaut None = comportement actuel inchangé.

- [ ] **Step 1 : Écrire le test**

```python
# tests/test_submitter_env_param.py
import asyncio
from reliquary.miner.submitter import build_state_url

def test_build_state_url_no_env():
    assert build_state_url("http://v:8080") == "http://v:8080/state"

def test_build_state_url_with_env():
    assert build_state_url("http://v:8080", "opencodeinstruct") == \
        "http://v:8080/state?env=opencodeinstruct"

def test_build_state_url_encodes():
    assert build_state_url("http://v:8080", "a b") == "http://v:8080/state?env=a%20b"
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `cd /root/subnet81/reliquary-miner-priv && PYTHONPATH=. python3 -m pytest tests/test_submitter_env_param.py -v`
Expected: FAIL (`build_state_url` n'existe pas)

- [ ] **Step 3 : Ajouter `build_state_url` + threader `env`**

En haut de `reliquary/miner/submitter.py`, ajouter l'import :
```python
from urllib.parse import quote
```
Ajouter le helper (près des autres fonctions de module) :
```python
def build_state_url(url: str, env: str | None = None) -> str:
    """`{url}/state`, with optional `?env=` (per-env cooldown, validator #88)."""
    base = f"{url}/state"
    return f"{base}?env={quote(env, safe='')}" if env is not None else base
```
Dans `get_window_state_v2`, ajouter le param `env: str | None = None` (après `*,`) et remplacer `f"{url}/state"` par `build_state_url(url, env)`.
Dans `get_window_state_v2_with_resp`, ajouter `env: str | None = None` (après `*,`) et remplacer `await client.get(f"{url}/state", ...)` par `await client.get(build_state_url(url, env), ...)`.

- [ ] **Step 4 : Lancer, vérifier le succès**

Run: `cd /root/subnet81/reliquary-miner-priv && PYTHONPATH=. python3 -m pytest tests/test_submitter_env_param.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5 : Non-régression** — `PYTHONPATH=. python3 -m pytest tests/ -k submitter -v` Expected: pas de nouveau FAIL. Checkpoint de revue.

---

## Self-Review

- §5 MixController (record_outcome / target_slots, EMA, floor, cold start) → Task 2 ✅
- §8 Port schéma Verdict (anti-crash + signal `rewarded`) → Task 1 ✅
- §6 submitter `?env=` (#88) → Task 3 ✅
- Types cohérents : `record_outcome(env, rewarded)` / `target_slots()` / `build_state_url(url, env)` utilisés identiquement test↔impl.
- Hors-scope (Plan C) : refonte engine.py par-env (selector/buckets/bake), mapping verdict→env, appel de `target_slots()` pour piloter le bake, registration de l'env opencode dans `environment/__init__.py`, génération vLLM. Ces pièces nécessitent le GPU pour validation end-to-end.
