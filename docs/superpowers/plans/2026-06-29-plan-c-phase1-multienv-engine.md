# Plan C — Phase 1 : câblage multi-env dans `engine.py` (mix forcé 100% math) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement task-by-task. Steps use checkbox (`- [ ]`).
> **NOTE:** `reliquary-miner-priv` is NOT git — replace every "commit" by a **review checkpoint**. Run pytest with `cd /root/subnet81/reliquary-miner-priv && PYTHONPATH=. python3 -m pytest ...`.

**Goal:** Brancher le signal de rendement multi-env dans le mineur de prod (chaque rollout porte son `env_name` → polling `/verdicts` → `MixController.record_outcome` → `target_slots()` disponible), **sans charger l'env code et sans changer d'un iota le comportement math actuel**. La répartition réelle du bake par-env (Task 6) et sa validation de parité (Task 7) se font **sur le GPU**.

**Architecture :** 2 couches en overlay (spec §4). Couche 1 = `MixController` (déjà codé+testé, Plan B). Ce plan fournit son **alimentation** : (a) tag `env_name` sur chaque entrée du pool + chaque `RolloutSubmission`, (b) un client `/verdicts` porté depuis `custom_miner`, (c) une boucle async qui mappe `merkle_root → env soumis → record_outcome`. Tout est pur/CPU-testable. Le chemin GRAIL/drand/submit reste **env-agnostique et inchangé** (spec C7).

**Tech Stack :** Python 3.12, pydantic v2, httpx, pytest. **Pas de GPU pour Tasks 1–5.**

## Global Constraints

- **Phase 1 = math-only.** `RELIQUARY_ACTIVE_ENVS` reste à sa valeur par défaut `"openmathinstruct"`. `self.envs` n'a **qu'une seule clé**. L'env `opencodeinstruct` n'est **jamais chargé** dans ce plan (il dépend de `data/opencode_cases.json`, build GPU — Phase 2).
- **Zéro régression math.** Toute modification du chemin de génération/submit doit être **byte-identique** pour le cas single-env. La parité réelle se valide sur GPU (Task 7).
- **`T_PROTO=0.9` est imposé par le validateur** (verifier recalcule à T_PROTO). **Interdit d'y toucher** (spec C4).
- **`env_name` vit sur `RolloutSubmission`** (`protocol/submission.py:103`), **PAS** sur `BatchSubmissionRequest`. Le routage par-env tague chaque `RolloutSubmission`.
- **Back-compat pool disque :** des entrées rechargées du disque (persistence legacy) peuvent ne **pas** avoir la clé `env_name` → toute lecture doit **défaulter**, jamais crasher.
- **Cut CPU/GPU :** Tasks 1–5 = codables ET testables sans GPU (livrables verts avant de prendre le GPU). Task 6 = codable sans GPU mais **non validable** sans GPU. Task 7 = GPU.
- Commandes de test : `cd /root/subnet81/reliquary-miner-priv && PYTHONPATH=. python3 -m pytest <chemin> -v`.

---

### Task 1 : Helpers purs `pick_bake_env` + `entry_env_name` (CPU)

**Files:**
- Modify: `reliquary/miner/mix_controller.py` (append deux fonctions module)
- Create: `tests/test_mix_helpers.py`

**Interfaces:**
- Produces:
  - `pick_bake_env(target_slots: dict[str, int], pool_counts: dict[str, int]) -> str` — l'env le plus en déficit (`target - current`) dans le pool ; tie-break déterministe par ordre d'insertion de `target_slots`.
  - `entry_env_name(entry: dict, default: str) -> str` — `entry["env_name"]` ou `default` si absent/falsy (entrées disque legacy).

- [ ] **Step 1 : Écrire le test**

```python
# tests/test_mix_helpers.py
from reliquary.miner.mix_controller import pick_bake_env, entry_env_name


def test_pick_bake_env_picks_largest_deficit():
    # math veut 6, en a 5 (déficit 1) ; code veut 2, en a 0 (déficit 2) → code
    assert pick_bake_env({"math": 6, "code": 2}, {"math": 5, "code": 0}) == "code"


def test_pick_bake_env_empty_pool_picks_first_with_highest_target():
    assert pick_bake_env({"math": 8, "code": 0}, {}) == "math"


def test_pick_bake_env_tie_breaks_by_target_order():
    # déficits égaux (4 et 4) → premier dans target_slots
    assert pick_bake_env({"math": 4, "code": 4}, {"math": 0, "code": 0}) == "math"


def test_pick_bake_env_single_env_always_that_env():
    assert pick_bake_env({"math": 8}, {"math": 3}) == "math"


def test_entry_env_name_reads_key():
    assert entry_env_name({"env_name": "code", "prompt_idx": 1}, "math") == "code"


def test_entry_env_name_defaults_when_missing():
    assert entry_env_name({"prompt_idx": 1}, "math") == "math"


def test_entry_env_name_defaults_when_falsy():
    assert entry_env_name({"env_name": "", "prompt_idx": 1}, "math") == "math"
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `cd /root/subnet81/reliquary-miner-priv && PYTHONPATH=. python3 -m pytest tests/test_mix_helpers.py -v`
Expected: FAIL (`ImportError: cannot import name 'pick_bake_env'`)

- [ ] **Step 3 : Implémenter les helpers**

Ajouter à la fin de `reliquary/miner/mix_controller.py` :

```python
def pick_bake_env(target_slots: dict[str, int], pool_counts: dict[str, int]) -> str:
    """Env to bake next: the one furthest below its target share in the pool.

    deficit(env) = target_slots[env] - pool_counts.get(env, 0). The env with
    the largest deficit is picked; strict ``>`` keeps the FIRST env in
    ``target_slots`` order on ties (deterministic). With a single env this
    always returns that env — Phase-1 behaviour is unchanged.
    """
    best: str | None = None
    best_deficit: int | None = None
    for env, target in target_slots.items():
        deficit = target - pool_counts.get(env, 0)
        if best_deficit is None or deficit > best_deficit:
            best, best_deficit = env, deficit
    assert best is not None, "target_slots must be non-empty"
    return best


def entry_env_name(entry: dict, default: str) -> str:
    """Env that baked a pool entry; falls back to ``default`` for legacy
    disk-reloaded entries (pre-multi-env persistence) that lack the key."""
    return entry.get("env_name") or default
```

- [ ] **Step 4 : Lancer, vérifier le succès**

Run: `cd /root/subnet81/reliquary-miner-priv && PYTHONPATH=. python3 -m pytest tests/test_mix_helpers.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5 : Non-régression MixController** — `PYTHONPATH=. python3 -m pytest tests/test_mix_controller.py -v` Expected: PASS. Puis checkpoint de revue.

---

### Task 2 : `env_name` sur le schéma d'entrée du pool + source du `RolloutSubmission` depuis l'entrée (CPU)

**Files:**
- Modify: `reliquary/miner/engine.py` (entrées : 1886-1891, 2175-2180, 2567-2572 ; consume : 2855 ; nouvelle méthode `_entry_env_name`)
- Create: `tests/test_entry_env_name.py`

**Interfaces:**
- Consumes: `entry_env_name` (Task 1).
- Produces: méthode `MiningEngine._entry_env_name(self, entry) -> str` ; clé `"env_name"` présente sur toute entrée fraîchement bakée.

Les 3 sites de construction d'entrée utilisent aujourd'hui un dict à 4 clés (`prompt_idx`, `problem`, `rollouts`, `checkpoint_n`). En Phase 1 le bake passe par `self.env`, donc l'env baker = `self.env.name`. On ajoute la clé ; on lit via le défaulteur (back-compat disque).

- [ ] **Step 1 : Écrire le test** (la méthode délègue au helper pur, testable sans construire un vrai engine via une instance factice)

```python
# tests/test_entry_env_name.py
from reliquary.miner.engine import MiningEngine


def test_engine_entry_env_name_reads_key():
    fake = object.__new__(MiningEngine)        # pas d'__init__ (pas de GPU)
    fake.active_envs = ["openmathinstruct"]
    assert MiningEngine._entry_env_name(fake, {"env_name": "opencodeinstruct"}) \
        == "opencodeinstruct"


def test_engine_entry_env_name_defaults_to_first_active_env():
    fake = object.__new__(MiningEngine)
    fake.active_envs = ["openmathinstruct"]
    assert MiningEngine._entry_env_name(fake, {"prompt_idx": 7}) == "openmathinstruct"
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `cd /root/subnet81/reliquary-miner-priv && PYTHONPATH=. python3 -m pytest tests/test_entry_env_name.py -v`
Expected: FAIL (`AttributeError: ... '_entry_env_name'`)

- [ ] **Step 3a : Ajouter la méthode + l'import**

En haut de `reliquary/miner/engine.py`, à l'import existant de `mix_controller` (ou en ajouter un), s'assurer d'avoir :
```python
from reliquary.miner.mix_controller import entry_env_name as _entry_env_name_fn
```
Ajouter la méthode dans la classe `MiningEngine` (juste après `_resolve_eos_ids`, ~L616) :
```python
    def _entry_env_name(self, entry: dict) -> str:
        """Env that baked ``entry``; defaults to the first active env for
        legacy disk-reloaded entries lacking the key (back-compat)."""
        return _entry_env_name_fn(entry, self.active_envs[0])
```

- [ ] **Step 3b : Taguer les 3 sites de construction d'entrée**

`engine.py:1886-1891` (`_pre_bake_entry`) — ajouter la clé :
```python
        return {
            "prompt_idx": prompt_idx,
            "problem": problem,
            "rollouts": rollouts_cache,
            "checkpoint_n": expected_ckpt_n,
            "env_name": self.env.name,
        }
```
`engine.py:2175-2180` (`_pre_bake_batch`) :
```python
            entries.append({
                "prompt_idx": prompt_idx,
                "problem": problem,
                "rollouts": subset,
                "checkpoint_n": expected_ckpt_n,
                "env_name": self.env.name,
            })
```
`engine.py:2567-2572` (`_process_one_completion`) :
```python
        entry = {
            "prompt_idx": prompt_idx,
            "problem": problem,
            "rollouts": subset,
            "checkpoint_n": expected_ckpt_n,
            "env_name": self.env.name,
        }
```

- [ ] **Step 3c : Sourcer `RolloutSubmission.env_name` depuis l'entrée**

`engine.py:2855` (`_finalize_pool_entry`, qui reçoit `entry`) — remplacer `env_name=self.env.name` par :
```python
                env_name=self._entry_env_name(entry),
```
(Laisser `_build_rollout_submission:1709` tel quel — `env_name=self.env.name` — c'est le chemin legacy single-rollout sans entrée, correct en single-env.)

- [ ] **Step 4 : Lancer, vérifier le succès**

Run: `cd /root/subnet81/reliquary-miner-priv && PYTHONPATH=. python3 -m pytest tests/test_entry_env_name.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5 : Sanity import + non-régression** — `PYTHONPATH=. python3 -c "import reliquary.miner.engine"` (Expected: pas d'erreur) puis `PYTHONPATH=. python3 -m pytest tests/ -q` (Expected: pas de nouveau FAIL). Checkpoint de revue.

---

### Task 3 : Client `/verdicts` porté dans le submitter (CPU)

**Files:**
- Modify: `reliquary/miner/submitter.py` (helper URL + fetch async)
- Create: `tests/test_verdicts_client.py`

**Interfaces:**
- Produces:
  - `build_verdicts_url(url: str, hotkey: str, since: float | None = None) -> str` — `{url}/verdicts/{hotkey}` + `?since=<ts>` optionnel.
  - `async fetch_verdicts(url, hotkey, *, client, since=None) -> VerdictsResponse | None` — GET + parse ; `None` sur erreur réseau/HTTP non-200 (jamais lever).

- [ ] **Step 1 : Écrire le test**

```python
# tests/test_verdicts_client.py
from reliquary.miner.submitter import build_verdicts_url


def test_build_verdicts_url_no_since():
    assert build_verdicts_url("http://v:8080", "5Hdd...") == \
        "http://v:8080/verdicts/5Hdd..."


def test_build_verdicts_url_with_since():
    assert build_verdicts_url("http://v:8080", "5HddX", since=12.5) == \
        "http://v:8080/verdicts/5HddX?since=12.5"


def test_build_verdicts_url_encodes_hotkey():
    assert build_verdicts_url("http://v:8080", "a/b") == \
        "http://v:8080/verdicts/a%2Fb"
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `cd /root/subnet81/reliquary-miner-priv && PYTHONPATH=. python3 -m pytest tests/test_verdicts_client.py -v`
Expected: FAIL (`build_verdicts_url` n'existe pas)

- [ ] **Step 3 : Implémenter le helper + le fetch**

Vérifier l'import en tête de `reliquary/miner/submitter.py` (ajouter si absent) :
```python
from urllib.parse import quote
from reliquary.protocol.submission import VerdictsResponse
```
Ajouter les deux fonctions (près de `build_state_url`) :
```python
def build_verdicts_url(url: str, hotkey: str, since: float | None = None) -> str:
    """`{url}/verdicts/{hotkey}`, with optional `?since=<ts>` incremental cursor.

    Mirrors the validator's GET /verdicts/{hotkey} endpoint (PR #25). The
    hotkey is path-encoded; ss58 addresses are URL-safe but encode defensively.
    """
    base = f"{url}/verdicts/{quote(hotkey, safe='')}"
    return f"{base}?since={since}" if since is not None else base


async def fetch_verdicts(url, hotkey, *, client, since=None):
    """GET the recent verdicts ring for ``hotkey``. Returns a VerdictsResponse,
    or ``None`` on any transport/HTTP/parse failure (caller treats None as
    'no new signal this tick' — never raises, never kills the loop)."""
    try:
        r = await client.get(build_verdicts_url(url, hotkey, since), timeout=5.0)
        if r.status_code != 200:
            return None
        return VerdictsResponse.model_validate(r.json())
    except Exception:
        return None
```

- [ ] **Step 4 : Lancer, vérifier le succès**

Run: `cd /root/subnet81/reliquary-miner-priv && PYTHONPATH=. python3 -m pytest tests/test_verdicts_client.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5 : Non-régression submitter** — `PYTHONPATH=. python3 -m pytest tests/ -k submitter -v` Expected: pas de nouveau FAIL. Checkpoint de revue.

---

### Task 4 : Tracking `merkle_root → env soumis` + `_apply_verdicts` → MixController (CPU)

**Files:**
- Modify: `reliquary/miner/engine.py` (init state dans `mine_window` ~L660 ; enregistrement dans `_submit_entry` ~L1479 ; nouvelle méthode `_apply_verdicts`)
- Create: `tests/test_apply_verdicts.py`

**Interfaces:**
- Consumes: `self._mix.record_outcome(env, rewarded)` (MixController, déjà présent) ; `Verdict.merkle_root`, `Verdict.rewarded` (Plan B Task 1).
- Produces: `self._submitted_env: dict[str, str]` (merkle_root → env_name) ; `MiningEngine._apply_verdicts(self, resp) -> float` (applique les outcomes, retourne le `ts` max vu pour le curseur `since`).

`_submit_entry` connaît à la fois `merkle_root` (retour de `_finalize_pool_entry`, L1477-1479) et l'entrée (→ env via `_entry_env_name`). Async sur la loop → mutation de `self._submitted_env` sans race.

- [ ] **Step 1 : Écrire le test**

```python
# tests/test_apply_verdicts.py
from reliquary.miner.engine import MiningEngine
from reliquary.miner.mix_controller import MixController
from reliquary.protocol.submission import VerdictsResponse


def _engine_with_mix(envs):
    e = object.__new__(MiningEngine)
    e.active_envs = list(envs)
    e._mix = MixController(envs, total_slots=8, slot_floor=1, alpha=1.0)
    e._submitted_env = {}
    return e


def test_apply_verdicts_records_rewarded_outcome_for_mapped_env():
    e = _engine_with_mix(["math", "code"])
    e._submitted_env = {"a" * 64: "code"}
    payload = {"verdicts": [{"merkle_root": "a" * 64, "accepted": True,
               "reason": "accepted", "ts": 9.0, "rewarded": True}]}
    e._apply_verdicts(VerdictsResponse.model_validate(payload))
    slots = e._mix.target_slots()
    assert slots["code"] >= slots["math"]   # code a payé → reçoit ≥


def test_apply_verdicts_ignores_unknown_merkle_root():
    e = _engine_with_mix(["math"])
    payload = {"verdicts": [{"merkle_root": "f" * 64, "accepted": True,
               "reason": "accepted", "ts": 1.0, "rewarded": True}]}
    # merkle inconnu → pas de crash, pas d'outcome
    assert e._apply_verdicts(VerdictsResponse.model_validate(payload)) == 1.0


def test_apply_verdicts_skips_when_rewarded_is_none():
    e = _engine_with_mix(["math"])
    e._submitted_env = {"b" * 64: "math"}
    payload = {"verdicts": [{"merkle_root": "b" * 64, "accepted": False,
               "reason": "grail_fail", "ts": 2.0}]}  # rewarded absent → None
    e._apply_verdicts(VerdictsResponse.model_validate(payload))  # ne lève pas


def test_apply_verdicts_returns_max_ts_for_cursor():
    e = _engine_with_mix(["math"])
    payload = {"verdicts": [
        {"merkle_root": "c" * 64, "accepted": True, "reason": "accepted", "ts": 3.0},
        {"merkle_root": "d" * 64, "accepted": True, "reason": "accepted", "ts": 7.5},
    ]}
    assert e._apply_verdicts(VerdictsResponse.model_validate(payload)) == 7.5
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `cd /root/subnet81/reliquary-miner-priv && PYTHONPATH=. python3 -m pytest tests/test_apply_verdicts.py -v`
Expected: FAIL (`AttributeError: ... '_apply_verdicts'`)

- [ ] **Step 3a : Ajouter la méthode `_apply_verdicts`**

Dans `MiningEngine` (près de `_entry_env_name`) :
```python
    def _apply_verdicts(self, resp) -> float:
        """Feed each verdict's reward outcome into the MixController, mapping
        merkle_root → the env we submitted it under. Returns the max ``ts``
        seen (advances the `since` cursor). Verdicts with rewarded=None or an
        unknown merkle_root are skipped (no signal)."""
        max_ts = 0.0
        for v in resp.verdicts:
            max_ts = max(max_ts, v.ts)
            env = self._submitted_env.get(v.merkle_root)
            if env is None or v.rewarded is None:
                continue
            self._mix.record_outcome(env, bool(v.rewarded))
        return max_ts
```

- [ ] **Step 3b : Initialiser l'état dans `mine_window`**

Dans `mine_window`, à côté de l'init du pool (~après L664, près de `self._pool_max_size`) :
```python
        # merkle_root → env_name we submitted it under (bounded; trimmed in the
        # verdicts loop). Maps each /verdicts outcome back to its env for the
        # MixController yield signal. Single-env in Phase 1 (always math).
        self._submitted_env: dict[str, str] = {}
        # Incremental cursor for GET /verdicts?since=
        self._verdicts_since: float = 0.0
```

- [ ] **Step 3c : Enregistrer le mapping dans `_submit_entry`**

`engine.py`, dans `_submit_entry`, juste après le bloc finalize (après L1485, une fois `merkle_root` obtenu et avant le build de la requête) :
```python
        # Record which env this submission belongs to so the verdicts loop can
        # map its outcome back to the MixController. Async context → no race.
        self._submitted_env[merkle_root] = self._entry_env_name(entry)
```

- [ ] **Step 4 : Lancer, vérifier le succès**

Run: `cd /root/subnet81/reliquary-miner-priv && PYTHONPATH=. python3 -m pytest tests/test_apply_verdicts.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5 : Non-régression** — `PYTHONPATH=. python3 -c "import reliquary.miner.engine"` puis `PYTHONPATH=. python3 -m pytest tests/ -q`. Expected: pas de nouveau FAIL. Checkpoint de revue.

---

### Task 5 : Boucle async `_verdicts_loop` + spawn dans `mine_window` (CPU)

**Files:**
- Modify: `reliquary/miner/engine.py` (nouvelle méthode `_verdicts_loop` ; spawn ~L790-802 ; trim de `_submitted_env`)
- Create: `tests/test_verdicts_loop.py`

**Interfaces:**
- Consumes: `fetch_verdicts` (Task 3), `_apply_verdicts` (Task 4).
- Produces: `async MiningEngine._verdicts_loop(self, url, client)` — poll périodique, jamais fatal.

- [ ] **Step 1 : Écrire le test** (une itération, avec un client + fetch mockés)

```python
# tests/test_verdicts_loop.py
import asyncio
from reliquary.miner.engine import MiningEngine
from reliquary.miner.mix_controller import MixController
from reliquary.protocol.submission import VerdictsResponse


def _engine(envs):
    e = object.__new__(MiningEngine)
    e.active_envs = list(envs)
    e._mix = MixController(envs, total_slots=8, slot_floor=1, alpha=1.0)
    e._submitted_env = {"a" * 64: "code"}
    e._verdicts_since = 0.0

    class _W:  # minimal wallet stub
        class hotkey:
            ss58_address = "5HotkeyStub"
    e.wallet = _W()
    return e


def test_verdicts_loop_one_tick_records_and_advances_cursor(monkeypatch):
    e = _engine(["math", "code"])
    payload = {"verdicts": [{"merkle_root": "a" * 64, "accepted": True,
               "reason": "accepted", "ts": 4.0, "rewarded": True}]}
    resp = VerdictsResponse.model_validate(payload)

    async def fake_fetch(url, hotkey, *, client, since=None):
        return resp

    import reliquary.miner.engine as eng
    monkeypatch.setattr(eng, "fetch_verdicts", fake_fetch, raising=False)

    # _tick_verdicts = corps d'une itération, extrait pour testabilité
    asyncio.run(e._tick_verdicts("http://v", client=None))
    assert e._verdicts_since == 4.0
    assert e._mix.target_slots()["code"] >= e._mix.target_slots()["math"]
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `cd /root/subnet81/reliquary-miner-priv && PYTHONPATH=. python3 -m pytest tests/test_verdicts_loop.py -v`
Expected: FAIL (`AttributeError: ... '_tick_verdicts'`)

- [ ] **Step 3a : Ajouter l'import + `_tick_verdicts` + `_verdicts_loop`**

En tête de `reliquary/miner/engine.py`, à côté des autres imports submitter au niveau module (ou lazy dans la méthode), exposer `fetch_verdicts` au scope module pour le monkeypatch :
```python
from reliquary.miner.submitter import fetch_verdicts
```
Méthodes dans `MiningEngine` :
```python
    async def _tick_verdicts(self, url, *, client) -> None:
        """One verdicts poll: fetch since the cursor, feed the MixController,
        advance the cursor, and trim the merkle→env map. Never raises."""
        hk = self.wallet.hotkey.ss58_address
        resp = await fetch_verdicts(url, hk, client=client, since=self._verdicts_since or None)
        if resp is None or not resp.verdicts:
            return
        new_ts = self._apply_verdicts(resp)
        if new_ts > self._verdicts_since:
            self._verdicts_since = new_ts
        # Bound the map: keep the most recent ~2000 submissions.
        if len(self._submitted_env) > 2000:
            for k in list(self._submitted_env)[:-2000]:
                self._submitted_env.pop(k, None)

    async def _verdicts_loop(self, url, client) -> None:
        """Background poll of GET /verdicts/{hotkey} → MixController yield
        signal. Independent of the latency-critical submit path; failures are
        logged and never kill the loop."""
        while True:
            try:
                await self._tick_verdicts(url, client=client)
                await asyncio.sleep(10.0)
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("verdicts loop iteration failed; continuing")
                await asyncio.sleep(10.0)
```

- [ ] **Step 3b : Spawn de la boucle dans `mine_window`**

`engine.py` ~L790-802, à côté du spawn de `gen_task`, ajouter un task verdicts et l'annuler proprement à la sortie. Remplacer le bloc :
```python
                gen_task = asyncio.create_task(
                    self._generator_loop(url, client, rng),
                )
                await self._trigger_loop(url, client, results)
```
(et son pendant async `_async_generator_loop`) par une version qui lance aussi `verdicts_task` et garde une référence forte. Concrètement, après la création de `gen_task` (les deux branches) et avant `await self._trigger_loop(...)` :
```python
                verdicts_task = asyncio.create_task(
                    self._verdicts_loop(url, client),
                )
                try:
                    await self._trigger_loop(url, client, results)
                finally:
                    verdicts_task.cancel()
```
(Conserver l'annulation existante de `gen_task` telle quelle dans le `finally`/sortie courante.)

- [ ] **Step 4 : Lancer, vérifier le succès**

Run: `cd /root/subnet81/reliquary-miner-priv && PYTHONPATH=. python3 -m pytest tests/test_verdicts_loop.py -v`
Expected: PASS (1 test)

- [ ] **Step 5 : Sanity import + suite complète** — `PYTHONPATH=. python3 -c "import reliquary.miner.engine"` puis `PYTHONPATH=. python3 -m pytest tests/ -q`. Expected: tout vert, pas de nouveau FAIL. **C'est le dernier livrable CPU : à ce stade le signal multi-env est entièrement câblé et testé, prêt à déployer math-only.** Checkpoint de revue.

---

### Task 6 : Routage du bake par-env (mécanique, single-env identique) — **codable sans GPU, validable GPU uniquement**

**Files:**
- Modify: `reliquary/miner/engine.py` (`_generator_loop` 886/910/916 ; `_async_pick_next_prompt` 2331-2355 ; cooldown 1061/1262 ; retry `_retry_rollouts`)

**Interfaces:**
- Consumes: `pick_bake_env` (Task 1), `self._mix.target_slots()`.

**⚠️ Cette tâche n'a AUCUN effet observable en Phase 1** (un seul env → `pick_bake_env` retourne toujours `"openmathinstruct"`, `self.envs[env] is self.env`). Elle prépare Phase 2. Comme elle touche le chemin de génération chaud et n'est **pas** testable sans vLLM/GPU, **la faire sur le GPU** et la valider immédiatement par Task 7. **Ne PAS la livrer avant d'avoir le GPU.**

Transformation (à appliquer site par site, en préservant le comportement single-env) :
- Au début d'une itération du générateur : `env_name = pick_bake_env(self._mix.target_slots(), self._pool_counts_by_env())` ; `env = self.envs[env_name]`. Ajouter un helper `_pool_counts_by_env()` qui compte `entry["env_name"]` dans `self._pool` (défaut via `_entry_env_name`).
- Remplacer chaque `self.env` du chemin de pick/bake par `env` (la variable locale) : `_generator_loop` L886/910/916, `_async_pick_next_prompt` L2336/2346/2354.
- Cooldown : écrire `self._cooldowns[env_name] = set(state.cooldown_prompts)` au lieu de (ou en plus de) `self._cached_cooldown` (1061), et lire `self._cooldowns[env_name]` côté pick. En single-env, `_cooldowns["openmathinstruct"]` == l'ancien `_cached_cooldown`.
- Retry : remplacer `self._retry_rollouts: dict[int, list]` par `self._retry_by_env: dict[str, dict[int, list]]` keyé par env (car `prompt_idx` est par-env). Adapter les lectures/écritures (générateur 878-896/928-949, async 2331/2638/2749) à `self._retry_by_env[env_name]`.

- [ ] **Step 1 : Implémenter la transformation** (sur le GPU, en suivant les sites ci-dessus).
- [ ] **Step 2 : Sanity import** — `PYTHONPATH=. python3 -c "import reliquary.miner.engine"`. Expected: OK.
- [ ] **Step 3 : Suite CPU** — `PYTHONPATH=. python3 -m pytest tests/ -q`. Expected: pas de nouveau FAIL (garde-fou de non-régression structurelle).
- [ ] **Step 4 : La validation comportementale réelle = Task 7.** Checkpoint de revue.

---

### Task 7 : Validation de parité sur GPU (math-only) — **GPU**

**Files:** aucun (déploiement + observation).

Objectif : confirmer **zéro régression** du chemin math avant d'activer le code (Phase 2). Préalable : accès H100 (clé SSH), miner-priv synchronisé.

- [ ] **Step 1 : Sync local → GPU**
```bash
rsync -rcni --exclude='__pycache__' --exclude='*.log' --exclude='*.pyc' \
  --exclude='data/' -e "ssh -i /root/subnet81/.ssh/id_ed25519" \
  /root/subnet81/reliquary-miner-priv/ root@86.38.238.43:/root/reliquary-miner-priv/
```
Attendu : seuls les fichiers modifiés par ce plan listés ; relancer sans `-n` pour appliquer.

- [ ] **Step 2 : Lancer le mineur (math-only forcé)** — `RELIQUARY_ACTIVE_ENVS` non défini (défaut math) :
```bash
cd /root/reliquary-miner-priv && PYTHONPATH=. \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RELIQUARY_BAKE_BATCH_SIZE=6 \
  /root/venv/bin/python -m reliquary.cli.main mine \
    --wallet-name camille81 --hotkey miner1 --network finney --netuid 81 \
    --checkpoint Qwen/Qwen3-4B-Instruct-2507 --log-level INFO \
    2>&1 | tee /root/reliquary-miner-priv/miner.log
```

- [ ] **Step 3 : Vérifier la parité** sur ~10 windows : taux ACCEPTED, selecteds/h, fréquence des rejets (`OUT_OF_ZONE`, `STALE_ROUND`, `GRAIL_FAIL`) **identiques** au baseline pré-plan. Confirmer dans les logs : `_verdicts_loop` poste sans erreur, `_submitted_env` se peuple, `record_outcome` appelé, `target_slots()` = `{openmathinstruct: 8}`.
- [ ] **Step 4 :** Si parité OK → Phase 1 validée. Sinon, `systematic-debugging` sur le diff Task 6. Puis enchaîner Phase 2 (build `data/opencode_cases.json`, activer `RELIQUARY_ACTIVE_ENVS=openmathinstruct,opencodeinstruct`).

---

## Self-Review

**Couverture spec (§ du spec `2026-06-11-miner-multi-env-design.md`) :**
- §5 MixController (record_outcome / target_slots) → **alimenté** par Tasks 4-5 (déjà codé Plan B). ✅
- §6 plomberie par-env (selector/buckets/cooldown/bake par env, tag `env_name`) → Task 2 (tag) + Task 6 (routage). ✅
- §8 port schéma Verdict (`rewarded`) → **prérequis Plan B Task 1** (fait) ; consommé Task 4. ✅
- §5 signal via `/verdicts` → Tasks 3-5 (client + boucle + mapping). ✅
- §10 Phase 1 (mix forcé 100% math, valider parité) → Global Constraints + Task 7. ✅
- §10 Phase 2 (build cases + activer code) → hors-scope, pointé en Task 7 Step 4.

**Scan placeholders :** aucun « TBD/TODO/handle edge cases » ; tout step de code montre le code. Les sites de Task 6 sont décrits par transformation + numéros de ligne exacts (édition mécanique sur GPU, non-TDD car non testable sans modèle).

**Cohérence des types :** `pick_bake_env(target_slots, pool_counts) -> str`, `entry_env_name(entry, default) -> str`, `_entry_env_name(entry) -> str`, `build_verdicts_url(url, hotkey, since) -> str`, `fetch_verdicts(...) -> VerdictsResponse | None`, `_apply_verdicts(resp) -> float`, `_tick_verdicts(url, *, client)`, `_verdicts_loop(url, client)` — utilisés identiquement test↔impl↔appelant.

**Cut CPU/GPU :** Tasks 1-5 = livrables verts sans GPU (le but de la session : « finir ce qu'on peut sans GPU »). Tasks 6-7 = sur le GPU pris juste après.
