# Spec — Gestion multi-env du mineur (opencode curated + sélection par-env + allocation)

**Date :** 2026-06-30
**Cible :** `reliquary-miner-priv` (mineur de production)
**Statut :** design validé en brainstorming, prêt pour plan d'implémentation
**Prérequis :** Plan C Phase 1 (signal multi-env) + Task 6 (routage bake par-env) DÉJÀ codés/testés CPU.

---

## 1. Contexte & problème

Le mineur peut désormais router le bake par-env (Task 6), mais il **ne peut pas réellement
jouer le code** pour deux raisons vérifiées dans le code courant du subnet (`origin/main`
@ `196e275`, clone à jour) :

1. **Dataset opencode désaligné.** Notre env OCI pointe sur `R0mAI/opencodeinstruct-prompts`
   @ `f50bef12…` (prompt-only, ancien). Le validateur courant utilise
   `R0mAI/opencodeinstruct-curated` @ `d3caaefc3b46f8642b251f9efaeccf0d1e95b0a7`
   (commit `c8f3f9c`, #91), avec les `structured_cases` **embarqués** et un prompt-contract
   mis à jour (`c960d3d`). → `prompt_idx` ne mappe plus → `PROMPT_MISMATCH`/mauvais prompts.

2. **Sélection in-zone binaire-only.** `_try_select` (engine.py) bucketise les rollouts par
   `reward == 1.0` / `reward == 0.0`. Le reward code est **continu** (`passed/total ∈ [0,1]`,
   `grader_client.evaluate_cases`), donc un rollout à 0.6 n'entre dans aucun bucket → ignoré
   → **impossible de composer une soumission code**.

**Économie (vérifiée, `service.py:863`) :** `pool_per_env = 1.0/len(env_mix) = 0.5`. Chaque env
distribue 0.5, **brûlé si non rempli**, reward additif par hotkey. → Un mineur math-only est
**plafonné à 50 %**. Jouer le code = capturer l'autre 0.5.

**Zone (vérifiée, `verifier.py:481`, `constants.py:256`) :** `SIGMA_MIN = 0.43` pour les DEUX
envs (`is_in_zone` purement σ-based). σ se calcule sur les **8 rollouts d'un même prompt** :
- Math (binaire) : σ≥0.43 ⟺ **2–6 corrects** sur 8 (8/8 ou 0/8 → σ=0 → rejeté).
- Code (continu) : σ≥0.43 ⟺ **8 fractions `passed/total` assez dispersées** (quasi-bimodal :
  des rollouts proches de 1 ET proches de 0 ; tout groupé → σ bas → rejeté).

## 2. But

Le mineur soumet sur les 2 envs **sans gaspiller de slot** : il pré-filtre l'in-zone en local
(math binaire / code continu) sur un grader **aligné byte-pour-byte sur la valeur** du
validateur courant, et n'envoie que les groupes σ≥0.43 (+ marge).

## 3. Non-objectifs (YAGNI)

- ❌ Prédicteur de difficulté TF-IDF (`difficulty_probe.py`) — étape séparée, ré-évaluée après.
- ❌ Sandbox gVisor/runsc — on exécute **notre propre** code modèle (pas adverse) : subprocess
  isolé (timeout + rlimit) suffit, comme l'actuel `code_grader.py`.
- ❌ Refonte du chemin math (binaire) — **inchangé**.
- ❌ Politique d'allocation « valeur/compétition » (raffinement futur) — on garde l'adaptatif
  par taux + plancher.
- ❌ Validation GPU / activation Phase 2 (Task 7) — hors ce spec.

## 4. Contraintes vérifiées (code courant)

| # | Contrainte | Source |
|---|---|---|
| C1 | Dataset code = `R0mAI/opencodeinstruct-curated` @ `d3caaefc…`, lu via `VirtualParquetDataset(repo, rev, columns=["input","structured_cases"])`. « Both validator and miner load it. » | `reliquary/environment/opencodeinstruct.py:68,115-120` (validateur) |
| C2 | `validator_authoritative_reward = True` → pas de `REWARD_MISMATCH`, **mais le filtre σ s'applique** (notre σ local doit matcher le sien). | idem `:112` |
| C3 | Reward code = `passed/total ∈ [0,1]` continu. | `grader_client.py:evaluate_cases` |
| C4 | **Format d'un `structured_case`** : `{"entry": {"kind":"function","name":<fn>}, "args": [...], "expected": <valeur>}`. Le worker résout l'entry vers une **fonction retournant une valeur**, l'appelle avec `args`, compare la sortie à `expected` **par valeur** (matrix-shape-aware). | `grader/worker.py:319-332`, `test_grader_server.py:38-41`, fixes `e3265df`/`7159542` |
| C5 | `SIGMA_MIN = 0.43` (bootstrap 0.33), `M_ROLLOUTS = 8`, `MAX_SUBMISSIONS_PER_HOTKEY = 8` (= groupes/window). | `constants.py` |
| C6 | `VirtualParquetDataset` est **ABSENT** de miner-priv (base reliquary plus ancienne). | `find` négatif |
| C7 | Prompt-contract code à jour (`c960d3d` « pin the function-call contract »). | log origin/main |

## 5. Architecture — 4 composants

### A. Env opencode → curated
- Repointer `opencodeinstruct.py` (miner-priv) sur `R0mAI/opencodeinstruct-curated` @
  `d3caaefc…` (defaults `_CURATED_REPO`/`_CURATED_REVISION`, overridables
  `RELIQUARY_OCI_REPO`/`RELIQUARY_OCI_REVISION`).
- **Porter `VirtualParquetDataset`** dans miner-priv (ou chargement direct si l'empreinte
  disque est acceptable — décision d'implémentation, par défaut porter le virtual-parquet pour
  ne fetch que les row-groups touchés, comme le validateur).
- Exposer `input` (prompt) + `structured_cases` (cases) par `prompt_idx`. Prompt-contract à
  jour pour que l'encodage canonique matche (anti-`PROMPT_MISMATCH`).
- `_DEFAULT_SHARDS`/alignement : N/A (le curated est un dataset unique ordonné, pas de shards).

### B. Grader local en parité (`passed/total`)
- Remplacer l'interface assertion-string de `code_grader.py` par la **sémantique du worker**
  validateur (C4) : pour chaque case `{entry, args, expected}`, résoudre la fonction d'entry
  dans le code du rollout, l'appeler avec `args`, comparer la sortie à `expected` **par valeur**
  (mêmes règles que le worker — matrix-shape, structures). Reward = `passed/total`.
- Exécution : **un subprocess isolé** par rollout (timeout + rlimit), builtins restreints,
  jamais lever (0.0 sur crash/timeout/no-case). Pas de gVisor (non-objectif §3).
- Objectif de parité : pour un même `(code, cases)`, notre fraction = celle du validateur. Les
  fixes de comparaison (`e3265df` entry→valeur, `7159542` matrix-shape) doivent être répliqués.

### C. Sélection in-zone PAR-ENV (`_try_select` env-aware)
- `_try_select` dispatch selon le type de reward de l'env :
  - **Math (binaire) : inchangé** — buckets `==1.0`/`==0.0`, bande-k k∈[K_MIN,K_MAX]=[3,5],
    contraintes bt_ok existantes.
  - **Code (continu) : nouvelle voie σ-continue** — sur les 8 fractions (en gardant les
    valeurs intermédiaires), composer le sous-ensemble de M_ROLLOUTS qui **maximise la
    dispersion** jusqu'à `std ≥ SIGMA_MIN + marge`, en privilégiant les rollouts `bt_ok` et la
    qualité locale (q10). Si aucun sous-ensemble n'atteint le seuil → retourner en retry
    (multi-phase) ou drop (comme le math).
- Le seuil et la marge sont paramétrés (`RELIQUARY_CODE_SIGMA_MARGIN`, **défaut 0.03** au-dessus
  de 0.43 → cible `std ≥ 0.46`) pour absorber un léger écart grader local↔validateur.
- L'env connaît son type (`binary` vs `continuous`) — flag sur la classe Environment (math
  binaire, opencode continu).

### D. Pré-filtre + allocation
- **Pré-submit :** le bake code grade en local (B) → calcule σ → sélection (C) → ne soumet que
  l'in-zone. Identique au pipeline math. Zéro slot gaspillé en `OUT_OF_ZONE`.
- **Allocation :** **inchangée** — `MixController` adaptatif (rendement réel rewarded/investi
  via `/verdicts`) + plancher ≥1/env. La correction B+C est ce qui permet au code de
  *convertir*, donc au rendement code de devenir non-nul et au MixController de lui allouer des
  slots. Pas de split fixe (un 8/8 code = σ=0 = non soumis → un split forcé gaspillerait).

## 6. Flux code de bout en bout (cible)

```
prompt OCI (curated) ──► 8 rollouts (solutions Python) @ T_PROTO=0.9
                              │  chaque solution exécutée en local contre les MÊMES structured_cases
                              ▼
        8 rewards continus passed/total   ex: [1.0, 0.6, 0.0, 0.9, 0.2, 0.7, 0.1, 0.3]
                              │
                              ▼  sélection σ-continue (C) : sous-ensemble std ≥ 0.43 + marge ?
                  ┌───────────┴───────────┐
                 oui                      non → retry multi-phase / drop (slot non gaspillé)
                  ▼
   soumission (8 rollouts + preuve GRAIL, env_name=opencodeinstruct, tag Task 6)
                  ▼
   validateur ré-exécute les cases (autoritaire) + GRAIL + σ-zone → crédite le pool code (0.5)
```

## 7. Stratégie de tests (CPU, sans GPU)

- **Grader (B) :** code factice + cases `{entry,args,expected}` → fraction correcte ;
  all-pass → 1.0, all-fail → 0.0, partiel → fraction exacte ; crash/timeout → 0.0 ; parité de
  comparaison par valeur (matrix-shape).
- **Sélection σ-continue (C) :** 8 fractions all-pass → rejet ; bimodal → accept ; intermédiaire
  groupé → rejet ; vérifier le sous-ensemble retourné franchit σ≥0.43+marge.
- **Env curated (A) :** chargement (repo/révision), `get_problem(idx)` renvoie `input` +
  `structured_cases`, type continu déclaré.
- **Dispatch `_try_select` :** math binaire inchangé (non-régression) ; code prend la voie continue.

## 8. Phasing

1. **A + B** (env curated + grader parité) — testable CPU avec cases factices ; un test réseau
   optionnel charge réellement le curated.
2. **C** (sélection σ-continue) — pur CPU, TDD.
3. **D** déjà en place (MixController) — juste câbler le type d'env dans le dispatch.
4. Puis (hors-spec) : validation GPU (Task 7) → activer `RELIQUARY_ACTIVE_ENVS=
   openmathinstruct,opencodeinstruct` → audit du reste → ré-évaluer le prédicteur.

## 9. Risques & garde-fous

- **Parité grader imparfaite** → on croit in-zone mais validateur σ<0.43 → `OUT_OF_ZONE`.
  Garde-fou : répliquer fidèlement la sémantique worker (C4) + **marge σ**.
- **Le curated rebump de révision** (upstream) → désalignement `prompt_idx`. Garde-fou : suivre
  `_CURATED_REVISION` dans le check de session ; révision overridable par env var.
- **Math ne doit jamais régresser** → dispatch laisse la voie binaire intacte ; tests de
  non-régression.
- **VirtualParquetDataset à porter** → si le port est lourd, fallback chargement direct du
  curated (coût disque) en v0, virtual-parquet en optimisation.
