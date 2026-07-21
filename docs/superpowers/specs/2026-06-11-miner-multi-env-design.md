# Spec — Passage du miner en multi-env (math + code)

**Date :** 2026-06-11
**Cible :** `reliquary-miner-priv` (miner de production, fork single-env)
**Statut :** design validé en brainstorming, prêt pour plan d'implémentation

---

## 1. Contexte & objectif

Le validateur live de finney netuid 81 (`http://86.38.238.30:8080`) tourne en
**dual-env** : `openmathinstruct` ET `opencodeinstruct` sont actifs (HTTP 200
sur `/state?env=` pour les deux, 2026-06-10). Le budget d'émission par window
est **splitté par-env** : `pool_per_env = 1.0 / len(env_mix) = 0.5`, et la part
d'un env non rempli est **brûlée** (non redistribuée).

Notre miner est **single-env (math seul)** → plafonné à ~50% du pool d'émission ;
la tirelire code (0.5) lui est inaccessible.

**Objectif :** faire jouer le miner sur les **2 envs dans la même window**, en
répartissant nos 8 slots dynamiquement vers l'env le plus rentable, mesuré par
nos propres verdicts.

Mesure de contestation (2026-06-10) : code ≈ 78% de l'activité math
(cooldown_len 2560 vs 3278 ; valid_subs 9 vs 16) → 2e tirelire réelle, ni
désertée ni saturée. Gain attendu : expansion ~50% du pool atteignable.

## 2. Non-objectifs (YAGNI)

- ❌ Pas de refonte du pipeline optimisé existant (bake pool, σ-zone,
  oversampling, multi-phase) — on le **préserve** (overlay).
- ❌ Pas de support générique >2 envs (on code pour math+code).
- ❌ Pas de re-tuning du mix par epoch (par-window lissé ; epoch = garde-fou).
- ❌ Pas de grader code-exec en v0 (voir §7).

## 3. Contraintes établies (vérifiées dans le code à jour)

| # | Contrainte | Source |
|---|---|---|
| C1 | **Cas de test reconstructibles** : le validateur retire `structured_cases` de SON miroir, mais ils viennent de la source PUBLIQUE `nvidia/OpenCodeInstruct` (cc-by-4.0) qui expose `id` + `unit_tests`. Le build copie l'`id` verbatim → **jointure miroir↔NVIDIA par `id` = cases exactes** (vérifié 2026-06-11 : 4/4 ids matchés, unit_tests en clair). → **grading code local EXACT possible**. | `opencodeinstruct.py`, `build_opencodeinstruct_subset.py`, vérif HF |
| C2 | **`validator_authoritative_reward = True`** pour le code → le validateur calcule le reward, **ne rejette pas** sur le reward déclaré. → soumission aveugle viable, pas de `REWARD_MISMATCH`. | `opencodeinstruct.py:101` |
| C3 | **Reward code = continu** (`passed/total` ∈ [0,1]). Zone gate = **σ ≥ 0.43** sur les 8 rollouts (binaire pour math, continu pour code). | `grader_client.py`, `constants.py` |
| C4 | **Température FIXE T_PROTO=0.9 imposée par le validateur** (verifier recalcule la distribution à T_PROTO ; checks logprob + token-authenticity). **Interdit d'y toucher.** Diversité des 8 rollouts = aléa naturel du sampling à T_PROTO, pas un réglage. | `verifier.py:424,609` |
| C5 | **Alignement prompts code** garanti par construction : miroir public = mêmes rows/ordre que le subset privé. Pas de shards ; alignement = **révision** `_DEFAULT_PROMPT_REVISION = f50bef12…` (embarquée dans l'env porté). | `build_opencodeinstruct_subset.py`, `opencodeinstruct.py:107` |
| C6 | **Rate-limit global par-hotkey = 8 slots/window**, partagés entre les 2 envs (compteur keyé hotkey seul). | `server.py:819`, `constants.py:328` |
| C7 | **Window partagée** : `window_n`, `randomness`, `drand_round`, checkpoint identiques pour les 2 envs. Submit path (GRAIL/drand) **env-agnostique, inchangé**. | `server.py`, `engine.py` (commit GRAIL ne référence pas l'env) |

## 4. Architecture — 2 couches en overlay

```
COUCHE 1 — MixController (cerveau)
  lit /verdicts → rendement[env] (EMA) → alloue les 8 slots → ratio cible
        │ ratio {math: x, code: 8-x}
COUCHE 2 — Plomberie par-env (overlay)
  selector/buckets/cooldown INSTANCIÉS par env ; bake pool tagué par env
  piloté par le ratio ; submitter tague env_name + /state?env=
        │
COUCHE ENV
  openmathinstruct (existant, fix reward #1 fait)
  opencodeinstruct (À PORTER, mode prompt_only, reward aveugle en v0)
```

Le commit GRAIL, la récup drand et le timing de submit sont **inchangés**
(env-agnostiques, C7). Le risque sur le chemin latency-critical est nul.

## 5. Couche 1 — MixController

Responsabilité unique : allouer les 8 slots/window entre les envs.

**Signal (version simple — taux ; raffinement par valeur = tâche #7) :** via `/verdicts/<hotkey>`, par env :
```
rendement[env] = récompensés[env] / max(1, investis[env])   # EMA sur ~20 windows
```
**Allocation :**
```
slots[env] = round(8 × rendement[env] / Σ rendement)
slots[env] = max(SLOT_FLOOR, slots[env]) ; renormaliser pour Σ = 8
```
**Garde-fous :**
- Cold start : ratio neutre (plancher partout, reste 50/50), converge sur
  quelques windows.
- `SLOT_FLOOR = 1` : toujours ≥1 slot/env (exploration — ne jamais devenir
  aveugle à un env qui se viderait).
- EMA (pas window unique) → pas de yo-yo.

**Interface (CPU-testable, sans GPU) :**
```python
class MixController:
    def observe_verdicts(self, verdicts: list[Verdict]) -> None
    def target_slots(self) -> dict[str, int]   # Σ = 8
```

## 6. Couche 2 — plomberie par-env (overlay)

`engine.py` : `self.env` → `self.envs = {name: Env}` ; `self.selectors`,
`self.buckets`, `self._cached_cooldown` deviennent **des dicts par env**
(car `prompt_idx` est par-env — le 4217 math ≠ 4217 code).

- `selector.py` / `buckets.py` : **inchangés en interne**, juste instanciés
  par env (zéro risque sur le calcul σ-zone).
- **Bake pool** : entrées taguées `env_name` ; le générateur lit le ratio cible
  et bake l'env sous-représenté : `pick_env_per_ratio → pick_prompt_idx(
  selectors[env], cooldown[env]) → générer (T_PROTO=0.9) → reward via
  envs[env].compute_reward`.
- `submitter.py` : tag `env_name=entry.env_name` (existe déjà) + **param `?env=`**
  sur `/state` (port du fix #88, maintenant nécessaire) → cooldown par-env.
- Rejets par-env (`PROMPT_FULL`/cooldown sont par-batcher).
- Respect du budget global 8 slots à travers les 2 envs (C6).

## 7. Couche env — opencode (grader LOCAL EXACT)

**Découverte clé (vérifiée 2026-06-11, voir C1) :** les cas de test ne sont
PAS réellement privés. Ils sont reconstructibles en joignant le miroir public
↔ la source publique `nvidia/OpenCodeInstruct` par `id`. → **grader local
EXACT possible** → l'env code devient un quasi-jumeau de l'env math (sélection
σ-zone complète). Le staging "v0 aveugle / v1 self-consistency" est **obsolète**.

Composants :
1. **Prompts** : porter `opencodeinstruct.py` en `RELIQUARY_OCI_PROMPT_ONLY=1`
   (prompts alignés via révision `f50bef12…`, C5).
2. **Cases (build one-time, sur GPU)** : streamer `nvidia/OpenCodeInstruct`,
   garder les rows dont l'`id` ∈ miroir, parser `unit_tests` →
   `(args, expected)` en réutilisant `parse_unit_tests` / `_call_to_case` du
   build upstream. → artefact compact `id → cases` (~50k entrées). NE PAS
   garder le dataset NVIDIA complet au runtime.
3. **Grader local (sandbox LÉGÈRE)** : exécuter nos complétions contre les
   cases dans un subprocess isolé (timeout + RLIMIT + builtins restreints, à la
   `worker.py`). **gVisor/runsc NON requis** — on exécute le code de NOTRE
   propre modèle, pas un adversaire (le validateur, lui, a besoin de gVisor car
   il exécute du code de miners inconnus).
4. **Reward = taux de cas passés** (continu, C3) → σ-zone (pipeline existant) →
   on ne soumet que les groupes in-zone, exactement comme le math.

Génération : identique au math (8 échantillons à T_PROTO=0.9, C4 — diversité =
aléa naturel). Le reward soumis est notre taux local (le validateur reste
autoritaire, C2, donc même une petite divergence ne déclenche pas de
`REWARD_MISMATCH`, mais viser l'exactitude aligne notre σ sur le sien).

**Risque à surveiller :** branche upstream non-mergée
`feat/per-window-prompt-range` — *"switch env to curated dataset"* (2026-06-11).
Si le validateur change la source du dataset opencode, re-vérifier la jointure
NVIDIA avant de s'y fier.

## 8. Port du schéma Verdict

Notre `Verdict` (protocol/submission.py) n'a pas les champs d'observabilité
ajoutés upstream (`rewarded`, `selected_for_batch`, `accepted_into_pool`, …).
Le MixController en a besoin (signal de rendement). → porter ces champs
optionnels (default None). Sans ce port, parsing strict casse sur les verdicts
enrichis du validateur live.

## 9. Stratégie de tests

**CPU, sans GPU (faisable maintenant) :**
- MixController : allocation, EMA, plancher, cold start (verdicts factices).
- Plomberie : instanciation par-env, routage `env_name`, cooldown par-env.
- Env opencode (prompt_only) : chargement miroir, bonne révision, idx→prompt.
- Schéma Verdict : parsing des champs enrichis.

**GPU (intégration, reporté au déploiement) :** génération vLLM réelle 2 envs,
soumission réelle, bout-en-bout.

## 10. Phasing (ne JAMAIS casser le chemin math)

```
Phase 1 — Refonte plomberie multi-env, mix FORCÉ 100% math
          → valider PARITÉ avec l'actuel (zéro régression) → déployer → confirmer émission math
Phase 2a — Build de l'artefact `id → cases` (join NVIDIA, one-time GPU)
           + grader local (sandbox légère) → testable contre quelques prompts
Phase 2b — Activer env code (prompt_only) + grader local exact + σ-zone,
           plancher ≥1 slot code → MixController alloue selon /verdicts
```

Tout le code Phase 1 + tests CPU se fait **maintenant**, GPU down. Déploiement
gated par : accès GPU (clé SSH changée) + miner actuellement à l'arrêt
(verdicts vides).

## 11. Risques & garde-fous

- **Math ne doit jamais régresser** (revenu) → Phase 1 math-only = filet.
- **Gaspillage de slots code** en v0 → MixController réduit au plancher seul.
- **Révision opencode** : suivre les bumps de `_DEFAULT_PROMPT_REVISION` dans
  les commits upstream (ajouter au check de session).
- **Plancher permanent** = ~1 slot/window perdu si code ne paie jamais →
  option future : exploration probabiliste.

## 12. Raffinements futurs (post-v0)

- **Mix pondéré par la VALEUR du reward** (pas juste le taux récompensés/investis)
  — tâche de suivi #7.
- **Exploration probabiliste** du plancher (§11).
- **Self-consistency** : fallback uniquement si la jointure NVIDIA casse un jour
  (validateur change de dataset) — sinon inutile (grader exact, §7/C1).
- Branche upstream à surveiller : `feat/per-window-prompt-range` (#83) —
  accord validateur/miner sur un seed de range de prompts (potentiellement
  breaking pour la sélection).

## 13. Points hors-scope mais liés (déjà notés CLAUDE.md)

- Shards OMI 2→4 (fix 1 ligne, différé).
- Enum `REWARD_DISTRIBUTION` manquante (cosmétique).
- Blocker SSH GPU + miner down (préalable au déploiement de toute phase).
