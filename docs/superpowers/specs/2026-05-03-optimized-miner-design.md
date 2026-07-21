# Optimized private miner for Reliquary subnet 81 — design

**Date :** 2026-05-03
**Statut :** approuvé par l'utilisateur, prêt pour implementation plan
**Cible :** un miner privé qui maximise le score EMA sur subnet 81 via une sélection de prompts intelligente, un backend vLLM, et un pipeline async depth-2

---

## 1. Contexte

Reliquary subnet 81 fait du GRPO décentralisé. Les miners produisent des rollouts (8 completions à T=0.9 sur des prompts MATH), un validateur en sélectionne les 16 premiers valides par window pour former le batch d'entraînement, puis publie le checkpoint mis à jour sur HF. Le score EMA d'un miner est calculé sur 72 windows : `score_new = 0.027 × (slots_won / 8) + 0.973 × score_old`.

**Surface d'optimisation principale (extraite de `docs/mining.md` et `validator/batcher.py`) :**

1. **Sélection de prompts** : l'acceptation au validateur dépend de `σ ≥ 0.43` (entre 2 et 6 succès sur 8). Le sweet spot est `p_succès ∈ [0.25, 0.75]` pour le checkpoint courant. Un `OUT_OF_ZONE` brûle gen + proof + submit.
2. **Latence FIFO** : à `prompt_idx` égal entre deux miners, le `signed_round` le plus petit gagne (`SUPERSEDED` pour l'autre). Réduire le délai poll → submit est directement convertible en slots gagnés.
3. **Filtrage local pré-submit** : les rewards sont calculées localement (`env.compute_reward`). On peut donc calculer σ avant de payer GRAIL+réseau, et discard à coût quasi-nul si σ < 0.43.

## 2. Décisions de design

| # | Décision | Justification |
|---|---|---|
| 2.1 | Fork local privé du repo officiel, branche `priv`, pas de remote `origin` | Le code de référence n'expose pas d'API stratégie pour `pick_prompt_idx` ; un fork est plus simple qu'un monkey-patch via une dep pip |
| 2.2 | Modifs confinées à `reliquary/miner/` (jamais toucher `protocol/`, `validator/`, `environment/`, `infrastructure/grail*`) | Garde le rebase upstream indolore et garantit la conformité protocole par construction |
| 2.3 | Hardware : 2× H100 80 GB | vLLM sur GPU 0 + HF sur GPU 1 sans contention VRAM ; permet le pipeline async véritable |
| 2.4 | Selector Bayésien à 2 niveaux + Thompson sampling | Apprend en ligne sur chaque submission, smoothe les cold-prompts via les buckets, exploration gratuite |
| 2.5 | Filtre local σ pré-submit (clé du design) | Élimine les `OUT_OF_ZONE` côté validateur ; économise GRAIL+réseau ; remplace le shadow mode |
| 2.6 | Pipeline asyncio depth-2, `signed_round` stampé au submit (pas au pick) | Masque T_proof derrière T_gen N+1, sans pénaliser le FIFO |
| 2.7 | Déploiement en 3 phases mesurables (selector seul → +vLLM → +pipeline) | Chaque phase a un critère de succès quantifiable ; rollback si KO |

## 3. Architecture

### 3.1 Structure du fork

```
~/reliquary-miner-priv/
├── .git/
│   └── remotes:
│       └── upstream → reliquadotai/reliquary
├── reliquary/
│   ├── miner/
│   │   ├── engine.py          ← modifié (orchestrateur, single-event-loop)
│   │   ├── selector.py        ← NOUVEAU
│   │   ├── vllm_backend.py    ← NOUVEAU (GPU 0, hot-reload)
│   │   ├── hf_backend.py      ← NOUVEAU (GPU 1, reload sync)
│   │   └── pipeline.py        ← NOUVEAU (asyncio.Queue × 3 stages)
│   └── ...                    ← intouché (protocol, GRAIL, env, validator, infrastructure)
├── tests/
│   └── miner_priv/            ← NOUVEAU
└── scripts/                   ← inchangé
```

### 3.2 Topologie GPU

```
GPU 0 (H100, 80 GB)              GPU 1 (H100, 80 GB)
┌──────────────────────┐         ┌──────────────────────┐
│ vLLM engine          │         │ HF model             │
│  Qwen3-4B  ~8 GB     │         │  Qwen3-4B  ~8 GB     │
│  KV cache  ~60 GB    │         │  proof workspace     │
│  → génération 8 roll │         │  → teacher-forcing   │
│    en continuous     │         │    GRAIL sketches    │
│    batching          │         │                      │
└──────────────────────┘         └──────────────────────┘
        │                                ▲
        │  tokens (8 séquences)          │
        └────────────────────────────────┘
```

### 3.3 Pipeline asyncio (depth-2)

```
state_poller ── current_state, current_round, current_ckpt_n (broadcast)
      │
      ▼
┌─────────────┐    ┌─────────────┐    ┌────────────────────────┐
│ selector    │───▶│ gen_stage   │───▶│ proof_submit_stage      │
│ (CPU/RAM)   │    │ (GPU 0)     │    │ (GPU 1 + filtre σ + net)│
└─────────────┘    └─────────────┘    └────────────────────────┘
      ▲                                         │
      └─── update(prompt_idx, response) ◀───────┘
```

Chaque queue a `maxsize=1` (contrôle de flux). Au plus 2 batches en vol : un en gen, un en proof+submit.

## 4. Le Selector

### 4.1 Fondement quantitatif

Pour `n=8` rollouts binaires, `P(in-zone | p)` :

| `p` | P(in-zone) |
|---|---|
| 0.5 | 0.93 |
| 0.3 ou 0.7 | 0.74 |
| 0.2 ou 0.8 | 0.49 |
| 0.1 ou 0.9 | 0.19 |

L'objectif est d'estimer `p` par prompt sous le checkpoint courant et choisir ceux dans la sweet zone `p ∈ [0.25, 0.75]`.

### 4.2 Modèle

```python
class Selector:
    # Niveau 1 — par-prompt × checkpoint
    prompt_post: dict[(prompt_idx, checkpoint_n), Beta(α, β)]

    # Niveau 2 — par-bucket (type, level) lus depuis le dataset brut
    bucket_post: dict[(bucket_key, checkpoint_n), Beta(α, β)]

    # Carry décayé au changement de checkpoint (1 GRPO step bouge peu p)
    def on_checkpoint_change(self, old_n, new_n, decay=0.5):
        # Niveau 1
        for (idx, n), post in list(self.prompt_post.items()):
            if n == old_n:
                self.prompt_post[(idx, new_n)] = Beta(
                    1 + decay * (post.α - 1),
                    1 + decay * (post.β - 1),
                )
        # Niveau 2 (mêmes buckets, decay identique)
        for (bk, n), post in list(self.bucket_post.items()):
            if n == old_n:
                self.bucket_post[(bk, new_n)] = Beta(
                    1 + decay * (post.α - 1),
                    1 + decay * (post.β - 1),
                )
        # Décroissance exponentielle de competitor_seen (signal périme vite)
        for idx in list(self.competitor_seen):
            self.competitor_seen[idx] = int(self.competitor_seen[idx] * 0.5)

    # Pénalité competitor (anti-SUPERSEDED)
    competitor_seen: dict[prompt_idx, int]  # decayed exponentially per window
```

### 4.3 Algorithme `next(cooldown_set)`

```
1. candidats = {prompt_idx ∈ env, hors cooldown}
2. Pour chaque candidat:
   - Si prompt_post existe (current ckpt) : posterior = prompt_post[idx]
   - Sinon (cold) : posterior = bucket_post[bucket_of(idx)]
3. Thompson sampling: tirer p ~ posterior
4. score(p, idx) = P(in-zone | p, n=8) × exp(-γ × competitor_seen[idx]) avec γ ≈ 0.3
5. retourner argmax candidats du score
```

### 4.4 Update post-soumission

```
accepted (rewards = liste des 8 rewards binaires):
    k = sum(rewards)   # nombre de succès
    prompt_post[idx,n].α += k
    prompt_post[idx,n].β += (8 - k)
    bucket_post[bucket].α += k
    bucket_post[bucket].β += (8 - k)

LOCAL_REJECT (filtre local pré-submit, rewards connus):
    # On a calculé les rewards localement avant le check σ → k connu sans ambiguïté
    # (note : σ seul ne suffit pas car σ = √(k(8−k))/8 est symétrique en k ↔ 8−k)
    k = sum(rewards)
    même update que accepted

OUT_OF_ZONE (rejet validateur, ne devrait quasi pas arriver grâce au filtre local) :
    même update — on a les rewards locaux

SUPERSEDED:
    competitor_seen[idx] += 1   # décayé exponentiellement par window
    posterior pas touché

WRONG_CHECKPOINT, PROMPT_IN_COOLDOWN, GRAIL_FAIL, BAD_SIGNATURE :
    aucun update posterior (pas un signal sur p)
```

### 4.5 Persistance

Au shutdown : `pickle` les posteriors + `competitor_seen` dans `~/.cache/reliquary-priv/selector.pkl`.
Au boot :
1. Recharge le pickle (les clés sont versionnées par `checkpoint_n`).
2. Lit `/state` pour connaître le `checkpoint_n` courant.
3. Si on a des posteriors pour un `checkpoint_n` antérieur, applique `on_checkpoint_change(stored_n, current_n)` pour les décayer dans le checkpoint courant.
4. Sinon (premier démarrage), tous les posteriors restent à Beta(1, 1).

## 5. Backends

### 5.1 vLLM (GPU 0)

- **Version** : `vllm==0.6.5` (à pinner après mesure de stabilité)
- **Config** : `gpu_memory_utilization=0.85`, `dtype=bfloat16`, `attn_implementation=flash_attention_2`
- **API utilisée** : `LLMEngine` async, `add_request()` pour batch=8 en parallèle
- **Hot-reload** : tentative `update_weights_from_disk` ; fallback restart engine (10-30 s) si hot-reload non supporté ou échoue

### 5.2 HF teacher-forcing (GPU 1)

- **Modèle** : `AutoModelForCausalLM.from_pretrained(..., torch_dtype=bfloat16, attn_implementation=flash_attention_2)`
- **Rôle** : forward pass déterministe sur les tokens générés par vLLM, pour produire les hidden states que GRAIL hash
- **Reload** : synchrone, ~10 s, déclenché par `state_poller` quand `checkpoint_n` avance
- **Note** : la conformité bit-identique avec le validateur exige cette étape ; vLLM ne peut pas s'y substituer (kernels différents)

## 6. Pipeline async

### 6.1 Stages

```python
async def state_poller():
    while True:
        state = await get_state()
        if state.checkpoint_n > current_ckpt_n:
            window_event.clear()                 # pause pipeline
            await asyncio.sleep(0.2)             # drain in-flight
            await asyncio.gather(
                hf.reload(state.checkpoint_revision),
                vllm.reload(state.checkpoint_revision),
            )
            current_ckpt_n = state.checkpoint_n
            selector.on_checkpoint_change(state.checkpoint_n)
            window_event.set()
        current_state = state
        await asyncio.sleep(POLL_INTERVAL_SECONDS)

async def selector_stage(queue_in, queue_back):
    while True:
        # Drain feedback non-bloquant
        while not queue_back.empty():
            prompt_idx, resp = queue_back.get_nowait()
            selector.update(prompt_idx, resp)
        next_idx = selector.next(current_state.cooldown_prompts)
        await queue_in.put((next_idx, current_state.checkpoint_n))

async def gen_stage(queue_in, queue_mid):
    while True:
        prompt_idx, ckpt_at_pick = await queue_in.get()
        if ckpt_at_pick != current_ckpt_n:
            continue   # stale, drop
        problem = env.get_problem(prompt_idx)
        completions = await vllm.generate(problem, n=8, T=0.9)
        await queue_mid.put((prompt_idx, completions, ckpt_at_pick))

async def proof_submit_stage(queue_mid, queue_back):
    while True:
        prompt_idx, completions, ckpt = await queue_mid.get()
        if ckpt != current_ckpt_n or not current_state.OPEN:
            continue

        problem = env.get_problem(prompt_idx)
        rollouts_data = build_rollout_data(completions, problem)
        rewards = [r.reward for r in rollouts_data]

        # FILTRE LOCAL — clé du design
        sigma = population_std(rewards)
        threshold = 0.33 if in_bootstrap else 0.43
        if sigma < threshold:
            # Note : rewards transmis pour update sans ambiguïté (cf §4.4)
            await queue_back.put((prompt_idx, LocalReject(rewards=rewards)))
            continue   # PAS de GRAIL, PAS de réseau

        sketched = build_grail_sketches(rollouts_data)   # GPU 1
        signed_round = current_state.current_round       # stamp tardif
        request = BatchSubmissionRequest(
            prompt_idx=prompt_idx,
            signed_round=signed_round,
            checkpoint_hash=current_state.checkpoint_revision,
            rollouts=sketched,
            ...
        )
        resp = await submitter.submit(request)
        await queue_back.put((prompt_idx, resp))
```

### 6.2 Gestion des erreurs

| Stage | Erreur | Action |
|---|---|---|
| state_poller | 503, network | sleep + retry, current_state reste last-good |
| gen | OOM, kernel fail | log, drop batch, ne pas crash le pipeline |
| gen | vLLM engine dead | restart engine, miner skip 1-2 windows |
| proof | HF forward fail | log, drop batch |
| submit | timeout réseau | retry x2 backoff 0.5 s, puis drop |
| submit | OUT_OF_ZONE | feedback selector, continue (devrait être rare grâce au filtre local) |
| submit | SUPERSEDED | feedback selector (competitor_seen++), continue |
| submit | WRONG_CHECKPOINT | force re-poll, drop in-flight, continue |

## 7. Tests

### 7.1 Pyramide

```
                   E2E (intégration vs validateur local)
                   │
              Pipeline async (fake backends)
              │
         Unit (selector, posteriors, score function)
         │
   Property-based (Hypothesis)
```

### 7.2 Unit — `tests/miner_priv/test_selector.py`

Cas obligatoires :
- Cold start : tous Beta(1,1) → next() retourne un prompt valide hors cooldown.
- Update accepted (k=4) : posterior bouge, bucket idem.
- Update LOCAL_REJECT : posterior s'update à partir des rewards bruts (pas de σ ambigu).
- Update SUPERSEDED : competitor_seen++, posterior intouché.
- Reset checkpoint avec decay=0.5 : Beta(10, 10) → Beta(5.5, 5.5).
- Cooldown filter : prompt en cooldown jamais retourné.
- Thompson distribution : sur 10 000 tirages, fréquence conforme au modèle (test stat avec tolérance).
- `score(p)` analytique : valeurs tabulées vs formule binomiale.

### 7.3 Pipeline async — `tests/miner_priv/test_pipeline.py`

Avec fake backends :
- Pipeline progresse à depth 2 (gauge `pipeline_in_flight` ∈ {1, 2}).
- Stale checkpoint drop : reload mid-pipeline, in-flight droppés silencieusement.
- LOCAL_REJECT feedback : selector reçoit l'update, GRAIL pas appelé.
- SUPERSEDED feedback : competitor_seen incrémenté.
- Submit retry : timeout 1× puis success → batch passe.
- Window CLOSED → OPEN : pipeline pause puis reprend.

### 7.4 Integration — `tests/miner_priv/test_integration.py`

Validateur local (mock chain) + miner privé sur `127.0.0.1:8888`. ~5 windows. Vérifications :
- Submissions arrivent avec bons `checkpoint_hash`, `prompt_idx`, `signed_round`.
- Selector apprend (posteriors évoluent).
- RSS stable sur 100 windows.
- Reload de checkpoint : force-publish sur le validator de test, miner reload sans crash.

### 7.5 Property-based (Hypothesis)

- ∀ historique d'updates, `next()` retourne un prompt valide hors cooldown (ou raise si env-fully-cooldowned).
- ∀ posterior, `0 ≤ score(p) ≤ 1`.
- Update commutatif sur batches indépendants.
- Reset(checkpoint_n) idempotent.

## 8. Déploiement progressif

### Phase 0 — Setup + baseline (½ journée)

```bash
git clone https://github.com/reliquadotai/reliquary.git ~/reliquary-miner-priv
cd ~/reliquary-miner-priv
git remote rename origin upstream
git checkout -b priv
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .
pip install vllm==0.6.5
reliquary mine --network finney --netuid 81 ... 2>&1 | tee baseline.log
```

Calculer sur baseline.log :
- `accepted_rate_baseline`
- `out_of_zone_rate_baseline`
- Distribution `signed_round` à l'acceptation

### Phase 1 — Selector + filtre local pré-submit (1-2 jours)

Code : `selector.py`, modif `engine.py`. Configuration HF : **un seul modèle HF chargé sur GPU 0** (utilisé séquentiellement pour génération puis proof). GPU 1 reste idle pendant cette phase, il sera réveillé en phase 2 quand vLLM arrivera. Pas de vLLM, pas de pipeline async — la loop reste celle d'upstream avec deux greffons logiques (selector pour le pick, filtre σ avant le submit).

Critères de succès :
- `out_of_zone_rate_validator` → ~0
- `local_discard_rate` < 30 %
- `accepted_rate / submissions_attempted` > +20 % vs baseline
- EMA score on-chain monte (visible après ~50 windows)

### Phase 2 — vLLM backend (2-3 jours)

Code : `vllm_backend.py`, `hf_backend.py`, modif `engine.py`. Toujours séquentiel gen → proof → submit.

Critères de succès :
- `time_gen_seconds` p50 ÷ 2-3 vs phase 1
- `time_total_seconds` p50 -30-40 %
- Distribution `signed_round_at_accept` décalée vers rounds plus petits

### Phase 3 — Pipeline async depth-2 (1-2 jours)

Code : `pipeline.py`, modif `engine.py` (orchestrateur).

Critères de succès :
- `pipeline_in_flight` oscille 1-2
- `time_total_seconds` p50 -20-30 % vs phase 2
- 0 submission avec mauvais `checkpoint_hash`

## 9. Métriques (toutes phases)

Logger structuré JSON sur stdout :

```json
{"ts": "...", "event": "submit_attempt", "prompt_idx": 7321,
 "sigma_local": 0.484, "passed_local_filter": true,
 "outcome": "accepted", "time_gen_ms": 4200, "time_proof_ms": 1800,
 "time_submit_ms": 320, "signed_round": 12345678,
 "checkpoint_n": 42, "selector_p_estimate": 0.51}
```

Métriques clés à surveiller en continu :
- `submissions_total{outcome=accepted|local_reject|superseded|out_of_zone|other}`
- `time_gen_ms`, `time_proof_ms`, `time_submit_ms`, `time_total_ms` (histogrammes)
- `signed_round_at_accept` (histogramme)
- `pipeline_in_flight` (gauge)
- `selector_posterior_entropy` (gauge — si ↘ le selector converge)
- `out_of_zone_rate_validator` — DOIT rester < 5 %, sinon coupure et debug

## 10. Hygiène opérationnelle

- **Persistance posteriors** : pickle toutes les 10 windows.
- **Process supervision** : `systemd` user unit ou tmux + restart wrapper.
- **Upstream rebase** : `git fetch upstream` chaque vendredi. Rebase obligatoire si diff touche `protocol/`, `validator/`, `infrastructure/grail*`, ou `environment/math.py`.
- **Hotkey safety** : alarme si `out_of_zone_rate_validator` > 5 % → coupure et debug.

## 11. Hors scope (v2 ou plus tard)

- Speculative multi-prompts : générer pour plusieurs candidats en parallèle, ne soumettre que celui dont σ passe localement. Coût : rollouts jetés. Reportable une fois v1 stable.
- Proxy small-LM pour pré-screen σ avant la génération full Qwen3-4B.
- Online learning d'embeddings de prompts pour améliorer le bucket smoothing.
- Multi-validateur : adapter quand v2.2 du subnet ship le consensus multi-validator.
