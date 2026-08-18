# RUNBOOK — Lancement mineur v4 (jour du merge upstream)

> Rédigé le 2026-08-18 après : port complet (branche `feat/port-v4-dapo`,
> réaligné sur upstream `a6456b4`), audit multi-agents du 17/08 (18 items) et
> balayage final du 18/08 (12 gaps G1-G12, tous corrigés côté code).
> Validateur live au moment de la rédaction : **encore v3** (`cb23a1e`).
> Box GPU : H200 `ssh -p 40300 root@31.22.104.180`, arbres `/workspace/*`.
> ⚠️ Une fois le validateur en v4, il n'y a PAS de retour v3 possible :
> tous les leviers sont « fix forward ».

---

## Phase 0 — Détection de la bascule (ne RIEN déployer avant)

```bash
watch -n 300 'curl -s http://209.20.157.231:8080/health | python3 -m json.tool | head -40'
```

La bascule est CONFIRMÉE quand `/health` montre TOUT ceci :
- `"protocol_version": 4` et `"generation_profile_id": "qwen3-4b-base-dapo-v4"`
- contrat : `Qwen/Qwen3-4B-Base @ 906bfd4b4dc7f14ee4320094d8b41684abff8539`
- `collection_seconds: 150`, `max_new_tokens: 8192` (2 envs), `bft: null` (2 envs)
- sampling `T 1.0 / top_p 1.0 / top_k 0`, `prompt_encoding: "raw"`

Puis `curl -s http://209.20.157.231:8080/state` : noter
`checkpoint_repo_id`/`checkpoint_revision` (si vide au tout début du run,
le fallback pinné `Qwen/Qwen3-4B-Base@906bfd4b…` est déjà câblé, cli/main.py).

⚠️ Si upstream re-pousse des commits sur la branche avant le merge :
re-diff `git -C /root/subnet81/reliquary diff a6456b4..origin/feat/qwen3-base-dapo-v4-profile`
AVANT de dérouler (précédent : le canal grader `Answer:` a été supprimé 24 h
après notre premier port).

## Phase 1 — Préparation (faisable AVANT la bascule, sans risque)

1. ~~Réconcilier l'arbre prod → branche v4~~ **FAIT (18/08, commit `a87699e`)** :
   fixes OOM (`_kill_stale_engine_cores`) + hot-swap borné (reload 10→3 min,
   `reload_weights_inplace`/`request_interrupt`) + watchdog v2.1 +
   payable_memo mergés dans la branche (+ fix `getattr` défensif sur
   `_interrupt`, 9 tests streaming réparés). Le rsync ci-dessous est complet.
   ⚠️ Reste vrai : avant le rsync, re-vérifier `git -C
   /root/subnet81/reliquary-miner-priv status` — si de NOUVELLES modifs prod
   sont apparues depuis le 18/08, re-réconcilier d'abord (même méthode :
   `git diff > patch` puis `git apply --3way` dans le worktree).
2. **Pré-télécharger le modèle** sur la box (~8 Go, indépendant de la bascule) :
   ```bash
   HF_HOME=/workspace/hf huggingface-cli download Qwen/Qwen3-4B-Base \
     --revision 906bfd4b4dc7f14ee4320094d8b41684abff8539
   ```
3. **Miroirs drand DEPUIS la box** (critique : 1 miroir mort = ~15-20 s de gel
   par tirage sur 5, fatal en fenêtre 150 s) :
   `curl --max-time 6 https://<miroir>/public/latest` sur les 5 ; ne mettre QUE
   les répondants dans `RELIQUARY_DRAND_URLS` (posé dans launch_miner_v4.sh).
4. **Rsync** (source = worktree réconcilié, JAMAIS l'arbre prod seul) :
   ```bash
   rsync -rc --delete --exclude='__pycache__' --exclude='*.log' --exclude='data/' \
     --exclude='.pytest_cache' -e "ssh -p 40300" \
     /root/subnet81/.worktrees/miner-priv-port-v4-dapo/ \
     root@31.22.104.180:/workspace/reliquary-miner-priv/
   ```
5. **Déployer la chaîne de relance v4 (G1 — sinon le 1er restart du watchdog
   relance en v3 = 100 % rejets)** :
   ```bash
   scp -P 40300 ops/launch_miner_v4.sh root@31.22.104.180:/workspace/launch_miner_v4.sh
   ssh -p 40300 root@31.22.104.180 'echo /workspace/launch_miner_v4.sh > /workspace/.miner_launcher'
   ```
   (`restart_miner.sh` lit ce marqueur ; sans lui, fallback launch_miner.sh v3.)
6. `verify_deployed.sh` sur la box avec `RELIQUARY_PROTOCOL_VERSION=4` (G8) :
   doit dire `contrat v4 conforme (4 reliquary-forced-seed-v4 15616 8192
   qwen3-4b-base-dapo-v4)`.

## Phase 2 — Gates GPU (obligatoires, APRÈS bascule confirmée, AVANT mining)

1. **Gate forced-seed v4** (full-support T=1.0 ; planchers : groupe ≥ 0.80,
   pire rollout ≥ **0.70**) — défauts déjà v4 sous le flag (G6) :
   ```bash
   cd /workspace/reliquary-miner-priv && source /workspace/venv/bin/activate && \
   export VLLM_USE_DEEP_GEMM=0 CUDA_HOME=/workspace/venv/lib/python3.12/site-packages/nvidia/cu13 && \
   RELIQUARY_PROTOCOL_VERSION=4 PYTHONPATH=. python ops/validate_vllm_forced_seed_group.py
   ```
   Pièges connus : env vLLM du launcher requis ; `max_num_seqs<=64` ;
   `GATE_EAGER=0` à repasser si CUDA graphs en prod. Référence 4B/v3 du
   08-06 : PASS 0.9793 / 0.9231 — attendre le même ordre de grandeur.
   ÉCHEC = ne pas miner ; diagnostiquer (révision modèle ? domaine ?).
2. **Sanity constants** : `bash /workspace/launch_miner_v4.sh` s'arrête tout
   seul si une constante n'est pas v4 (asserts intégrés, dont
   GENERATION_PROFILE_ID et MATH_ANSWER_FORMAT).
3. **Hot-swap et FS_GRAPH restent OFF** (items 14/15 différés : recalibrer
   sous T=1.0 / re-mesurer la VRAM à M=16 avant toute activation).

## Phase 3 — Lancement

```bash
ssh -p 40300 root@31.22.104.180
tmux new-session -d -s miner "bash /workspace/launch_miner_v4.sh 2>&1 | tee /workspace/miner_v4.log"
# puis le watchdog (v2.1, réconcilié en Phase 1) :
tmux new-session -d -s watchdog81 "bash /workspace/watchdog.sh"
```

Le launcher v4 fait foi (env posé + purge v3). **NE PAS reprendre de
`start_miner_h200.sh`** : `K_MIN=2/K_MAX=6`, `MAX_NEW_TOKENS=16384/2600`,
`PROMPT_PREDICTOR(_2)` (priors v3 → laisser le fallback uniforme),
`SPRINT_MAX_WAIT_S=90`, `MIN_LOCAL_Q10=0.05`. **GARDER** `SAMPLE_DUMP`
(labels v4 → prédicteur v5). GPU_FRACTION = 0.76 (post-OOM).

**Test de la chaîne de relance avant de laisser tourner** :
`bash /workspace/reliquary-miner-priv/ops/restart_miner.sh` puis
`grep -m1 "v4 constants OK" /workspace/miner.log` — si absent, le marqueur
`.miner_launcher` n'est pas lu : STOP, corriger avant de partir.

## Phase 4 — Surveillance H+0 → H+2

**Source de vérité = verdicts** (seuls à montrer les rejets différés) :
```bash
curl -s 'http://209.20.157.231:8080/verdicts/5DvpFN3QEa9iimQiA5jQaRmx8dbW2uxonM53j51Cw3kBva7q' | python3 -m json.tool | grep -E '"reason"|"accepted"|rewarded' | sort | uniq -c
```
(Le polling interne du mineur marche depuis G2 ; tout échec de parse est
maintenant loggé `verdicts: fetch/parse failed`.)

**Greps log de premier réflexe** :
```bash
grep -E 'pre_bake\[(termination|drop_truncated|out_of_zone|uncertain|malformed|auction_score)\]|réalisé fenêtre|verdicts:' /workspace/miner_v4.log | tail -40
```

Métriques attendues / seuils d'alerte des premières fenêtres :
- **Boxing spontané math** (`SAMPLE_DUMP` : part des groupes openmathinstruct
  à rewards > 0) — LA nouvelle métrique de revenu : non-boxé = reward 0.
  Si ~0 → le 4B-Base ne boxe pas assez : prioriser l'env code, et surveiller
  les drops `pre_bake[uncertain_out_of_zone]`.
- **Cap-hit 8192** (`completion_lens == 8192` dans le dump) : attendu ~0 au
  ck0 (médiane upstream ~500) ; s'il monte, c'est le « thermostat » upstream —
  ils remonteront le cap.
- **seed self-gate / gate** : ~0.95 groupe ; `WindowTally` : « PAYABLES P » et
  distribution des k (payable = k∈[1,15] ; k=0 dominant math = boxing).
- **Quota** : viser jusqu'à 32 soumissions/fenêtre ; latence prêt→soumis ~5 s.

**Diagnostic par verdict** :
| Verdict | Cause la plus probable | Réflexe |
|---|---|---|
| GENERATION_CONTRACT/PROTOCOL_MISMATCH | flag v4 absent (restart v3 ?) | `grep "v4 constants OK"` + marqueur G1 |
| PROMPT_MISMATCH | chat template ou manifest OMI non filtré | chercher `<\|im_start\|>` dans un dump ; vérifier `OMI_TRAIN_SHARDS_ONLY` |
| SEED_MISMATCH | domaine/révision modèle | re-passer le gate Phase 2 ; comparer `/state` revision |
| REWARD_MISMATCH | contrat boxed | rejouer le grading local sur le rollout depuis SAMPLE_DUMP |
| BAD_TERMINATION | trou de garde (HAUT RISQUE : 32/fenêtre = quarantaine) | vérifier budget truncated (1/3) + MIN_EOS 0.001 |
| OUT_OF_ZONE | miroir uncertain insuffisant | comparer aux logs `pre_bake[uncertain_kept]` |
| RATE_LIMITED | quota 32 atteint — BON signe | rien, surveiller stale_round |
| MALFORMED_FINAL_ANSWER | box coupée au cap | vérifier drops `pre_bake[malformed_final_answer]` locaux |

Archive fenêtre : `https://reliqua.ai/api/r2/window/<N>` ; dashboard
`https://reliqua.ai/dashboard`. Paiement = EMA α≈0.027 (demi-vie ~25 fenêtres).

## Phase 5 — Leviers de correction (env var + `restart_miner.sh`)

- `RELIQUARY_AUCTION_MIN_SCORE` : remonter si le quota part en groupes
  non-payés ; `RELIQUARY_K_MIN/K_MAX` : resserrer la bande 1/15 ;
  `RELIQUARY_ZONE_SIGMA_MIN` : override du 0.24 (diagnostic) ;
  `RELIQUARY_RANKING_BUDGET_S` (12 s) ; `EXPLORE_SLOTS` : ne PAS monter.
- Re-bench sprint dès la première heure calme : `SPRINT_MAX_WAIT_S` ≲ 10-15 s,
  `SPRINT_SIZE` 1-2 (à M=16, SIZE=4 = 64 séquences en vol), `BAKE_BATCH_SIZE`.
- Arrêt propre : Ctrl-C dans tmux — JAMAIS `kill -9` (zombies EngineCore →
  OOM au relaunch ; `_kill_stale_engine_cores` couvre, une fois réconcilié).
- Collecte prédicteur v5 : laisser `SAMPLE_DUMP` tourner ; ré-entraîner à
  ~400-500 payables v4 (les modèles v4.x sont morts avec le monde v3).

## Reste à faire AVANT le jour J (rappel)

1. ~~Réconciliation fixes OOM/hot-swap prod → branche~~ **FAIT** (`a87699e`) —
   re-vérifier seulement qu'aucune modif prod nouvelle n'est apparue depuis.
2. Go utilisateur sur le déploiement (règle projet : pas de restart mineur
   sans accord explicite).
3. (Optionnel) push GitHub de `feat/port-v4-dapo`.
