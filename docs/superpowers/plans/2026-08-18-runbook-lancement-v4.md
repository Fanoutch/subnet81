# RUNBOOK — Lancement mineur v4 (jour du merge upstream)

> Rédigé le 2026-08-18 après : port complet (branche `feat/port-v4-dapo`,
> réaligné sur upstream `a6456b4`), audit multi-agents du 17/08 (18 items) et
> balayage final du 18/08 (12 gaps G1-G12, tous corrigés côté code).
> **MAJ 18/08 après-midi — LE MERGE A EU LIEU** : `origin/main = 96bfb46`
> (PR #180), zéro diff de contenu vs notre point d'alignement `a6456b4`.
> Validateur live : **encore v3** (`cb23a1e`, status degraded). Annonce
> Discord : rester en v3 jusqu'à la fenêtre d'ACTIVATION qu'ils annonceront ;
> « update to the final release when published » → re-diff
> `git -C /root/subnet81/reliquary diff a6456b4..origin/main` avant tout
> déploiement s'ils repoussent. Les env vars de l'annonce
> (RELIQUARY_PROTOCOL_PROFILE/ENVIRONMENTS) = mineur de référence upstream ;
> notre fork = RELIQUARY_PROTOCOL_VERSION=4 + RELIQUARY_ACTIVE_ENVS (déjà
> dans launch_miner_v4.sh) — rien à changer.
>
> **Phase optionnelle pré-activation — banc débit v4 sur H200 louée ~2 h**
> (mineur v3 prod intouché) : `scripts/deploy_bench_v4.sh <HOST> <PORT>
> setup|gate|bench` (dev box) + `ops/bench_v4_matrix.sh` (P1 coût FS
> full-support = LA grande inconnue ; P2 per-seq→SPRINT_SIZE ; P3 CUDA
> graphs/prefix-caching sur modèle DENSE ; P4 courbe longueur). Dé-risque
> aussi la Phase 2 (gate forced-seed) avant le jour J. EN ATTENTE : go +
> SSH de la box.
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

## ✅ BANC v4 FAIT (18/08, H200 louée 38.255.28.21 — résultats définitifs)

18 mesures (P1→P5 + frontière graphs), gate forced-seed v4 **PASS dans les
2 modes** : eager 0.9572/0.9123, **graphs 0.9674/0.9388** (planchers 0.80/0.70).
- **Forced-seed full-support ≈ 4,7 %** de surcoût (2344 vs 2460 tok/s) — le
  tri top_p a disparu ; chantier kernel MORT.
- **CUDA graphs ×2,19** (5134 vs 2338 à 32 seqs) — LE levier ; parité validée.
- **Prefix caching : OFF** (neutre à négatif, eager comme graphs).
- **Courbe longueur PLATE** 512→8192 (~72 tok/s/seq eager constant) — les
  groupes longs ne coûtent pas plus cher par token.
- **Frontière graphs** (per-seq ↘ quand la couverture ↗) :
  32 seqs = 5134 tok/s (**160/seq**) · 64 = 8469 (132) · 128 = 12525 (98) ·
  256 = 16028 (63). Par-groupe = per-seq×16 → sprint étroit 2-3 prompts
  (2560-2240 tok/s/groupe) pour le RANG, scan à 8 prompts pour la COUVERTURE.
- `VLLM_ATTENTION_BACKEND` n'existe plus (vLLM 0.24) — sans objet.
- Pièges d'install rencontrés (reportés dans les scripts) : CLI HF = `hf
  download` (l'ancienne `huggingface-cli` affiche l'aide) ;
  `VLLM_USE_FLASHINFER_SAMPLER=0` OBLIGATOIRE (JIT ninja tue l'EngineCore).
Config appliquée dans `launch_miner_v4.sh` : graphs ON, prefix OFF,
MAX_NUM_SEQS 256, BAKE_BATCH_SIZE 8, GPU_FRACTION 0.76 (0 OOM au banc).
Logs archivés : `/root/subnet81/data/bench_v4{,_p5}.log`, `bench_supp.log`.

## Phase 2 — Gates GPU (le jour J = simple RE-confirmation, déjà PASS au banc)

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

## Architecture 2 mineurs (décision utilisateur 18/08) — opencode D'ABORD

- **Box 1 = la prod actuelle (31.22.104.180)** : mineur **opencode-only**
  (`RELIQUARY_ACTIVE_ENVS=opencodeinstruct`, défaut du launcher). C'est le
  lancement de la Phase 3 ci-dessous.
- **Box 2 = 2e H200 à louer** : mineur **openmath-only**
  (`RELIQUARY_ACTIVE_ENVS=openmathinstruct` en override). Justifié par
  l'étude offline 18/08 : boxing spontané 91,5 %, payable 94,5 %,
  répétabilité corr 0.881 — le math v4 paie, et code-only brûlerait ~50 %
  des émissions. Déploiement : même recette que la box banc
  (`scripts/deploy_bench_v4.sh setup`/`gate`) + wallet (coldkeypub + hotkey
  SEULEMENT, jamais la coldkey secrète — cf. deploy_h200_2026-08-16.sh:3/6)
  + marqueur `.miner_launcher` + drand testés depuis la box.
- **Sûreté du montage** (vérifié code validateur) : un batcher PAR env,
  compteur hotkey par batcher → quotas 32+32 indépendants sur la même
  hotkey ; espaces de prompts disjoints → aucune collision forced-seed
  entre nos deux box. Chaque box a ses propres dumps (fichiers locaux) ;
  `pull81` à étendre à la box 2 (mêmes 4 JSONL, DEST_DIR séparé, ex.
  `data/box2/`) pour que les corpus math et code restent séparés.

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

## Prior v5 — reconstruire le prédicteur pour le nouveau modèle

**Les modèles v1→v4.5 sont INUTILISABLES** (appris sur le monde v3 : Qwen3.5
thinking, M=8, T=0.6, zone 0.43, manifest OMI complet). **La MÉTHODE est
l'actif** — les leçons payées sur 6 semaines, à réappliquer telles quelles :

| Leçon (source) | Application v5 |
|---|---|
| La recette v4.3 gagne sur le long terme : TF-IDF sur cible d'enchère, duel SANS FUITE, puis juge live (18.1 % vedettes / 77 fenêtres) | Réutiliser `scripts/train_v42_2026-08-14.py` + `retrain_prior_daily.py` tels quels, corpus v4 pur |
| Labels CENSURÉS au mauvais cap = prédicteur qui fuit (v1-v4.1) | Corpus = dump du MINEUR au cap prod 8192 — jamais de probe à un autre cap |
| 7 duels invalidés par la datation (éval sur lignes VUES : top13 7.00 vu / 0.00 jamais-vu) | Chaque ligne du dump porte maintenant `window_n` + `ts` (mesure, pas pull) → éval = fenêtres STRICTEMENT postérieures aux corpus de TOUS les candidats |
| Volume avant retrain : v4.2 (48 positifs) ne bat pas v4.1 ; v4.3 (1 044 payables) écrase | Ne PAS entraîner avant ~**400-500 groupes vedette-grade** (~24-48 h de collecte) ; avant ça, mémo seul |
| Le juge fiable = taux de vedettes LIVE (~40-50 fenêtres pour un verdict), pas le duel offline | Tout candidat passe : duel propre → flip live → mesure 40-50 fenêtres |
| Hybride mémo+modèle : la mémorisation est un ATOUT si on la sépare du modèle | `payable_memo` (store v4 frais, auto-amorcé du dump) couvre le « vu », le TF-IDF couvre l'inédit |
| Vedette lourde/2e prédicteur : redondant, l'ancien numérateur est NUISIBLE (E=95) | Un seul modèle, cible unique — pas de PREDICTOR_2 |

**⚠️ La CIBLE dépend d'un réglage validateur invisible d'avance** :
`DIFFICULTY_AUCTION_FLAT_VALUE` (env validateur, défaut off dans le code).
Diagnostic aux premières fenêtres (verdicts + `canonical_rank`) :
- **Régime plat** (comme le live v3) : valeur 1.0 pour tout robust>0, rang =
  DÉBIT (tokens/rounds) → cible v5 = celle de v4.3 (poids de tokens des
  groupes en zone, `Σlens/(1916+max)`-like recalibrée cap 8192).
- **Régime difficulté** : rang = std·(1-mean)^δ (min robuste si incertains) →
  cible v5 = le champ `score` du dump (déjà écrit par ligne), pénalisé du
  min robuste pour les groupes à non-boxés/tronqués.
Le dump capture TOUT ce qu'il faut pour les deux cibles (rewards, lens,
score, n_truncated) — la collecte n'attend pas le diagnostic.

**Séquence** :
1. H+0 : `SAMPLE_DUMP=/workspace/samples_v4.jsonl` (frais, JAMAIS l'ancien
   fichier v3) + `MEMO_SLOT=1` (store auto-amorcé v4) — déjà dans
   `launch_miner_v4.sh`. Sur la DEV box :
   `tmux new-session -d -s pull81 "bash ops/pull_samples_v4.sh"`.
2. H+2 : diagnostic du régime de classement (ci-dessus) + première photo de
   la distribution des `score` v4 → calibrer `RELIQUARY_MEMO_MIN_SCORE`
   (passer le mémo de « table de payables » — quasi universel en zone 0.24 —
   à « table de vedettes »).
3. ~24-48 h (≥400-500 groupes vedette-grade) : entraîner v5.0 (recette v4.3,
   cible du régime diagnostiqué), duel SANS FUITE (éval window_n > max des
   corpus), flip live si gagnant, verdict à 40-50 fenêtres.
4. Ensuite : cron `retrain_prior_daily.py` (la datation par window_n répare
   son défaut d'origine), churn attendu à re-mesurer (41 % en v3).

## Reste à faire AVANT le jour J (rappel)

1. ~~Réconciliation fixes OOM/hot-swap prod → branche~~ **FAIT** (`a87699e`) —
   re-vérifier seulement qu'aucune modif prod nouvelle n'est apparue depuis.
2. Go utilisateur sur le déploiement (règle projet : pas de restart mineur
   sans accord explicite).
3. (Optionnel) push GitHub de `feat/port-v4-dapo`.
