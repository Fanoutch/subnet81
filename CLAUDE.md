# Subnet 81 (Reliquary) — custom miner project

Bittensor GRPO RL training subnet, netuid 81 mainnet (finney).
Wallet `camille81-v2` / hotkey `hotkey81` **ENREGISTRÉE** (uid 167, SS58
`5DvpFN3QEa9iimQiA5jQaRmx8dbW2uxonM53j51Cw3kBva7q`, keyfile régénéré 08-11).

**⚠️ Les affirmations de ce fichier sont des hypothèses** : vérifier contre le
code avant d'asserter un bug/gap (cf. mémoire feedback_verify_code_not_claudemd).

## 🔒 POINT DE REPLI — la version qui tourne (21/08 16h)

**Tag `prod-avant-rollouts-courts` = commit `9bff082`**, poussé sur
`github.com/Fanoutch/subnet81`, branche `feat/port-v4-dapo`.
**Vérifié md5-identique à la prod** : `engine.py`, `vllm_backend.py`,
`prompt_predictor.py`, `submitter.py`, `launch_miner_v4.sh`. Pour revenir :
`git checkout prod-avant-rollouts-courts`, rsync vers la box, redémarrer.

Mineur démarré le 21/08 à 08:43, sans incident (0 Traceback, 0 OOM sur
32 000 lignes de journal). Réglages en vol : `MAX_INFLIGHT_FIRES=3`,
`SHORT_RISK_LAMBDA=0.08`, `COOLDOWN_POLL_S=20`, `EXPLORE_SLOTS=2`,
`BAKE_BATCH_SIZE=8`, `SPRINT_SIZE=8`, `predictor_v50.json`.
⚠️ `volume_v1.json` ABSENT de la box → **bonus de volume jamais actif**,
malgré `VOLUME_MU=0.05` exporté.

## ⚙️ MÉCANIQUE v4 — CORRIGÉE EN SOURCE LE 21/08 (4 erreurs de longue date)

Vérifié sur `80c112f` et `/health` live. Détails : mémoire
`reference_mecanique_v4_corrigee`.

1. **`B_BATCH = 16` par ENVIRONNEMENT** (`constants.py:586`), pas 32. Les
   « 32 » sont notre quota (`MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW`) et
   `2×B_BATCH` de tentatives de preuve. La part d'émission est quantifiée par
   pas de 3,125 % = 1/32 car le validateur est dual-env (16 × 2).
2. **`batch_filled` vient de `MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW = 64`**
   (`constants.py:354`), consommé par TOUT LE MARCHÉ en 25-35 s. Pas d'un
   batch de 16 prompts plein.
3. **Pas de seal anticipé sous l'enchère** (`batcher.py:2316`) : collecte
   jusqu'à la deadline fixe.
4. **`canonical_rank` est POSITIONNEL** (1..N) donc **relatif au marché**.
   Seuils stables : rang 0-15 → 93-100 % payé | 15-25 → 26-45 % |
   **≥25 → 0 % sur 288 entrées**.

**Le rang reste `min(Σ completion_lens, 8192×16) // (rounds × 50)`.** Le modèle
du validateur raccourcit (−11 % en 40 h) mais **pour tout le monde**, donc
c'est relativement neutre : l'effet est de pousser tout le peloton vers le
seuil. À 3 rounds il faut ~3 750 tokens pour être payable, à 4 rounds il en
faut 5 000. **Gagner un round vaut plus de 1 000 tokens.**

## 🎯 LE SEUL FIX EN ATTENTE — retirer le gate anti-rollout-court

PR #188 (mergée 21/08 12:22, **live**) vérifie désormais les complétions
<32 tokens à couverture complète au lieu de les rejeter d'office. Le
commentaire du commit dit exactement ce que notre forensique du 20/08 disait :
« 100 % de ses rejets, aucun n'était un vrai désaccord ».

Notre gate `RELIQUARY_MIN_ROLLOUT_LEN=32` (`engine.py`, `_pre_bake_entry`)
jette encore **17,3 % de la production** (1 253 groupes sur 7 229) pour une
règle qui n'existe plus. Ces groupes sont **92 % `in_zone`** (contre 85 %) et
**43 % dépassent le seuil de bucket payant**. Gain estimé **+0,4 payée/fenêtre**
au plafond, concurrence figée.

**Déploiement** : ajouter `export RELIQUARY_MIN_ROLLOUT_LEN=0` au launcher (il
n'y figure PAS, il vaut 32 par le défaut du code) + retirer
`HF_XET_HIGH_PERFORMANCE=1` (inerte) + redémarrer. Repli : remettre 32.
⚠️ **Risque** : un échec de vérification retombe au stage `logprob`, **qui est
un stage à dette** — 2 échecs et le reste de la fenêtre est refusé. La vigie
`scripts/watch_rollouts_courts.py` compte les `logprob_mismatch` et alerte ;
verdict en 4-8 min, coût d'erreur 1-2 fenêtres.
✅ **Le malus `SHORT_RISK_LAMBDA=0.08` NE bouge PAS** : mesuré inerte sur la
sélection (courts dans le top-3 : 8 % à λ=0, 8 % à 0,08, 9 % à 0,16).

## ⛔ MESURÉ ET REJETÉ LE 21/08 — ne pas re-explorer

- **Bonus de volume** : +2,4 % de volume à μ=0,01, +14,6 % à 0,05 — il en faut
  **+32 %** pour passer du bucket 22 au bucket 29. Insuffisant à tout réglage.
- **Slots d'exploration 2→1** : +6,3 % de volume, **aucun effet observable sur
  le paiement** (75 % / 73 % / 83 % de fenêtres payantes selon 0, 1 ou 2
  explore dans le top-3).
- **Traîne** : effet réel (uniforme devant dans 71 % des paires appariées) mais
  **aucun levier** — réordonner la file est un no-op (le plafond de 32 ne mord
  que dans 1,4 % des fenêtres), prédire la traîne donne Spearman +0,142, et le
  tri canonique `sha256(prompt_idx)` donne −0,010.
- **`RELIQUARY_SPEC_PROOF`** : parallélise la preuve avec le grading, or le
  grading coûte **0,06 s**. Rien à gagner. (Le chemin de preuve fusé, lui, est
  déjà actif par défaut.)
- **Queue du POST** : 1,4 fenêtre perdue sur toute la période. Négligeable.
- **Le gate ne coûte PAS de rang** : testé, corrélations −0,09 entre part de
  courts jetés et arrivée/rang. Il coûte du **nombre d'entrées**, pas des places.

## 📉 CE QUI A CHANGÉ AUTOUR DE NOUS (4 agents, 21/08)

- **Cadence des fenêtres : +30 %** depuis 08h (cycle 273-303 s → 186-225 s).
  ⚠️ Tout ce qui se compte « PAR FENÊTRE » baisse donc mécaniquement. **Compter
  PAR HEURE.** Les payées/heure ne montrent aucune tendance.
- **Durcissement de vitesse réel** : falaise d'admission 13,0 s → 10,5 s, lag
  marché p50 −6 s sur 8 heures consécutives hors amplitude. Le peloton **ne
  s'épaissit pas** (49 mineurs avant/après) — les mêmes vont plus vite.
  Commence le **20/08 vers 18h**, donc **AVANT PR #188** : ne pas lui attribuer.
- Notre rang se dégrade de **+0,26 à +0,45 position/heure** à volume ET arrivée
  contrôlés (t = 2,7 à 5,2).
- **Notre pipeline est intact** : génération plate, rejets internes plats
  (40-46 %), file d'envoi **améliorée** (attente 8-15 s la nuit → 0,5-0,8 s).

## 🪤 PIÈGES DE MESURE — chacun a produit une conclusion fausse le 21/08

1. **DÉDUPLIQUER les verdicts par `merkle_root`** — le dump ré-écrit la même
   ligne à chaque poll (1 524 lignes pour 743 merkle). Les vieilles fenêtres
   accumulent 2-4 copies, les récentes une seule : compter les lignes fabrique
   une chute inexistante.
2. **Une fenêtre n'est mûre que si TOUTES ses entrées ont `rewarded != None`**
   — pas une seule. Les verdicts mettent jusqu'à **28 min** à se décider ;
   compter les indécis comme non payés fabrique une baisse.
3. **Un trou chez nous n'est une perte que si le validateur était ouvert.**
   Toujours croiser avec `windows_v4.jsonl` ou le marché.
4. **Un indicateur qui bouge une fois n'est pas une tendance.** Exiger
   plusieurs tranches consécutives hors amplitude.
5. **`flip_offset_s` est NOTRE référentiel.** Round = `int(off // 3) + 1`,
   exact à 84 % contre `arrival_drand_round`. Bon pour l'agrégat, pas pour une
   entrée isolée.
6. **Âge du moteur** — ne jamais comparer une période post-redémarrage à une
   période mûre.

## 🔥 2026-08-20 21h — BOX RECONSTRUITE : nouveau port, sauvegardes en place

**La H200 a redémarré vers 21h20 et `/workspace` était ENTIÈREMENT vide** (venv,
modèle, wallet, code, corpus, scripts). **Nouveau port SSH : `ssh -p 20098
root@38.255.28.21`** — l'ancien 20089 est mort. Le port change à CHAQUE reboot.

Reconstruction complète en ~40 min, mineur relancé à 21h52, identique à
l'avant-coupure (vérifié par 3 agents : md5 du code, du prior v5.1, du malus).
**Procédure réutilisable : `ops/RECONSTRUCTION_BOX.md`** (9 étapes + 6 pièges).

**Trois protections posées ce soir — la leçon de la panne** :
1. `data/box_backup_2026-08-20_2200/` **poussé sur GitHub** (8,4 Mo) : prior,
   malus, **corpus mémo** (29 924 groupes = plusieurs jours de minage,
   irremplaçable), scripts, gel des 212 paquets. ⛔ jamais le wallet.
2. `scripts/backup_miner_samples.sh` réécrit → **cron */30**, dossiers DATÉS,
   refuse toute sauvegarde plus petite que la précédente, rétention 14 j. (Il
   visait deux box mortes depuis des semaines et ne sauvegardait plus rien.)
3. `ops/pull_samples_v4.sh` : **garde anti-écrasement**. `rsync --append-verify`
   suppose que le distant grandit ; sur une box neuve il ÉCRASE le local — c'est
   arrivé à 21h45, `submits_v4.jsonl` réduit à 10 lignes (données de timing du
   20/08 perdues).

**Pièges vérifiés, à ne pas re-découvrir** : le script d'install archivé visait
`Qwen3.5-4B` (lire le modèle sur `/health`, pas dans un script) ; `.miner_launcher`
doit contenir un CHEMIN, pas « v4 » ; **oublier `samples_v4.jsonl` = mémoire des
vedettes vide** (92 payables au lieu de 21 000, perte INVISIBLE dans les logs) ;
`ops/bench_env.sh` force un vieux checkpoint 2B ; miroir drand `secureweb3` mort ;
un `pkill -f <motif>` en ligne SSH tue son propre shell si le motif y figure.

⚠️ **Bug trouvé : `scripts/retrain_prior_nightly.sh` ne promeut RIEN** — il
cherche `predictor_v50_*` alors que l'entraîneur écrit `predictor_v5.3_*`. v5.2 et
v5.3 ont gagné leurs duels (0,427 vs 0,378) et sont restées sur la dev box.

## ⚡ 2026-08-20 — ÉTAT DU MINEUR (à lire en premier)

**Mécanique de paiement v4 (établie, 373/373)** : `selected == rewarded`, et
tout se joue à l'ARRIVÉE. Taux de sélection par bande : **0-9 s → 55 %**,
10-19 s → 15 %, 20-29 s → 6 %, ≥30 s → 0 %. Payées/h ≈ fenêtres/h × entrées
postées en <10 s × 0,55. Le seal est une deadline fixe (100 s de collecte)
MAIS le batch (16 prompts distincts) se remplit en 8-25 s selon l'affluence.
Validateur : image `a083e5d` (pipelined windows #185 + perfs #186/#187),
cycle open→open ~215-390 s, ~45 % de ses requêtes meurent en 502 par vagues.

**Causes racines résolues aujourd'hui (ne pas les re-chercher)** :
1. **Fuite de sous-processus de grading** = LA cause de la « mort lente » (et
   des 7 h à zéro de la nuit 19→20). `subprocess.run`→`Popen` (fix fork du
   19/08) avait supprimé le kill automatique au timeout → 1,23 zombie/min à
   100 % CPU, quota conteneur = **24 CPU** (pas 192 !) → vLLM affamé à 0,01
   CPU. Fix : `finally: proc.kill()` dans `grade_structured_cases` +
   `setitimer` dans le driver. Vérifié : `gen1` reste à ~3 s après 3 h (avant :
   3 s → 8 s en 45 min).
2. **`CHALLENGE_K = 32`** : le validateur rejette (`inf`) tout groupe dont un
   rollout fait <32 tokens — 100 % des `logprob_mismatch` (67/67), **0 payé
   sur 333**. Fix : gate `min(completion_lens) < 32` (env
   `RELIQUARY_MIN_ROLLOUT_LEN`) + **malus de tri anti-court** (modèle
   `data/risk_short_v1.json`, AUC 0,679, `SHORT_RISK_LAMBDA=0.08`) : taux de
   courts 32 % → 24 %. ⛔ FA2 inutile (0/811 rejets attribuables à sdpa↔FA2).
3. **Dette de preuve** (`MAX_EXPENSIVE_PROOF_FAILURES_PER_HOTKEY_PER_WINDOW=2`,
   remise à zéro CHAQUE fenêtre) : 2 échecs et le reste de la fenêtre est
   refusé. Aujourd'hui elle n'est plus déclenchée que par `worker_dropped`
   (stage `code_grader` = LEUR grader qui plante) : 72 cas, 3 % des fenêtres,
   qui rapportent alors 0.

**Résultat mesuré (21 fenêtres après le restart de 13:55)** : présence 90 %,
4,0 acceptées/fen, **1,57 payée/fen**, 71 % de fenêtres payantes, 39 % de
conversion. Comparaison : 0,03 payée/fen pendant la nuit, 0 entre 12:27-12:53.

**Chantier PRÊT, NON DÉPLOYÉ (soir du 20/08)** — 4 commits sur la branche
`feat/port-v4-dapo` du worktree, en attente du go utilisateur. Déploiement en
UN restart : `bash ops/deploy_latence_volume.sh` (refuse de partir si /health
du validateur n'est pas 200, sauvegarde avant copie, `--dry-run` disponible).
1. **⚠️ `ba3358b` poussé sur GitHub ne contenait AUCUN code** (index vide,
   seul `risk_short_v1.json`). Réparé par `545f4e1`, vérifié md5-identique à la
   prod = vrai point de rollback. **Rien n'est encore poussé.**
2. **Le VOLUME de tokens EST le rang** (mémoire volume_tokens_est_le_rang) :
   7 % de payées sous 3 000 tok contre 54 % au-dessus de 6 000, et c'est une
   propriété du prompt (82 % de la variance). Modèle `data/volume_v1.json`,
   Spearman +0,63 hors échantillon ; `RELIQUARY_VOLUME_MU=0,05` fait passer la
   part de groupes ≥6 000 tok de 31 % à 65 % **sans perdre un groupe payable**.
   Contrepartie : +1 000 tok = +0,86 s d'arrivée, absorbée car `rounds` est
   quantifié par pas de 3 s.
3. **File d'envoi sérialisée** (mémoire file_envoi_serialisee) : un seul POST
   en vol → quand il traîne, toute la fenêtre part d'un bloc. Mesuré 13-14 s
   bloquées sur 29888/29889. `RELIQUARY_MAX_INFLIGHT_FIRES=3` + re-clamp du
   budget sous `_pool_lock` (sans lui : 96 envois pour un plafond de 32).
4. Latence : un seul GET /state par tour, sleep 1,0→0,1 s en zone rouge, sonde
   `submit_diag` off, `_build_precommit` hors boucle asyncio.

**⚠️ RÈGLE D'OR** : ne JAMAIS juger une config sans contrôler l'ÂGE DU MOTEUR
(temps depuis le dernier restart). C'est ce piège qui a fait retirer puis
remettre la cascade, juger sprint 8 bon puis mauvais, et croire à des « heures
en or » qui n'étaient que des redémarrages récents.

**Chantier ouvert (soir du 20/08)** : la concurrence se durcit — le rang de
notre 1re entrée est passé de 14 à 28 à offset constant (+10 s) en une heure,
la fermeture du batch de +26 s à +19 s. Détection du flip vérifiée PARFAITE
(0 s d'écart vs poller). Leviers en réserve, non appliqués : brancher
`effective_gen_cap` dans `_bake_stream_fire` (**bug : la garde pré-flip
« capped » ne fait rien aujourd'hui**), cap de course 2000-2500 tok (~45 s de
GPU/fenêtre, permet un 2e lot précoce), borner le grading à ~12 sous-processus
(aujourd'hui jusqu'à 128 pour 24 CPU), `OMP_NUM_THREADS=4`, reaper de zombies
au watchdog, `VLLM_BATCH_INVARIANT=1` (A/B à faire pour les `token_tampered`).

## 🔭 SURVEILLANCES — à relancer au début de chaque session (2026-08-20)

Toutes lisent les 4 writers du mineur + le log de la box. Rien n'écrit sur la
prod. Relancer via l'outil Monitor (persistant) ou en tâche de fond.

| # | Script | Ce qu'il montre | Cadence |
|---|---|---|---|
| 1 | `scripts/live_monitor.sh` (+ `live_monitor_tick.py`) | **par fenêtre** : ENVOIS (offsets des acceptées, rejets) puis SEAL (rangs, payées 💰, échecs de vérif) + erreurs mineur en delta | 60 s |
| 2 | `scripts/watch_fixes.sh` | santé des fixes du 20/08 : `age` moteur, `gen1` (temps du 1er groupe), `fantomes` (processus de grading zombies), `courts` (taux de groupes à rollout <32 tok) | 10 min |
| 3 | `scripts/watch_competition.sh` (+ `competition_tick.py`) | **vigie concurrence** : rang de NOTRE 1re entrée, heure de fermeture du batch, alerte `⚠️ DURCISSEMENT` si rang méd >20 ou fermeture <9 s | 15 min |
| 4 | `scripts/harvest_window_timing.py` | moissonne flip/offsets/seal par fenêtre dans `data/window_timing_v4.jsonl` (survit aux troncatures de `miner.log` au restart) | 20 min |
| 5 | `scripts/poll_dashboard.sh` (tmux `dash81`) | vue MARCHÉ par fenêtre depuis reliqua.ai → `data/dashboard_market.jsonl` (peloton : rollouts, 32 acceptées, lags, enchère) | 30 s |
| 6 | `ops/pull_samples_v4.sh` (tmux `pull81`) | rapatrie les 4 JSONL de la box vers `data/` | 30 min |

⚠️ Le champ `upload_lag_ms` du dashboard n'est PAS dans le même référentiel que
notre `flip_offset_s` (mesuré 20/08) — ne jamais les comparer directement.

**Analyses à la demande** (scripts prêts) : `scripts/etude_debut_v4.py` (bilan
+ avant/après), `scripts/ab_v51.py` (A/B prior), `scripts/gain_fused.py`,
`scripts/extract_error_windows.py` (dossier des fenêtres à erreurs pour les
multi-agents), `scripts/competition_tick.py`, et sur la box `/tmp/*.py`
(`dissect_all`, `recent_offsets`, `ranks_now`, `funnel2`, `drops_timing`).

**Sur la box** : tmux `miner` + `watchdog81` (v3 : wedge 15 min, OOM 5/5 min,
**garde VRAM >128 Go**) + `monitor`. `restart_miner.sh` relance TOUJOURS
watchdog+monitor (fix 19/08). Marqueur `.miner_launcher` → `launch_miner_v4.sh`.

## 🎯 2026-08-15 — PLAN ACTIF : banc H200 → relance

**MAJ 2026-08-18 (17h) — ⚡ NOUVELLE BOX PROD = 38.255.28.21:20098, VALIDATEUR DOWN (bascule probable)** :
l'ancienne H200 (31.22.104.180) est RENDUE, mineur v3 éteint. La box louée
devient LA prod : install+modèle+gates (PASS eager 0.9572 / graphs 0.9674 sur
CETTE carte) + bancs + DRY-RUN CONFORMITÉ **PASS 7/7 ACCEPTED** (vrai mineur
vs validateur factice : signature/merkle/schéma/zone/gates + 4 writers étude).
Wallet réel copié (SS58 vérifié), chaîne de relance armée (marqueur
`.miner_launcher` → launch_miner_v4.sh, watchdog v2.1), tunnel inverse
tunnel81 prêt (egress box indéterminé tant que le validateur est down — le
launcher auto-détecte direct/tunnel et ABORT sinon). Validateur injoignable
depuis ~16h (dev box aussi) = coupure de bascule probable ; watcher /health
actif → à son retour : si v4, re-diff upstream puis LANCER (launch =
`tmux new-session -d -s miner "bash /workspace/launch_miner_v4.sh 2>&1 | tee /workspace/miner.log"`
+ watchdog + pull81 dev box).

**MAJ 2026-08-18 (après-midi) — ⚡ MERGE v4 FAIT, ACTIVATION EN ATTENTE** :
`origin/main = 96bfb46` (PR #180), zéro diff vs notre alignement `a6456b4`.
Validateur live TOUJOURS v3 (`cb23a1e`, degraded) — Discord : rester v3
jusqu'à la fenêtre d'activation annoncée ; re-diff upstream avant déploiement
s'ils repoussent. Prochaine étape décidée : banc débit v4 sur H200 louée ~2 h
(`scripts/deploy_bench_v4.sh` dev box + `ops/bench_v4_matrix.sh` worktree,
P1 = coût forced-seed full-support) — EN ATTENTE go + SSH box. Prior v5 :
collecte armée dans le launcher (dump v4 vierge + mémo auto-amorcé +
`ops/pull_samples_v4.sh`), cible à diagnostiquer H+2 (flat vs difficulté).

**MAJ 2026-08-18 (soir) — RUNBOOK JOUR-J PRÊT** : balayage multi-agents final
→ 12 gaps corrigés (dont BLOQUANT : restart/watchdog relançait en v3 → fix
marqueur `.miner_launcher` + `ops/launch_miner_v4.sh` ; miroir du gate
« uncertain » PR #178 ; polling /verdicts mort réparé). Branche = 20 commits.
**Au merge upstream, dérouler
`docs/superpowers/plans/2026-08-18-runbook-lancement-v4.md`** (worktree) —
préalable n°1 : réconcilier les fixes OOM non commités de l'arbre prod.

**MAJ 2026-08-18 (matin) — upstream v4 @ `a6456b4` (« release blockers » fermés →
merge imminent), branche locale RÉALIGNÉE** : canal grader `Answer:` supprimé
upstream (tamper guard) → v4 math = `\boxed{}` obligatoire sinon reward 0 ;
porté (`MATH_ANSWER_FORMAT`). Audit multi-agents du 17/08 : 18 items, 15
appliqués (dont bloquant `GENERATION_PROFILE_ID` — sans lui, 100 %
GENERATION_CONTRACT_MISMATCH), 3 restent (GPU jour-J). 17 commits, fixlist =
`docs/superpowers/plans/2026-08-17-audit-v4-fixlist.md` (worktree).

**MAJ 2026-08-17 (midi) — PORT v4 PRÊT** : branche upstream
`feat/qwen3-base-dapo-v4-profile` @ `8c38992` (PAS mergée, validateur live
toujours v3) = protocole v4 complet : Qwen3-4B-**Base**, 16 rollouts, sampling
1.0/1.0/topk0, **sans BFT**, cap 8192, fenêtre 150 s, zone σ 0.24 (k∈[1,15]
payable), 32 soumissions/fenêtre, prompts RAW sans chat template, OMI shards
`train-*` only. **Port de conformité FAIT** : branche `feat/port-v4-dapo`
(worktree `/root/subnet81/.worktrees/miner-priv-port-v4-dapo`, 8 commits,
bascule = flag unique `RELIQUARY_PROTOCOL_VERSION=4`, défaut v3
byte-identique, zéro régression suite complète). Plan + étapes jour-J :
`docs/superpowers/plans/2026-08-17-port-v4-dapo.md` (worktree) + mémoire
project_port_v4_dapo_branch. ⚠️ NB : `reliquary-miner-priv/` EST un repo git
(branche `feat/predicteur-tfidf-k2`, la ligne « non git-tracké » du Layout
ci-dessous est périmée) ; fixes OOM du 16/08 encore non commités dans l'arbre
prod — à réconcilier avant tout déploiement v4.

**MAJ 2026-08-17 ~02:00 (nuit)** : fixes OOM validés en continu (zéro
OOM/wedge depuis 15:00, tous les reloads propres). Taux vedettes v4.3 mesuré
**18,1 % [12,4-23,7]** (77 fenêtres). Acquis vérifiés : seal = 8e prompt
distinct (B_BATCH=8, pas 300 s) ; latence prêt→soumis 5 s médiane (pas un
levier) ; explo uniforme 0,7 % vs classés 3,3 % (ne pas monter EXPLORE_SLOTS).
⚠️ **7 duels prior INVALIDES** (éval biaisée pro-incumbent : v4.3 mémorise —
top13 7.00 vu / 0.00 jamais-vu) → cf. mémoires
project_v43_memorisation_eval_flaw + project_session_2026-08-16_17_nuit
(TODO réveil : datation éval → duel propre → prior hybride mémo+modèle).
PREDICTOR_2/vedette lourde : abandonné (redondant, l'ancien numérateur est
nuisible E=95). Ligne retirée du start_miner.sh.

**MAJ 2026-08-16** : box H200 ACTIVE `ssh -p 40300 root@31.22.104.180`
(SAMPLE_DUMP+EXPLORE_SLOTS=3, pull81 actif). **2 chantiers déployés le 16** :
① **v4.3 + SIZE=4 depuis 11:29** (`predictor_v43_2026-08-16.json`, 17 410
lignes/1 044 payables, duel sans fuite 38.5% vs 24.7% v4.1) — premières ~3 h :
**5/12 fenêtres payées (42%) vs 21% la nuit en v4.1/SIZE=2**, meilleur rang
médian 19→7 ; verdict statistique à ~40-50 fenêtres. ② **Fixes OOM depuis
15:00** (`ops/deploy_fixes_2026-08-16.sh`) : GPU_FRACTION 0.78→0.76, kill des
EngineCore zombies au reload (`_kill_stale_engine_cores`, 7 tests), watchdog
**v2.1** (détection ≥5 `pre_bake failed`/5 min ; garde anti-warmup = fichier
état `/workspace/.watchdog_last_restart` car `ps -o etimes` renvoie des
valeurs aberrantes sur cette box). Cause racine OOM (4 épisodes le 16, 2-4
fenêtres perdues chacun) : AUTO-infligé — EngineCore vLLM ~112 Go (0.78) +
preuves GRAIL ~27 Go dans le process principal → ~200 Mo de marge. Backup :
`gpu_backup_2026-08-16_h200/` + `scripts/deploy_h200_2026-08-16.sh`.
Ré-entraîner v4.4 à ~400-500 payables collectés (dump : 212/2250 au 16).

Dernière box H200 morte le 08-13 ~17:00 (Lium 93.120.231.186:40498). **Dernier
état mesuré : sprint exclusif → 2 fenêtres payées / 3 (rangs 2 et 9).**
Dès le prochain GPU (H200 de TEST d'abord, décision utilisateur), dérouler :
1. Install scriptée ~25 min (venv `--no-deps` depuis pip_freeze 08-07 + noms
   corrigés dans install.sh de la session 08-12 ; tunnel inverse si egress
   validateur filtré : `ssh -f -N -R 8080:209.20.157.231:8080 -p <port> root@<box>`).
   ⚠️ Tester les 5 miroirs drand DEPUIS la box (`curl --max-time 6
   https://<miroir>/public/latest`) et mettre les répondants dans
   `RELIQUARY_DRAND_URLS` — un miroir mort dans la liste = ~15-20 s de gel de
   boucle 1 tirage sur 5 = flips ratés (incident 28968-70 le 08-15,
   secureweb3 injoignable depuis cette box-là uniquement).
2. **Banc étage 1** `ops/bench_sprint_matrix.sh` (~1 h) : per-seq à 1/2/4/8/16
   prompts × FS ON/OFF. Références : 102 tok/s/seq @32 seqs (H200), 70 (H100).
3. **Banc étage 2** `ops/bench_sprint_matrix2.sh` : graphs, max_num_seqs,
   courbe longueur 512/2048/8192 (Mamba hybride → vérifier la décroissance).
4. Si surcoût FS >~25 % à petit batch → chantier kernel processeur
   (`warp_fast` en réserve, bit-exact) + gate parité OBLIGATOIRE.
5. Gate forced-seed carte (`/workspace/run_gate_4b.sh` pattern) puis **relance
   mining** : `ops/start_miner_h200.sh` (cap 16384, v4.1, sprint exclusif
   SIZE=2/WAIT=90, re-warm fix, garde GPU-vide, AUCTION_MIN_SCORE=0) +
   `watchdog.sh` corrigé (`prod_backup_2026-08-12/`).
6. Collecte 16384 en minant (SAMPLE_DUMP + pull auto 30 min, tmux `pull81`
   dev box) → **v4.3** dès ~300-500 positifs longs (script
   `scripts/train_v42_2026-08-14.py` à réutiliser).

### Diagnostic ESTABLI (mesures 08-12→13, ne pas re-supposer)
- Classement live = #171 flat (tout k∈[2,6] vaut 1.0) + #173 : rang =
  min(Σ tokens groupe, 8×15616) ÷ rounds depuis l'ouverture. **Plafond du rang
  = vitesse_par_seq × 8 × 3/50** (102 tok/s → bucket ~49 max).
- L'arrivée n'a JAMAIS été le problème (+16-27 s mesurés) ; le numérateur oui
  (groupes 6-10 k vs podium 40-80 de bucket).
- Prédicteurs v1-v4.1 : entraînés sur labels CENSURÉS aux vieux caps → fuient
  les prompts longs. v4.2 (48 positifs longs) ne bat pas v4.1 — il faut du
  volume au cap 16384. Modèles dans `data/predictor_*.json` +
  `ops/prod_backup_2026-08-12/`.
- Données : `data/samples_code.jsonl` = 35 327 groupes (~1 600 ère 16384).
- Fixes déployés dans l'arbre : re-warm au reload (+ plateau VRAM à 2 gardes),
  garde GPU-vide au launcher, watchdog anti-silence-total. Tests
  `tests/test_rewarm_on_reload.py` 7/7.
- Interdits protocole : décodage spéculatif (tokens forcés par u_at),
  quantization (casse seed_consistency 0.80/0.75).

## 🔧 2026-08-05 — RUN PRODUCTION : 2 causes racines corrigées

Box H100 `192.222.55.74:20301` (egress DIRECT — **morte, cf. ci-dessus**).
Détails complets : [[project_miner_run_2026_08_05]].

✅ **Tranche périmée** — `_grade_chunk_streaming` vérifiait le checkpoint mais
PAS la randomness avant `_pool.append()` → entrées bakées sous la fenêtre N
ajoutées pendant N+1 → 16/16 rejets hors-tranche. Contrôle de fraîcheur posé.
✅ **`bad_termination`** — 63% des rollouts soumis SANS EOS. La garde existait
et `verify_deployed.sh` la validait… mais elle était sur `_pre_bake_batch` et la
boucle async, **deux chemins INACTIFS**. Le chemin réel (`_pre_bake_entry`) n'en
avait aucune. Garde posée dessus → 0/64 rollouts fautifs, 111 groupes
abandonnés. ⚠️ pas encore confirmé par un verdict (bloqué en amont).
⏳ **Reste `precommit_expired`** : on tire en FIN de fenêtre, la grâce est 33 s.
Chantier = soumettre dès qu'une entrée est prête.

⚡ **24-33% des groupes sont PAYABLES** sur le 4B (vs 0,2% sur le 2B) — le
verrou historique de sélection est mort, ~7 payables par fenêtre pour un quota
de 8. ⚠️ **Leçon** : vérifier qu'un code EXISTE ne vaut rien tant qu'on n'a pas
vérifié qu'il S'EXÉCUTE (traces sur le chemin réel > lecture de code).

## 📋 2026-08-04 — PLAN DE REPRISE GPU : `PLAN_REPRISE_GPU.md`

Pas de GPU actuellement (H100 93.120.231.186 morte). **Cause racine des 21
`bad_termination` trouvée et corrigée sans GPU** (le chemin vLLM ne tronquait
pas au 1er EOS + garde locale trop faible) — 26 tests verts, non déployé.
**Lire `PLAN_REPRISE_GPU.md` (= `ops/PLAN_REPRISE_GPU.md`) avant de relancer** :
étapes ordonnées (simulateur hors-ligne → probe gaspillage `ignore_eos` →
fenêtres hors zone → gate seed-consistency 4B → prod), avec les gardes de
sécurité (⚠️ `bad_termination` est un rejet « haut risque » : 32/fenêtre
quarantinent l'entraînement du validateur).

## 🚨 2026-08-02 — MIGRATION 4B/v3 MERGÉE ET **LIVE** (jour J déclenché)

Le 4B a mergé dans `main` (PR #162, +#164/#165/#166) et **le validateur live
tourne DÉJÀ le 4B/protocole v3** — `curl 209.20.157.231:8080/health` →
`image_revision 70e795b`, `protocol_version 3`,
`generation_profile_id qwen35-4b-auction-v3`. Il mine activement (56 submissions
code/fenêtre). **Notre miner 2B/v2 n'est PLUS conforme → 100% rejet si relancé
tel quel.** Détails/plan [[project_4b_migration_watchlist]].

**Contrat live exact (`/health` generation_contract) — cible du port :**
- Modèle **Qwen/Qwen3.5-4B** @ `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` ;
  checkpoint suivi via `/state` = `ReliquaryForge/qwen3.5-4b-reliquary-v4`.
- `protocol_version = 3` (on émet 2 → à passer 3).
- `max_new_tokens = 16384` (les 2 envs) ; **BFT math** thinking **15616** /
  answer 512 / `force_answer true` ; **BFT code = null** (pas de BFT en code).
- **throughput_tiebreak ACTIF** : token_cap 15616, bucket 50 tok/round →
  débit = revenu (nos 14k tok/s + grading // deviennent un levier direct).
- `collection_seconds = 300` (remonté de 100s → plus de marge), upload_grace 33s.
- Sampling identique (8, T0.6, top_p0.95, top_k20) ; forced_seed floors
  0.80/0.75 inchangés ; `legacy_merkle_root_enforced true` (shadow OK chez nous) ;
  `forced_seed_cdf_enforced false` (epsilon 0.002).
- Stack validateur : **torch 2.7.0+cu128, transformers 5.9.0,
  flash_attention_2** (nous = sdpa → RE-VALIDER seed_consistency sur GPU 4B).

**À PORTER avant tout redéploiement** (ordre) : ① modèle 4B + checkpoint v4 ;
② protocol_version 3 + forced-seed domaine v3 (lire `protocol/profiles.py`
nouveau) ; ③ constants BFT 15616 / cap 16384 ; ④ ~~re-valider parité
forced-seed GPU sur 4B~~ **FAIT 2026-08-06** : gate 4B/graphs/batched-FS
`ops/validate_vllm_forced_seed_group.py` → **PASS groupe 0.9793, pire rollout
0.9231** (planchers 0.80/0.75). ⚠️ 3 pièges de lancement de la gate : env vLLM
du launcher requis (deep_gemm off + CUDA_HOME venv), et `max_num_seqs<=64`
(défaut 1024 > blocs Mamba du 4B à 0.35 de VRAM) ; ⑤ le prédicteur passe au
second plan (verrou sélection 2B mort), débit = priorité revenu.
⚠️ Voir aussi [[project_4b_port_progress]] : port BFT env-configurable
partiellement commencé (TDD) — reprendre de là, aligner sur 15616/16384.

## ⚡ PERFORMANCE ACQUISE (2026-07-23, H200 143 Go) — NE PAS PERDRE

**Débit réel mesuré du mineur en régime (code-only, H200), sur 39+ appels
phase-1 batchés — pas un instantané de barre :**
| batch | séquences | débit médian | n | temps génération | fenêtre 100s |
|---|---|---|---|---|---|
| 16 | 128 | **10 961 tok/s** | 39+ appels | 23s médian (15-47) | OK large |
| 20 | 160 | ~14 400 tok/s (~+29%) | **n=10 seul.** | 28s médian | OK |
Config : CUDA graphs ON, gpu_fraction 0.55, forced-seed appliqué. 0 OOM.
⚠️ **Le batch 16 est solide (39+ appels, plusieurs fenêtres). Le batch 20 = 10
générations seulement (~2-3 fenêtres, min 14k max 18,3k) → +29% INDICATIF, pas
statistiquement confirmé.** La tendance (plus de séq en vol = mieux amorti) est
attendue et cohérente, mais RE-MESURER le batch 20 sur plusieurs fenêtres avant
de traiter le +29% comme acquis. Réglage batch 20 déployé quand même (au pire =
égal au 16, jamais pire).

**Code source = GitHub `Fanoutch/subnet81` (à jour).** Sur box fraîche : cloner,
dérouler bring-up (voir mémoires), lancer. Tout est committé.

### Les 2 optimisations qui ont donné ce débit (committées)
1. **Grading parallélisé** (`grade_group_parallel`, commit 597a67a) — LE gros
   gain. Avant : le mineur passait **52% du temps sur `reward` avec le GPU à
   0%** (correction des 8 rollouts en série via subprocess). Après :
   generation 48%→**92%** du temps, GPU 78%→**97%**, ~2× plus de prompts/h.
   Sûr : ne touche NI tokens NI preuves.
2. **CUDA graphs** (`RELIQUARY_VLLM_CUDA_GRAPHS=1`) — +48% à batch 8. Conformité
   forced-seed vérifiée (seed_consistency 0.988). Gate `GATE_EAGER=0` à repasser
   sur toute nouvelle carte.

### Ce qui NE change PAS le débit (mesuré, ne pas re-tester)
- `gpu_memory_utilization` 0.55 vs 0.85 : **0%** (le goulot n'est pas le cache KV)
- Batch au-delà de 20 : sature vite avec CUDA graphs
- La génération n'a JAMAIS été le goulot : vLLM nu = 26 000 tok/s. Le forced-seed
  et surtout l'alternance GPU/CPU série (grading) bridaient tout.

### Config launcher H200 (dans scripts/launch_miner.sh)
`RELIQUARY_ACTIVE_ENVS=opencodeinstruct` · `BAKE_BATCH_SIZE=20` ·
`VLLM_CUDA_GRAPHS=1` · `VLLM_GPU_FRACTION=0.55` · `VLLM_FORCED_SEED=1`.
⚠️ Le launcher garde parfois la valeur H100 (0.55 = OK ici de toute façon).
Egress validateur DIRECT sur cette box (pas de tunnel). Box Lium = efface tout
au reboot + port SSH change (`ssh root@134.199.206.142 -p 20299` au 2026-07-23).

### PIPELINE PROUVÉ SAIN (2026-07-23)
Verdicts validateur : **8 `accepted`, 0 rejet de conformité** (0 SEED_MISMATCH /
GRAIL_FAIL / TOKEN_TAMPERED). Ordre par cycle : ① génère 16-20 prompts (23-28s)
→ ② **grade** (parallélisé, ~1s, donne les scores) → ③ filtre σ → ④ **GRAIL
SEULEMENT si σ∈[2,6]** (proof-skip) → ⑤ precommit + envoi. Jusqu'à **8 submits
acceptés/fenêtre** (le max). Respecte les tranches (écart 4500/5000). Timing OK,
le `window_not_active` = quota 8 atteint, PAS trop lent.

### ⛔ LE SEUL VRAI VERROU RESTANT = SÉLECTION DE PROMPTS
Débit, pipeline, conformité : tout est réglé. Il reste **~0,2% de groupes
payables** (σ≥0.43, k∈[2,6]) : le modèle résout ou rate uniformément les prompts
de la tranche. En code-only : 230 groupes → 0 payable. C'est le prochain
chantier (prédicteur), et LE seul qui touche le revenu.
Détails complets : mémoire [[reference_bench_optimisation_2026_07_22]].

## Session-start checklist

**ALWAYS run first** : `bash /root/subnet81/scripts/check_reliquary_updates.sh`
(fetch du clone upstream + branches/forced-pushes). Si le protocole miner change
(nouveaux rejects, champs BatchSubmissionRequest, env), le signaler AVANT tout.

## Layout — 3 arbres, ne pas confondre

- **`reliquary-miner-priv/`** — LE miner de prod (`python -m reliquary.cli.main
  mine`). Fork reliquary avec la logique dans `reliquary/miner/`. **Non
  git-tracké** — copies rsync locale ↔ GPU.
- **`my_code/`** — ancien design custom_miner (git `master`). **Pas utilisé.**
- **`reliquary/`** — clone upstream read-only. ⚠️ Working tree @ `d9471f2`
  mais `origin/main` = **`89a0ed7`** (2026-07-14) — lire les fichiers récents
  via `git show 89a0ed7:<path>`.

## État 2026-07-13 — code PRÊT (CPU-vérifié vs b790e42), infra à refaire

Audit complet + fixes wiring faits (sessions 07-12/13) :
- **Forced-seed aligné** : `environment/forced_sampling.py` +
  `miner/forced_seed_sampler.py` byte-identiques à b790e42 ; constantes
  `FORCED_SEED_*` + `GRAIL_PROOF_VERSION=v7` identiques ; `protocol_version=1`
  émis (`engine.py` ~1650) ; invariant validateur `tokens==commit.tokens`
  satisfait par construction (`_finalize_pool_entry`).
- **Fixes wiring prod (2026-07-13)** : ① fallback `hf_model` quand
  `vllm_model=None` sous enforcement ; ② **fire-as-ready** sous enforcement
  (pool flushé à chaque flip de randomness → le legacy single-burst-à-l'OPEN
  forfaitait toutes les fenêtres) ; ③ persistance disque du pool OFF sous
  enforcement ; ④ **OMI → `VirtualParquetDataset`** 14M lignes, pinné
  `469216e3` = même révision que le validateur → **kill-switch
  `PROMPT_RANGE_ENFORCE` FERMÉ**. Tests : 23/23 verts
  (`tests/test_forced_seed_*.py`, `test_bft_*.py`, `test_generate_m_rollouts_bft.py`).
- **σ** : comportement effectif = **0.43 partout** (`zone.py`
  `ZONE_THRESHOLD_STEADY`, appelé `bootstrap=False` en dur ; branche continue
  vise 0.43 explicitement). `constants.SIGMA_MIN=0.33` = constante **morte**
  (upstream 0.43) — désalignement cosmétique seulement.
- **Fails de tests connus** : `tests/unit/test_submitter.py` +
  `test_batch_submission_schema.py` + `test_commit_model.py` = fixtures
  pré-existantes jamais mises à jour (`env_name` manquant, proof v5). PAS une
  régression.
- **Subnet live** : validateur (uid 237) a déménagé → **`209.20.157.231:8080`**
  (l'ancien `86.38.238.30` est mort). Transparent : auto-découverte via
  `discover_validator_url(metagraph)`. Checkpoint = **nouveau repo**
  `ReliquaryForge/qwen3.5-2b-reliquary-v2` (suivi dynamiquement via
  `state.checkpoint_repo_id`+`revision` → transparent aussi).
- **⚠️ AUCUN GPU** : H100 `86.38.238.43` re-provisionnée (host key changée,
  plus notre box) ; H200 Lium rendue le 2026-07-03. Backup partiel :
  `gpu_backup_2026-07-02/`.

### MAJ 2026-07-14 — upstream b790e42 → 89a0ed7 (#108–#124), re-audité

11 PRs mergées, **toutes validateur-side, zéro changement `reliquary/miner/`**.
Vérifié sain : `u_at`/`pick`/`warp` inchangés ; planchers forced-seed inchangés
(0.80/0.75) ; #108 termination inclut l'escape `terminal_pick_ok` (EOS tiré
légalement par le forced stream = accepté — sans lui 67% des soumissions
honnêtes mouraient `BAD_TERMINATION`) ; parité merkle byte-exacte confirmée vs
`protocol/legacy_merkle.py` (shadow, enforçable via `LEGACY_MERKLE_ROOT_ENFORCE`
plus tard → safe pour nous) ; `FiniteFloat` OK (nos logprobs = log_softmax fp32,
jamais ±inf) ; `virtual_parquet` = résilience fetch only (mapping consensus
intact). **Fix requis** : 3 nouveaux `RejectReason` (`hotkey_not_registered`,
`registration_unavailable`, `merkle_root_mismatch`) renvoyés en HTTP 200 →
notre parse enum strict (`BatchSubmissionResponse.reason`) lève ValidationError
→ **FIX FAIT (vérifié 2026-07-15)** : enum 39 membres identique à origin/main,
champs `Verdict` identiques, handling en place (`registration_unavailable`=
requeue, `hotkey_not_registered`=drop+compté). **Drop-on-ckpt AUSSI FAIT** :
`drop_pool_on_ckpt_advance()` force le drop en code sous `FORCED_SEED_ENFORCE`.

### MAJ 2026-07-15 — upstream 89a0ed7 → c33560d (#126–#131), re-audité

`u_at`/`pick` toujours intacts ; enum toujours en parité. Nouveau : ① **runtime
fingerprint** — le miner de référence POST un profil runtime (`/runtime-contract`
+ `RuntimeFingerprint` lié au nonce). **OPTIONNEL sur le wire** (`| None`,
binding nonce seulement si présent) → pas bloquant ; à porter plus tard (parité
+ calibration validateur). ② **Difficulty auction en SHADOW sur OMI** — zéro
reject aujourd'hui, mais valide la thèse du prédicteur (Couche 2). ③ Port
**wire-v2 anticipé côté nous** (`tests/test_wire_v2_port.py`, `protocol/merkle.py`,
`signatures.py`) : 6/8 verts, 2 rouges = features à finir (`env_name` unique par
groupe, reject `protocol_version_mismatch`) — wire-v2 PAS mergé upstream, non
bloquant. **Watchlist** : branches "wire-v2 cutover" + "protocol-admission-v2" +
plancher 0.90 candidat = upgrade protocole annoncé — re-checker chaque session.
**MAJ 15/07 soir** : #132 mergé = spec complète "difficulty auction" (805 l.,
toujours shadow, zéro impact wire). Branche **`design/difficulty-auction-v2`**
(non mergée) = le vrai upgrade qui se prépare : `u_at` **sans hotkey** (stream
forcé identique pour tous les miners → variance farming mort, domaine v2
séparé), seal **classé par difficulté**, deadline 300s, preuve différée. Si ça
merge : re-port u_at trivial, et le **prédicteur (Couche 2) devient LE cœur de
la stratégie** (le rang de difficulté décide qui est payé).
Fails de tests dev-box classés (aucune régression) : vllm/hypothesis absents
(env), fixtures stale (`ENVIRONMENT_NAME=="math"`, interdiction
`RELIQUARY_MAX_NEW_TOKENS`), tests GPU sans driver.

### MAJ 2026-07-17 — upstream c33560d → 0352476 (#133–#139), re-audité

**Wire-v2 toujours PAS mergé** (`agent/wire-v2-cutover` inchangé depuis 14/07).
`reliquary/miner/` + `reliquary/protocol/` : zéro changement. Tout est
validateur/training-side (KL reference fixe, recovery checkpoints, wandb, CI) ;
seul fichier partagé touché = `environment/grader/server.py`, diff purement
infra (IDs conteneur runsc, close métriques) → zéro sémantique grading.
Branche `design/difficulty-auction-v2` avancée (923be3e) mais delta = KL
training + un fix de gates score HTTP/batcher — `u_at`/planchers intacts.
Branches `design/difficulty-auction` (v1) et `codex/difficulty-auction-shadow`
supprimées (remplacées par v2). Watchlist inchangée.

### MAJ 2026-07-17 (soir) — ⚠️ #140 MERGÉ : forced-seed v2 → PORTÉ + testé

**`design/difficulty-auction-v2` a mergé dans `main` (#140, `6f32673`)** — et
pour la 1re fois ça touche `reliquary/miner/` + `environment/forced_sampling.py`.
Le **validateur LIVE tourne déjà `6f32673`** (`curl …:8080/health` →
`image_revision 6f32673`). En l'état v1 = **100% `SEED_MISMATCH`** (gate
pre-queue `protocol_version!=2` sous enforcement, `server.py`+`batcher.py`).
**Port FAIT (TDD, parité byte-exacte vs source 6f32673, 46 tests verts)** :
- `u_at()` **perd la hotkey** (`environment/forced_sampling.py`) + son appelant
  `miner/forced_seed_sampler.py` ; `FORCED_SEED_DOMAIN`→`…-v2` ;
  `FORCED_SEED_PROTOCOL_VERSION`→`2` (constants). Sampler reste byte-identique
  upstream (hotkey gardé sur le processor, juste plus passé à u_at).
- **NE PAS flipper `RELIQUARY_WIRE_V2`** : il couple 3 choses dont 2 FAUSSES vs
  6f32673 → Merkle canonique (validateur calcule **legacy**, `MERKLE_ROOT_MISMATCH`
  si enforce) + sig enveloppe domaine-v2 (préimage upstream sans `protocol_version`
  → `BAD_ENVELOPE_SIGNATURE`, enforce ON). Flag reste **0** ; on émet quand même
  `protocol_version=2` (découplé, via `FORCED_SEED_PROTOCOL_VERSION`). Test
  `test_wire_v2_port.py` mis à jour pour ce contrat.
- Reste sain : proof toujours **inline** (deferred = validateur diffère la
  *vérif*, pas l'envoi), `/verdicts` déjà géré, planchers 0.80/0.75 inchangés,
  `RuntimeFingerprint` optionnel, caps anti-farming (2 slots/op, dette preuve)
  n'affectent pas un mono-hotkey honnête.
- **Stratégie auction** : sigma gate 0.43 **MORT** (seul unanime rejeté). Score
  sélection = `std·(1-mean)` du vecteur reward → **payé pour prompts DURS** (peu
  de rollouts réussis). Le **prédicteur (Couche 2) = cœur du revenu** maintenant.
- **vLLM bring-up RÉSOLU** (H100 Lium) : nvcc==CUDART(13.0) + backend **triton**
  `additional_config gdn_prefill_backend=triton` (= chemin FLA du validateur) →
  `[smoke] SUCCESS`. Détails [[reference_vllm_qwen35_2b_bringup]].

## Checklist déploiement (jour J, dans l'ordre)

0. ~~Sync `RejectReason`~~ **FAIT** · ~~drop-on-ckpt~~ **FAIT** · ~~port
   forced-seed v2~~ **FAIT** (2026-07-17, 46 tests verts, parité vs 6f32673 —
   flag `RELIQUARY_WIRE_V2` reste 0). Optionnel : porter `RuntimeFingerprint`.
   ⚠️ Sync `reliquary-miner-priv/` vers la box GPU AVANT lancement (les 4
   fichiers portés ne sont que sur la dev box).
2. Louer une box GPU + installer le stack **HF-only suffit** : `torch 2.11 +
   transformers 5.9.0` (= pin GRAIL validateur) + huggingface_hub/pyarrow.
   **vLLM PAS nécessaire pour le MINING** (sous enforcement le moteur force la
   boucle sync HF et bypasse le backend) — mais **l'installer quand même pour
   le probe prédicteur** (cf. section Prédicteur : probe d'abord, mining
   ensuite). Recette : mémoire [[reference_vllm_qwen35_2b_bringup]].
3. `rsync -rc --exclude='__pycache__' --exclude='*.log' --exclude='data/'
   /root/subnet81/reliquary-miner-priv/ root@<GPU_IP>:/root/reliquary-miner-priv/`
4. **Créer + enregistrer la hotkey** — pas avant, pour ne pas brûler
   l'immunité à vide :
   `btcli wallet new_hotkey --wallet.name camille81-v2 --wallet.hotkey hotkey81`
   puis `btcli subnet register --netuid 81 --wallet.name camille81-v2 --wallet.hotkey hotkey81 --network finney`
   (burn ~τ0.0005).
5. Lancer en tmux :
   ```bash
   cd /root/reliquary-miner-priv && PYTHONPATH=. \
     python -m reliquary.cli.main mine \
       --wallet-name camille81-v2 --hotkey hotkey81 --network finney --netuid 81 \
       --checkpoint <Qwen/Qwen3.5-2B ou ReliquaryForge/qwen3.5-2b-reliquary-v2> \
       --log-level INFO 2>&1 | tee miner.log
   ```
   (plus le vieux `Qwen3-4B` ; le miner suit ensuite `/state` dynamiquement).
   `RELIQUARY_MAX_NEW_TOKENS` ≥ 2560 (défaut 8192 OK).
6. **Validation runtime** : `seed_consistency ~1.0` sur les premières fenêtres,
   verdicts `ACCEPTED` (zéro `SEED_MISMATCH`/`TOKEN_TAMPERED`), parité GRAIL
   (calcul preuve + `verify_proof`), timing window OK.

## Protocole actuel (b790e42) — essentiels

- **Modèle** : Qwen3.5-**2B THINKING**. **BFT math-only** : thinking cap 2048
  (exact) → force `</think>\n\nFinal Answer: \boxed{` → answer cap 512 ;
  sampler T=0.6 / top_p=0.95 / top_k=20 ; proof **v7** ; cap total 32768.
  `reliquary/miner/bft.py` = port byte-exact.
- **Forced-seed (#106, ENFORCED)** : chaque token forcé au pick inverse-CDF
  public `u_at(randomness, hotkey, prompt_idx, checkpoint_hash, rollout, t)` —
  1 génération légale par (miner, prompt, rollout, window), vérifiée par
  teacher-forcing. Gate validateur = `FORCED_SEED_ENFORCE && checkpoint pinné`
  (pas de bypass `protocol_version=0`). Planchers : 0.80 groupe / 0.75 rollout.
- **Conséquence archi** : génération randomness-dépendante → window-time only,
  pool intra-window (flush à chaque flip), HF-only, fire-as-ready. Ce qui
  survit au pré-calcul = le **prédicteur** (scoring de prompts sur texte).
- **Grader OMI symbolique** (`e7155f2`) porté byte-exact — math n'est PAS
  validator-authoritative → sans le port, `x+1`==`1+x` = REWARD_MISMATCH.
- **Rejects gérés** (`_submit_with_retries`) : ACCEPTED · SUBMITTED ·
  SUPERSEDED (retry 1×) · WINDOW_MISMATCH/WINDOW_NOT_ACTIVE (rebuild+retry) ·
  HASH_DUPLICATE (drop) · RATE_LIMITED/BATCH_FILLED (cap) · PROMPT_FULL (drop) ·
  STALE_ROUND/FUTURE_ROUND (drop) · WRONG_RANDOMNESS (drop+ERROR) · SEED_MISMATCH.
- **STALE_ROUND résiduel** = design validateur (tolérance 0, queue lag > 3s) ;
  mitigation = soumettre vite, une part est inévitable.

## Pièges connus (toujours valides)

- **Si ré-activation vLLM un jour** : défauts `max_tokens=1500` dans
  `vllm_backend.py` — la phase-1 math DOIT être 2048 exact, sinon 100%
  TOKEN_TAMPERED sur les forced (span pinné `prompt_len+2048`, aussi en
  preflight). L'async loop reste interdite sous enforcement (pas de
  LogitsProcessor côté vLLM).
- `truncated` = écrasé à l'ingestion (ne pas l'envoyer) ; `forced=True` wipé
  hors math ; token-auth 1e-8 (marge ~35×) ; all-token-auth (#105) ne rejette
  que les tokens ÉDITÉS (`chosen_prob<1e-5 ET argmax≥0.99`) → forced-seed passe.

## Multi-env (math+code) — statut

Le validateur live est dual-env, split émissions ~50/50 par env (part non minée
= **burn**) → OMI-only nous cappe à ~50%. Plans A+B+C **codés + CPU-testés** :
opencode curated aligné (grader byte-exact, `code_grader_driver.py`),
`MixController`, routage bake/slice/reward par env, `/verdicts` porté.
**Reste** : Task 7 parité GPU math-only, puis flip
`RELIQUARY_ACTIVE_ENVS=openmathinstruct,opencodeinstruct`. Propriété de sûreté :
single-env = strictement identique au comportement actuel. Détails :
`reliquary-miner-priv/docs/superpowers/plans/` + mémoires projet.

## Prédicteur de difficulté (Couche 2 — sélection de prompt)

Données à **régénérer from scratch** sur le 2B (`scripts/difficulty_probe.py`,
GPU, tmux, lent). **Ordre jour J** : générer les données probe EN PREMIER
(avant re-reg hotkey — pas de pression d'immunité pendant la géné), puis
mining. **vLLM requis pour le probe** (défaut `--backend vllm`, débit ; recette
bring-up = mémoire reference_vllm_qwen35_2b_bringup) — toujours PAS requis
pour le mining (HF-only sous enforcement). Probe vLLM et mining HF ne
tournent pas ensemble sans capper `gpu_memory_utilization` → séquencer.
~~FIX sampler probe math~~ **FAIT (2026-07-16)** : `bft_generate_math()` dans
`scripts/difficulty_probe.py` = sampler v7 (TOP_P/K_PROTO) + flux BFT complet
(phase-1 = 2048 exact → EOS naturel gardé / `</think>` continue sans force /
sinon template forcé → phase-2 512) ; `--max-tokens` ignoré en math (budgets
protocole). Tests : `tests/test_difficulty_probe_bft.py` (5, TDD). Câblage
`pick_prompt_idx` après validation AUC. Enjeu monté d'un cran :
sous auction-v2 (u_at sans hotkey), le rang de difficulté = ce qui est payé.

## Don't-do reminders

- **Pas d'install reliquary sur la dev box** — single-source sur GPU.
- **Pas de cache tokens pour replay** — HASH_DUPLICATE (rétention 10 000
  windows) ; le pool est one-shot by design.
- **Pas de commit sans demande** — l'utilisateur review le diff d'abord.

## How to update this file

Mettre à jour la section concernée quand quelque chose de matériel change.
**Garder < ~200 lignes** : l'historique va dans les mémoires
(`~/.claude/projects/-root-subnet81/memory/`) et `docs/superpowers/plans/`,
pas ici. Ancienne version longue : `CLAUDE.md.bak-2026-07-13`.
