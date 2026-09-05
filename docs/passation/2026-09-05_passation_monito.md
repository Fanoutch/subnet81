# PASSATION — Mineur Subnet 81 (Reliquary) — état au 2026-09-04

> But de ce fichier : reprendre dans une nouvelle conversation sans perdre le fil.
> Lire aussi `CLAUDE.md` (contexte complet) et `MEMORY.md` (index mémoires).

---

## 🎯 OBJECTIF ACTUEL

Augmenter la **présence** du mineur (part de fenêtres où on est PAYÉ).
- **Nous : 46,5 %** des fenêtres · **meneurs : 76-81 %** (mesuré R2, 883 fen).
- Dans les fenêtres où on est payé, nos **rangs sont bons** (rang 1 possible),
  vitesse comparable aux meneurs. Le problème n'est PAS la qualité par entrée,
  c'est le NOMBRE de fenêtres où on arrive à temps.

## 🔑 LE DIAGNOSTIC CENTRAL (établi cette session)

À **arrivée égale** (round 2) on génère **10235 tokens** contre **8137-8666**
pour les meneurs les + constants (5Gukj2/5CAmEg/5E5E3S). Plus long = plus gros
**traînard** (le + lent des 16 rollouts) = arrivée plus tardive et surtout plus
VARIABLE → seulement **63 %** de nos payées arrivent au round 2 contre **83-88 %**
chez eux. D'où la présence 47 % vs 80 %.

⚠️ **Les « trous » de fenêtres mortes ne sont PAS un mécanisme caché** : trou
moyen 2,20 fen vs 2,15 attendu par pur hasard à 47 %. C'est juste la signature
d'un taux à 47 %. Monter le taux → les trous disparaissent seuls.

## ⛔ CE QUI A ÉTÉ TESTÉ ET REJETÉ CETTE SESSION — la bande de volume

**Idée** : tirer notre sélection de prompts vers la bande de volume des meneurs
(~8800 tok) via un malus `−β·|volume_score − cible|` dans `ScoreTable.combined`.
Déployé 21:51, **REPLIÉ 22:43**. **RÉSULTAT : INERTE** — traînard 864 vs 823
(PAS réduit), présence 42 % vs 46,5 %, out_of_zone 1 (aucun dégât).

⚡ **LEÇON DE FOND (ne pas re-tenter) :** sous forced-seed, la longueur RÉELLE
d'un groupe est imposée par (prompt × randomness de la fenêtre). On PRÉDIT le
volume (Spearman 0,51) mais on ne le CONTRÔLE PAS par la sélection : « prédit
court » ne donne pas un groupe réellement court (split-half du traînard par
prompt = 0). Le gap 10235 vs 8300 est réel mais **non fixable par le prior** —
c'est du biais de sélection au paiement OU une config meneur différente.
→ mémoire `project_bande_volume_rejetee`.

## 🔒 ÉTAT LIVE DU MINEUR (à l'instant)

- **Box : `ssh root@157.10.162.245 -p 20301`** (H200). Code dans
  `/workspace/reliquary-miner-priv/`, launcher `/workspace/launch_miner_v4.sh`,
  cwd + PYTHONPATH = `/workspace/reliquary-miner-priv`.
- **REPLIÉ PROPREMENT** sur l'état 0,66/fen connu : `PROMPT_SCORES` =
  `prompt_scores_unique_v1.npz`, PAS de `VOLUME_BAND_MU`, `VOLUME_MU=0`,
  `MEMO_SLOT=0`. 0 traceback. Table active = unique_v1 (2 481 806 prompts).
- Le **code de bande reste sur la box mais INERTE** (band_mu défaut 0 =
  comportement strictement inchangé). Non commité (à commiter ou jeter plus tard).
- Restart = `bash /workspace/restart_miner.sh` (relance watchdog+monitor aussi).
- Point de retour vérifié : `ops/CONFIG_LIVE_2026-09-03.txt`.

## 🧰 CE QUI A ÉTÉ MODIFIÉ (worktree, NON commité)

Worktree prod : `/root/subnet81/.worktrees/miner-priv-port-v4-dapo/`
(branche `fix/course-2026-08-27`).
- `reliquary/miner/prompt_scores.py` : `combined()` a 2 params optionnels
  `volume_band_mu`, `volume_target` (défaut 0 = inerte). + test dans
  `tests/test_prompt_scores_precalcul.py` (10 verts).
- `reliquary/miner/engine.py` : charge `RELIQUARY_VOLUME_BAND_MU` /
  `_VOLUME_TARGET`, passe à `combined()` (chemin table + chemin live).
- `data/prompt_scores_unique_vol_v1.npz` : table greffée (score unique +
  colonne volume réelle de zone_v1, fingerprint identique). Conservée, réutilisable.
- **La box est à jour de ces fichiers** (md5 vérifiés), mais ils sont INERTES
  sans les variables d'env. **Décision à prendre** : commiter le terme de bande
  (inerte, propre, utile pour le μ volume classique) ou le retirer.

## 📋 PROCHAINES PISTES (par ordre, aucune n'est de la sélection de prompts)

1. **Mineur MATH sur 2e box** — le gros morceau, prévu « demain » (= aujourd'hui
   04/09). Sièges à barre basse (p50 48), +0,6-1,3/fen, indépendant du code
   actuel. Plan C existe. À trancher : même hotkey (quota 32 partagé) ou
   hotkey81.2 (créée, NON enregistrée). Prérequis : gate forced-seed sur la
   carte de la 2e box (`ops/validate_vllm_forced_seed_group.py`), tolère une
   carte moins rapide.
2. **2e décodeur / 2e carte pour LE mineur code** — seul moyen prouvé de sortir
   une tête précoce de plus sans contention (scan_holdoff/sprint-3 ont prouvé
   que sur 1 carte on ne peut pas). Gros chantier (gate forced-seed autre arch +
   dispatch dual dans engine). Latence PCIe OK (même box).
3. **Investiguer la config des meneurs** — s'ils cappent max_new_tokens plus bas,
   voir si on peut s'aligner SANS casser la terminaison sous forced-seed
   (risque bad_termination — à vérifier en source AVANT tout test).
4. **`RELIQUARY_STALE_FAST_REFIRE=1`** — patch déployé dormant (commit f31e71c),
   re-tir immédiat à round frais ; 21,6 % des têtes mouraient en stale_round.
   À armer + mesurer.

## ⚠️ VIGIES PERMANENTES (à chaque session)

- `seed_mismatch` / `token_tampered` = **ZÉRO toléré**.
- out_of_zone ≤ 0,7/fen ; payées/fen vs 0,66.
- Upstream (check auto au démarrage) : `design/fill-closed-v6` (paiement PAR
  token → changerait l'économie), `design/reliquary-v1-reconciliation`,
  `standalone-environment-integration-v2` (NOUVEAU 04/09),
  `fix/v5-seed-auth-false-positive` (NOUVEAU 04/09 — à regarder, touche l'auth).
- `RELIQUARY_PROTOCOL_VERSION=5` (validateur en v5 depuis 23/08).

## 🪤 RÈGLES DE MÉTHODE (chèrement apprises)

- **Jamais de restart mineur sans go explicite de l'utilisateur.**
- **≥30 fenêtres MÛRES** avant d'annoncer un chiffre (verdicts lag 20-28 min).
- **Ne jamais juger un moteur froid** (post-restart / rechargement checkpoint).
- Verdicts « décidés depuis T » = entrées SOUMISES avant T (retard verdict).
- Dédupliquer les verdicts par `merkle_root`.
- `flip_offset_s` = NOTRE référentiel ; l'arrivée validateur (R2
  `arrival_age_seconds` / `precommit_arrival_ts`) est la vérité de rang.
- R2 : archives via clé read-only `.env.r2` (JAMAIS commitée). `batch[]` =
  payées (avec rollouts/completion_length) ; `difficulty_auction.candidates` =
  toutes les admises (throughput_rank = −bucket).

---
*Fichier de passation créé le 2026-09-04. Mettre à jour l'état live avant de
fermer une session.*

---
## 📌 SESSION 04/09 (matin) — lecture R2 41000-41335 (335 fen depuis le repli 03/09 22:43)

**Chiffres** : 0,52 payée/fen · payées dans 40 % des fenêtres · ADMIS dans 96 %
(absents 3,6 % seulement) · quand payés : arrivée 7,1 s, P(payé)=0,90.
→ « présence » = présence PAYÉE ; on est presque toujours admis mais sous la barre.
Rejets validateur : reward_mismatch 8, out_of_zone 6, worker_dropped 1, **0 seed_mismatch / token_tampered**.

**Trouvaille n°1 — cooldown de CONTENU (levier propre, prêt à coder)** :
129/1605 de nos candidats (8 %) meurent `content_in_cooldown` = doublon de TEXTE
(autre idx) d'un prompt déjà sélectionné (par n'importe qui, jamais expiré :
BATCH_PROMPT_COOLDOWN_WINDOWS=1e6). Le validateur publie l'ensemble sur R2 :
`content_cooldown_snapshots/qwen3-4b-base-dapo-reasoning-v5-basereset-20260825.json.gz`
(rafraîchi toutes les 10 fen ; 136 866 digests code). Prédit 129/129, 0 faux positif.
Digest = sha256(b"reliquary/prompt-content/v1\0"+env+b"\0"+prompt utf8), reproduit
600/600 depuis notre `env.get_problem` (PROTOCOL 5). Table `/workspace/prompt_digests.npy`
(2 481 806 × 32 o, calculée 04/09) + `/workspace/burned_idx.npy` (163 807 idx = 6,6 %).
Coût mesuré : 15 fen perdues avec une entrée ≥ barre en cooldown, 34 fen où notre
1re entrée était en cooldown (23 non payées). Le prior SUR-sélectionne ces doublons ×7
(8 % des candidats vs 1,1 % de base). Greffe = `exclude |= burned_idx` (engine ~2865)
+ rafraîchissement du fichier depuis la dev box (clé R2 reste sur la dev box).
Gain attendu +0,04-0,06/fen. En bonus : 836 idx doublons de contenu de la liste noire σ=0.

**Trouvaille n°2 — anatomie des têtes lentes (104 fen)** : départ bake 0,1 s et g1 prêt
à 4,0 s IDENTIQUES aux fenêtres payées. Tout se perd APRÈS : (a) g1 σ=0 dans 53 %
(vs 18 % en payées) ; (b) quand g1 est envoyé, grade→arrivée precommit p75 8,2 s
(vs 1,5 s), avec stale_round en 1re tentative 16/37 (graded→1re tentative 6 s vs 2 s).
= le TEMPS B de la passation (instrumenter grade→preuve→finalize→precommit→POST).
Validateur lui-même : réponse p75 4,5 s, p90 6,6-9,6 s même en fenêtres payées.

**Trouvaille n°3** : `same_prompt_superseded` 69 candidats (8 fen perdues ≥ barre) =
loterie à bucket égal (tokens identiques sous forced-seed ; 41022 perdue en arrivant
AVANT). Collisions surtout avec 5Ey9cNoR. Non fixable sauf diversification.

**Upstream** : `fix/v5-seed-auth-false-positive` (non mergé) désactive le rejet
all-token-auth sous v5 (`ALL_TOKEN_AUTH_ENFORCE = PROTOCOL_VERSION != 5`) → notre
faux positif token_tampered disparaîtrait au merge. `reliquary-environment-suite-v1`
(+27, env « logic ») et `standalone-environment-integration-v2` (+29) = nouveaux envs
en préparation, à surveiller.

## 🔬 DÉPLOYÉ 04/09 14:21 — MÉMO DE TÊTE + VETO DES BRÛLÉS (fenêtre 41586)
Config live : `MEMO_SLOT=1`, `MEMO_HEAD_SLOTS=2`, `MEMO_RUN_START=32791`,
`MEMO_MIN_SCORE=0`, `BURNED_IDX=/workspace/burned_idx.npy`. Les 2 slots de sprint
vont aux ex-payables mesurés de la tranche (tri run courant > confirmations >
fraîcheur), hors brûlés/cooldown/liste noire. Chargé au boot : 66 636 payables
connus, 167 109 index brûlés (cooldown de tranche 258 → ~300-456 écartés).
Fichiers : `reliquary/miner/payable_memo.py` (top_in_range), `engine.py`
(burned_idx_load/_burned_active/memo_head_pick, exclude |= brûlés, mémo de tête),
`ops/launch_miner_v4.sh` ; tests `tests/test_burned_idx.py` + `test_payable_memo.py`
(40 verts). Rafraîchissement des brûlés : cron dev box */15
`scripts/refresh_burned_idx.py --push` (R2 → data/burned_idx.npy → box), digests
locaux `data/prompt_digests.npy`. **REPLI** : `RELIQUARY_MEMO_HEAD_SLOTS=0` (mémo
historique slot 3) ou `MEMO_SLOT=0` + restart ; launcher d'avant sauvé en
`/workspace/launch_miner_v4.sh.bak-avant-memo-tete-0904`.
Restart = 3 fenêtres perdues (41586 froid, 41587 couvre-feu sur horloge
d'ouverture fausse après restart — `fire_diag curfew:63`, 41588 reload ckpt).
À juger sur ≥30 fen mûres via R2 : hors-zone des têtes (réf 31 %), payées/fen
(réf 0,52), `content_in_cooldown` (réf 8 % des candidats), hash_duplicate /
same_prompt_superseded (course forced-seed), seed_mismatch = 0.
Attendu : 0,6-0,65/fen. Premières fenêtres : 41589 tête mémo à 5,4 s → soumise 8 s.

## 🧭 ORDRE DES CHANTIERS (décision utilisateur 04/09 15:50 : UN changement à la fois)
1. **EN COURS** : mémo de tête (14:42, fen 41602+). Ne rien toucher jusqu'à 30 fen
   mûres (lecture programmée 16:27 UTC : `scripts/memo_r2_compare.py --debut 41602`).
   Lecture partielle 17 fen : 0,71/fen vs 0,52, têtes hors zone 6 % vs 31 %.
2. **SUIVANT, prêt en patch** `ops/patches/2026-09-04_prefetch_hors_boucle.patch` :
   le sondage HF du préchargement (`list_repo_commits`, 718 commits = 15 requêtes,
   **3,83 s mesurées**) tourne en SYNCHRONE sur la boucle asyncio toutes les 15 s →
   boucle gelée ~20 % du temps (py-spy) → flip vu ≥2 s en retard dans 23 % des
   fenêtres (P(payée) 0,54 → 0,29 → 0,18 selon le retard), tirs/réponses retardés
   (stale_round). Correctif = `to_thread` + `model_info` (1 requête, 0,23 s, même
   sha), 2 tests. RETIRÉ de la box (code inactif qu'un restart watchdog aurait
   activé). À déployer SEUL après le verdict du mémo, puis mesurer.

## 🔬 DÉPLOYÉ 04/09 16:39 UTC — PRÉCHARGEMENT HORS BOUCLE (mémo de tête conservé)
Verdict mémo (52 fen mûres 41602-41653) : **0,79 payée/fen vs 0,52**, 60 % fen payées
vs 40 %, têtes hors zone 11 % vs 31 %, content_in_cooldown 0 % → GARDÉ.
Changement n°2 : patch `ops/patches/2026-09-04_prefetch_hors_boucle.patch` appliqué
(`checkpoint_prefetch.py` : sondage via `asyncio.to_thread` + `hf_latest_commit_only`
= 1 requête `model_info` 0,23 s au lieu de `list_repo_commits` 15 requêtes 3,83 s
en synchrone sur la boucle toutes les 15 s ; `engine.py` `_list` → `_cp.hf_latest_commit_only`).
md5 box == worktree (engine a4946906, prefetch 2b1da8fa). Restart 16:39:28, PID 2869227.
Preuve du mécanisme : fen 41668 — têtes prêtes 2,8/3,7 s, POST retenu 4 s par la
boucle gelée (thread principal muet 16:29:15→19), stale_round, re-tir → 10,6 s round 4.
À mesurer (lecture programmée 17:57 UTC) : stale_round en 1re tentative (réf 15 %),
retard de détection du flip (réf méd 0,8 / p90 3,8 s / ≥2 s 23 %), payées/fen vs 0,79.
Repli : `git checkout` des 2 fichiers + restart (le mémo reste).
**Suivant (n°3, non fait)** : mémo qui apprend des verdicts validateur — 17/18 rejets
`out_of_zone` validateur de l'ère mémo sont des têtes mémo jugées en zone localement
(k 3-10) et REJOUÉES (2046715 rejeté 3× en 16 fen) ; ~3-4 fen perdues/62. Fix =
retirer du mémo (et lister noir) tout prompt rejeté out_of_zone/reward_mismatch par le
validateur (`_apply_verdicts` : ajouter merkle→prompt_idx).

## 🔬 DÉPLOYÉ 04/09 19:16 UTC — LE MÉMO APPREND DES VERDICTS (mémo + préchargement conservés)
Lecture mémo+préchargement (60 fen mûres 41677-41736) : **0,87 payée/fen**, 68 % fen
payées, détection flip p75 1,8→0,8 s, ≥2 s 23→12 % (reste = checkpoint) ; stale_round
inchangé 18,7 % (= validateur, réponses 5-10 s à l'affluence, pas de gel chez nous).
Changement n°3 : `_apply_verdicts` — verdict validateur `out_of_zone`/`reward_mismatch`
→ prompt retiré du mémo + `sz_blacklist_ban` (20 000 fen, persistée) ; merkle→prompt
retenu à l'envoi (`_submitted_prompt`). Tests `tests/test_memo_verdict_ban.py` (3).
Pourquoi : 35 rejets validateur zone / 135 fen (vs 6/336 avant mémo), 29 prompts,
6 récidives (2046715 3×) — note locale en zone (k 4-9), validateur non, mémo rejouait.
Au déploiement : 146 prompts déjà rejetés (R2 ≥39275) injectés dans `sz_blacklist.json`
(10 160 → 10 274) AVANT le 1er chargement du nouveau process. engine md5 9edc5a0d.
À mesurer (≥30 fen mûres) : rejets validateur out_of_zone (réf 16/60 fen), récidives = 0,
payées/fen vs 0,87. Repli : `git checkout engine.py` (hunk verdicts) + restart.
Reste connu : /state = 1,16 Mo (141k cooldown idx) validé par pydantic à chaque tour
= 32 % du thread principal, ~0,3 s/tour (candidat n°4) ; stale_round côté validateur.

---
## 📋 FIX POTENTIELS — état au 05/09 10:30 UTC (pour le prochain merge / la prochaine session)

**Bilan nuit 04→05/09 (511 fen mûres 41763-42273)** : 1,08 payée/fen, 74 % fen
payées, rang 16 (part 3,39 %), présence payée 80 % — contre 0,52 / 40 % / rang 28 la
veille. Les 3 fix du 04/09 (mémo de tête + veto brûlés → préchargement hors boucle →
mémo qui apprend des verdicts) sont validés sur R2 ; TOUS sur la box, NON COMMITÉS
(worktree `fix/course-2026-08-27`, + le terme de bande volume inerte du 03/09).

**À commiter au prochain merge** (box == worktree, md5 vérifiés) :
`reliquary/miner/payable_memo.py` (top_in_range, confirmations, run courant),
`reliquary/miner/engine.py` (burned_idx_load/_burned_active, memo_head_pick,
_picked_this_window, sz_blacklist_ban + verdicts→mémo, hf_latest_commit_only wiring,
band volume inerte), `reliquary/miner/checkpoint_prefetch.py` (to_thread + model_info),
`ops/launch_miner_v4.sh` (MEMO_SLOT=1, MEMO_HEAD_SLOTS=2, MEMO_RUN_START=32791,
MEMO_MIN_SCORE=0, BURNED_IDX), `scripts/refresh_burned_idx.py` (+ cron dev box */15),
`scripts/memo_r2_compare.py`, tests : test_payable_memo, test_burned_idx,
test_memo_verdict_ban, test_checkpoint_prefetch (+2), test_prompt_scores_precalcul.
`data/prompt_digests.npy` (80 Mo) et `data/burned_idx.npy` : NE PAS commiter (dérivés,
regénérables : build_digests.py sur la box 286 s ; refresh_burned_idx.py).

**Pertes restantes (134 fen non payées / 511) et fix candidats, par ordre :**

| # | part | fix | état | gain attendu |
|---|---|---|---|---|
| **1** | **31 %** | **Tête trop légère** (arrivée <9 s, bucket < barre de ~8 = ~800 tok). Stocker le volume observé (Σ completion_lens) dans la table mémo, trier les têtes : run courant > volume ≥ 9 000 > confirmations > fraîcheur. Mesuré : volume répétable Spearman 0,75 ; ≥9 000 observé → 98 % passent bucket 84 au round 2 (87 % sans filtre), garde 79 % du stock (~15-20/tranche). | À CODER (30 lignes + test, 1 restart) | +0,05-0,08/fen |
| 2 | 10 % | **Tête rejetée out_of_zone par le validateur** (1re rencontre d'un désaccord de correcteurs ; 103/nuit, 18 avant 9 s). Lire la règle du validateur (mémoire d'une autre session : « cas technique en échec = rollout 0 », NON vérifiée en source) et l'appliquer localement → ne plus envoyer ces groupes du tout. | À INVESTIGUER (source validateur code_grader) | +0,02-0,03/fen |
| 3 | 26 % (13 % direct) | **Checkpoint** : 27 reloads/nuit (1/31 min), fenêtre suivante détectée à 42 s. Structurel : prefetch déjà actif (download instantané), reste vLLM restart 10 s + compile 13 s + graphes 7 s + warmup. Hot-swap NO-GO (gate FS_GRAPH ≠ reload_weights_inplace). Piste : baker moins tôt après reload ? mesurer d'abord le détail des 42 s. | STRUCTUREL | ? |
| 4 | 16 % | **stale_round sur la tête** : réponses validateur 5-10 s à l'affluence (endpoint precommit p99 7 s), chaîne locale 0,7 s. Piste : tir de couverture (re-precommit au round suivant si pas de réponse à ~2,5 s) — vérifier en source si un 2e precommit même prompt est accepté/doublon. | À CONCEVOIR, risqué | +0,03/fen max |
| 5 | 5 % (17 % des groupes) | **Gate d'auth locale des tokens** : écarte 629+142 groupes/nuit pour 1 token_tampered réel. Mais = règle enforcée du validateur (1e-5/0,99). Devient un levier SEULEMENT si `origin/fix/v5-seed-auth-false-positive` (ALL_TOKEN_AUTH_ENFORCE = PROTOCOL_VERSION != 5) est mergée → alors RELIQUARY_LTA_CHOSEN_MAX 1e-5 → 1e-8 (hard seul). | VIGIE UPSTREAM | +0,05/fen si merge |
| 6 | — | **/state à chaque tour** : 1,16 Mo, 141k cooldown idx validés par pydantic + set() = 32 % du thread principal, ~0,3 s/tour. Poll léger pour le flip (sans cooldown) + cooldown toutes les 20 s. | À CODER (petit) | détection −0,2 s |
| 7 | — | **Alimentation du mémo** : solde ≈ −450/nuit (+286 nouveaux en zone, −560 payés, −112 bannis) sur 57k propres ; memo_hits/tranche méd 96, min 16. Si < 30 : plus d'exploration au balayage ou prior ré-entraîné sur les 140k textes sélectionnés (R2). | SURVEILLER (`memo_hits` dans windows_v4.jsonl) | — |
| 8 | 2 % | same_prompt_superseded (loterie forced-seed) | NON FIXABLE | — |

**Vigies** : seed_mismatch 0, token_tampered 1/nuit, logprob_mismatch 2/nuit,
worker_dropped 54/nuit en vagues (leur correcteur), tracebacks = rename bénin
`sz_blacklist.json.tmp` (à corriger un jour : écrire le tmp dans le même répertoire,
ignorer FileNotFoundError). Upstream : `integration/reliquary-v1-final` (+170, 05/09) —
nouveau, à lire ; `fix/v5-seed-auth-false-positive` = déclencheur du fix 5.
