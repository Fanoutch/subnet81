# Passation — étude R2 du 08/09 (mineur uid 167, subnet 81)

But de la session : comparer notre mineur au marché via le bucket R2 du validateur
(clé lecture seule dans `.env.r2`) et trouver où agir. Ce fichier remplace la
relecture du fil : tout ce qui est ci-dessous est **mesuré**, avec la taille
d'échantillon. Le reste du contexte projet est dans `CLAUDE.md`.

---

## 0. Démarrage à froid — tout ce qu'il faut avant de toucher à quoi que ce soit

### Accès
```bash
ssh root@157.10.162.245 -p 20301        # la box (H200). Le port change à chaque reboot Lium.
#   journal   /workspace/miner.log                (TRONQUÉ à chaque restart)
#   dumps     /workspace/submits_v4.jsonl         (1 ligne par tir)
#             /workspace/samples_v4.jsonl         (277 k groupes, champ completion_lens)
#             /workspace/windows_v4.jsonl, verdicts_v4.jsonl
#   relance   bash /workspace/restart_miner.sh    (relance AUSSI watchdog + monitor)
#   tmux      miner · watchdog81 · monitor
curl -s http://209.20.157.231:8080/health        # le validateur (protocol_version, profil, status)
```
Wallet `camille81-v2` / hotkey `hotkey81`, **uid 167**,
SS58 `5DvpFN3QEa9iimQiA5jQaRmx8dbW2uxonM53j51Cw3kBva7q`.
Clés R2 (lecture seule) dans `/root/subnet81/.env.r2` — **jamais commitées**.
Code de prod : branche `fix/course-2026-08-27`, box == GitHub au commit **`6d77f11`**
(vérifié le 09/09 : 59/59 fichiers `reliquary/**/*.py` md5 identiques, arbre propre, poussé).
🪤 **La source de prod est le WORKTREE `.worktrees/miner-priv-port-v4-dapo`, PAS le checkout
principal `reliquary-miner-priv/`** — celui-ci est resté sur une vieille branche
(`feat/predicteur-tfidf-k2` @ `137173d`) avec des modifs non commitées. Un `rsync` depuis
là déploierait le mauvais code sans prévenir.

### Vérifier l'état en 30 secondes
```bash
ssh -n root@157.10.162.245 -p 20301 "date -u; ps -o lstart= -p \$(pgrep -f 'reliquary.cli.main mine'|head -1); \
  grep -ac Traceback /workspace/miner.log; tail -3 /workspace/miner.log"
ssh -n root@157.10.162.245 -p 20301 "tr '\0' '\n' < /proc/\$(pgrep -f 'reliquary.cli.main mine'|head -1)/environ | grep RELIQUARY_ | sort"
```

### Les invariants du protocole (vérifiés en source, ne pas re-dériver)
- **`bucket = min(tokens, cap) // ((round_arrivée − round_ouverture) × 50)`, SANS +1** —
  vérifié 33 408/33 408 sur les payées.
- Clé de tri = `(−valeur, throughput_rank, tiebreak)` avec `throughput_rank = −bucket`.
  **La valeur est 1,0 pour tout le monde** (enchère plate) ⇒ **le bucket décide**, puis
  une loterie drand entre ex aequo.
- **16 sièges payés par env et par fenêtre.** La « barre » = le bucket du 16e siège.
- drand quicknet : `round(t) = floor((t − 1692803367)/3) + 1`, période 3 s,
  **tolérance arrière ZÉRO** ⇒ `stale_round`.
- Frontières de round réelles ≈ **2,5 s + 3k** : être au round 2 = **arriver ≤ 8,4 s**.
- Le rang est estampillé à l'arrivée du **precommit**, pas du corps.
- Porte de zone locale : `sigma ≥ 0,24` sur les 16 rewards. Le validateur est
  **authoritative** sur les rewards code.
- Forced-seed v5 : les tokens sont imposés par (randomness, prompt_idx, checkpoint,
  rollout) ⇒ **on ne choisit pas la longueur d'un groupe, seulement le prompt**.
  ⛔ Interdits : quantization et décodage spéculatif (cassent `seed_consistency`).

### Schéma des archives R2 (`reliquary/dataset/window-<N>.json.gz`)
`window_opened_wall_ts_by_environment` (l'ouverture, référence de toutes les arrivées) ·
`difficulty_auction[env].candidates[]` = **toutes les admissions en zone** avec
`throughput_rank` (= −bucket), `precommit_arrival_ts`, `arrival_drand_round`, `status`
(selected / not_needed / proof_failed / content_in_cooldown / same_prompt_superseded),
`proof_duration_seconds`, `body_read_ms` · `batch[]` = les **payées**, avec par rollout
`reward`, `completion_length`, `completion_text` · `rejected[]` (hors zone, doublons,
échecs de vérif) · `rewards_by_hotkey`.

### Règles de travail — elles ont toutes été apprises à leurs dépens
1. **Jamais de restart du mineur sans go explicite de l'utilisateur.** Expliquer quoi
   et quel effet AVANT de modifier.
2. **Un changement à la fois → 30-40 fenêtres mûres → verdict R2 → suivant.** Un
   correctif entre-temps est un patch, pas un nouveau bras.
3. **Fixer le seuil de repli AVANT le test**, et le tenir (c'est ce qui a sauvé le
   sprint 3 en 31 minutes).
4. **Vérifier contre le CODE, pas contre `CLAUDE.md`** — les affirmations du journal
   sont des hypothèses jusqu'à la ligne qui les utilise.
5. **Lire la BARRE avant de conclure à une régression** (cf. §1).
6. Ne jamais annoncer une tendance sur une tranche horaire (n≈17) : exiger plusieurs
   tranches consécutives hors amplitude.
7. `git status` + md5 box/worktree **avant et après** chaque déploiement.

### Par où commencer une session neuve
```bash
bash /root/subnet81/scripts/check_reliquary_updates.sh      # vigie upstream (obligatoire)
curl -s http://209.20.157.231:8080/health | head -c 400     # le validateur a-t-il basculé ?
# puis, pour un verdict marché :
cd /root/subnet81 && python3 scripts/r2_etude/tete_h.py <fen_debut> <fen_fin>
```
Ordre de lecture : **ce fichier** → §9 pour ne pas refaire un fix rejeté → `CLAUDE.md`
seulement si un protocole de mesure ou un repli précis est nécessaire.

---

## 1. Le résultat central — ne pas repartir sur un faux diagnostic

**Notre mineur n'a rien perdu. Le marché a grossi de 53 %.**

Comparaison de NOS indicateurs absolus sur 5 ères / 2 279 fenêtres :

| ère | fen | pay/f | rang | r1/f | r2/f | r3/f | arr r2 | vol tête | buck | BARRE | cand/f |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 04→05/09 nuit | 502 | 1,09 | 8 | 0,01 | 0,89 | 0,42 | 6,9 s | 9 750 | 87 | 76 | **55** |
| 05→06/09 | 430 | 0,91 | 11 | 0,02 | 0,88 | 0,35 | 6,8 s | 9 563 | 85 | 82 | 64 |
| 06→07/09 | 403 | 1,06 | 11 | 0,01 | 0,85 | 0,43 | 7,0 s | 9 350 | 82 | 74 | 67 |
| 07→08/09 | 853 | 1,07 | 9 | 0,00 | 0,88 | 0,45 | 7,0 s | 9 233 | 82 | 71 | 71 |
| 08/09 après-midi | 91 | 0,78 | 10 | 0,02 | 0,85 | 0,41 | 6,9 s | 9 040 | 81 | 80 | **84** |

Entrées au round 2, arrivée à ce round, volume : **plats**. Seul `cand/fen` bouge.

⚡ **NOUS SOMMES MARGINAUX PAR CONSTRUCTION** : notre bucket de tête (81-87) est posé
**sur** la barre du 16e siège. barre 71/74/76 → 1,06-1,09 payée ; barre 80/82 → 0,78-0,91.
**Règle de lecture : avant de conclure « le mineur a régressé », lire la BARRE.**
Trois fois sur cinq, la variation des payées vient entièrement d'elle.

Cause de la densification (08/09 14h) : **le validateur a accéléré** — precommit
aller-retour 0,98 → 0,48 s (p90 4,80 → 1,03), corps 1,66 → 1,36, nos `stale_round`
6,8 % → 2,1 %. Tout le peloton gagne ~0,5 s et monte d'un round. 10 hotkeys neuves
au round ≤2 depuis 14h (+3,3 entrées précoces/fen), titulaires +2,5.

---

## 1bis. SESSION DU 09/09 — la cadence du validateur a doublé, et le diagnostic du §1 se précise

### Le fait extérieur : fenêtres 90 s → 190 s le 08/09 à midi
Vérifié par **trois sources indépendantes** (horodatages d'ouverture R2, notre
`submits_v4.jsonl`, et `cadence_ms` du dashboard reliqua.ai) sur **24 heures
consécutives** : p50 174-210 s, aucun retour à la normale. Distribution **large** :
p10 ≈ 135 s, p50 ≈ 190 s, p90 ≈ 280 s (min 102, max 393) — donc **aucun seuil calé en
dur sur la médiane n'est valide**. Fenêtres/h : 40 → 18,5.
**Cause** (`/health`) : `training_accumulator_ready: True`, counts 16/16 = targets,
`training_trained_windows_since_publish: 0` ⇒ **contre-pression du trainer du
validateur**. Ni déploiement ni redémarrage côté validateur (image `84dcc571`
inchangée, process up depuis 11 jours). L'enchère ferme toujours à 100 s max
(`auction_collection_ceiling_seconds`), donc ~90 s de trou entre deux fenêtres.
⚠️ **Économiquement NEUTRE sur l'incentive** : la part est normalisée PAR FENÊTRE.
Elle rend seulement le compteur « payées/HEURE » trompeur (−60 % alors que la part
n'a perdu que 21 %). **Ne jamais conclure sur un compteur par heure.**

### La baisse d'incentive est la BARRE, pas nous — décomposition par slot
Incentive en chaîne le 09/09 : **uid 167, rang 12/50 actifs, 0,02914 = 2,92 %** —
cohérent avec la part R2, qui est donc un proxy valide.
Décomposition sur l'horloge du validateur, 4 ères de ~500 fenêtres :

| ère | payées/fen | g1 arr | g1 bucket | **g1 payée** | g2 arr | g2 bucket | **g2 payée** | **BARRE** |
|---|---|---|---|---|---|---|---|---|
| 06→07/09 | 1,09 | 7,3 s | 82 | **65 %** | 10,0 s | 64 | **41 %** | **72** |
| 07/09 | 1,03 | 7,4 s | 81 | 60 % | 10,2 s | 67 | 41 % | 72 |
| 08/09 matin | 0,84 | 7,2 s | 79 | 54 % | 11,0 s | 58 | 32 % | 77 |
| 08/09 midi → 09/09 | 0,82 | 7,2 s | 80 | **53 %** | 10,1 s | 63 | **33 %** | 76 |

**Notre arrivée et notre bucket ne bougent pas.** Seule la barre monte (72 → 76) et
notre taux de paiement s'effondre (g1 65 → 53 %, g2 41 → 33 %). **+4 points de barre
= −0,27 payée/fen = −21 % d'incentive**, parce que notre bucket est posé SUR la barre.
La baisse a commencé le 08/09 **matin**, donc AVANT la bascule de cadence : les deux
événements sont indépendants.

### Le prix d'une seconde d'arrivée — contrefactuel sur barres RÉELLES (437 fen)
| arrivée gagnée | payées/fen | Δ | g1 au round 1 | g2 au round ≤2 | part d'émission |
|---|---|---|---|---|---|
| — | 0,89 | — | 0 % | 21 % | 2,78 % |
| **0,5 s** | 1,19 | **+0,30** | 15 % | 36 % | **3,72 %** |
| **0,7 s** | 1,24 | **+0,35** | 21 % | 38 % | **3,87 %** |
| 1,0 s | 1,31 | +0,42 | 32 % | 43 % | 4,10 % |
| 1,7 s | 1,40 | +0,51 | 52 % | 48 % | 4,38 % |
| 2,5 s | 1,45 | +0,56 | 62 % | 49 % | 4,52 % |
⚡ **La courbe est raide puis plate : les 0,7 premières secondes rapportent les deux
tiers du gain total** et ramènent la part au-dessus du niveau du 07/09. Chaque seconde
compte double (g1 vers le round 1, g2 vers le round 2). Réserves : le marché est figé
(la barre pourrait monter en réaction) et le volume est inchangé (correct : sous
forced-seed, décoder plus vite donne les mêmes tokens plus tôt).

### Budget de détection du flip — ~0,7 s inexpliquées, la cible la moins chère
Mesuré depuis la box : `/state` sur connexion **chaude** = **0,36 s** (gzip DÉJÀ actif,
1,64 Mo → 650 Ko), parse JSON + pydantic = **16 ms** (hypothèse « le parse coûte cher »
**testée et RÉFUTÉE**). Le poll est en boucle serrée (5 ms) ⇒ détection prédite 0,57 s.
Détection **observée ~1,0-1,3 s** (16 fenêtres, log × R2, granularité log 1 s).
⇒ **~0,7 s ne sont ni le réseau ni le parsing.** À caractériser (contention de la
boucle d'événements ?). C'est le seul levier d'arrivée qui **ne coûte rien en bucket**.
- `/state` = 1,64 Mo dont **99,9 % de `cooldown_prompts`** ; `randomness` est le
  **DERNIER champ** (offset 1 643 142 / 1 643 222) ⇒ **aucune sortie anticipée possible**
  par lecture en flux. `state`/`window_n` sont eux aux offsets 1 et 16.
- **`/miner-state` existe en amont** (cooldown en bitmap, ETag, 304 conditionnel) mais
  répond **404** sur l'image en vol : il arrive avec la bascule. Il ramènerait le poll
  de 0,36 s à ~200 octets.
- **Sous v6/fill-closed la randomness devient PUBLIQUEMENT DÉRIVABLE** :
  `_bind_public_window_randomness(beacon, window_n)` (service.py:3118-3136), contre
  `_bind_window_activation_randomness` + nonce secret `os.urandom(32)` en v5. Le jour
  de la bascule, on calcule la randomness localement et on n'attend plus le validateur.

### ⛔ DEUX PISTES OUVERTES PUIS RETIRÉES DANS LA MÊME SESSION
- **Re-caler la garde pré-flip** (`PREFLIP_GUARD_S` 50 → 150, `LATE_BAKE_FROM` 35 → 135)
  pour récupérer les ~110 s de GPU inactif par fenêtre : **RETIRÉE**. La tranche de
  prompts est retirée au hasard à chaque fenêtre (5 000 index sur 2,48 M) —
  **0 recouvrement sur 301 paires de fenêtres consécutives**. Rien de baké pendant la
  fenêtre N ne peut servir à N+1 : le GPU inactif est **structurellement inutilisable**,
  pas de la capacité perdue.
- **`batch_filled` (0,02 → 0,3-0,5/fen après la bascule)** : **sans effet sur le revenu**.
  Il ne touche **0 %** de nos entrées payantes (<8,4 s) ; il ne mord que sur la bande
  8,4-27 s, c'est-à-dire les rounds 3-9, qui ne paient presque rien.

### 🪤 Piège de mesure neuf
**`flip_offset_s` du dump est estampillé APRÈS la réponse du POST** (engine.py:4160,
au même point que `t_post`), pas à l'arrivée. Le validateur ayant accéléré le 08/09
(precommit A/R 0,98 → 0,48 s), ce champ s'est déplacé **sans que rien change chez nous** :
une comparaison avant/après sur `flip_offset_s` fabrique un faux progrès. **Pour tout
ce qui touche au round, utiliser l'arrivée R2 (`precommit_arrival_ts`), jamais ce champ.**

---

## 2. Où est réellement l'écart avec les meneurs

Détail par round (91 fen, 08/09 après-midi ; `n/f · payées/f · arrivée`) :

| hotkey | round 1 | round 2 | round 3 |
|---|---|---|---|
| **nous** | 0,02 · 0,02 · 4,8 s | **0,85 · 0,60 · 6,9 s** | 0,41 · 0,11 · **9,1 s** |
| 5GxSiK (2e du classement) | 0,04 · 0,03 · 5,6 s | **1,43 · 1,12** · 7,4 s | 0,60 · 0,19 · 10,3 s |
| 5CAmEgH8 | 0,38 · 0,37 · 5,0 s | 1,30 · 0,65 · 6,8 s | 0,86 · 0,07 · 10,2 s |
| 5E5E3SZV | 0,38 · 0,37 · 5,2 s | 1,25 · 0,64 · 6,9 s | 1,03 · 0,09 · 10,0 s |
| 5DPeiThb | 0,35 · 0,35 · 5,3 s | 1,18 · 0,53 · 6,7 s | 1,09 · 0,12 · 9,9 s |

Deux faits qui orientent tout :
- **Notre arrivée au round 2 est la leur** (6,9 s contre 6,7-7,4). On n'est pas lents
  sur ce qu'on y place. Ce qui manque est le NOMBRE : 0,85 contre 1,18-1,43.
- **Notre conversion au round 2 est la 2e meilleure** (0,60/0,85 = 71 %, contre 45-59 %
  pour le trio) — grâce à notre volume supérieur. On ne perd pas là où on est présent.

⇒ **Le seul chiffre qui résiste à toutes les vérifications : notre 2e entrée arrive à
9,1 s pour une frontière de round à 8,4 s.** Elle est payée 11 % au lieu de 60 %.
Cause décomposée : **décodage** (écart de disponibilité g1→g2 = 1,48 s p50, p90 6,89 s),
**pas la chaîne** (+0,50 s seulement : +0,12 preuve, +0,21 precommit A/R).
`HEAD_FIFO` innocenté : g2 est prêt avant la fin de la preuve de g1 dans 37 % des cas.

---

## 3. Pistes FERMÉES par la mesure — ne pas re-tester sans raison neuve

| piste | verdict | preuve |
|---|---|---|
| **Mémo qui s'érode** | ⛔ hors de cause | volume de tête, round 2 et arrivée 7,0-7,9 s plats sur 15 h. Une érosion se verrait en tête jetée localement → 1re entrée à 9-10 s. Absent. |
| **Matériel des meneurs** | ⛔ identique | durée de preuve **validateur** 1,66-1,80 s pour les **13** mineurs ; nous 1,80 = −0,1 s. |
| **Génération différente** | ⛔ identique | test apparié même fenêtre / MÊME prompt_idx (38-285 paires/concurrent) : leur volume − le nôtre = −39 à +64 tokens, « eux plus court » 43-54 % = pile ou face. |
| **Hors-zone** | ⛔ déjà optimal | nous 0,00/fen, comme les meilleurs. |
| **Sprint 3** | ⛔ testé, replié | 08/09 13:24→13:55. g1 prêt 3,30 → 4,50-4,90 s. La carte est **saturée à 32 séquences** : +50 % de séquences = +10 % de débit total et +36-48 % de latence par séquence. |
| **Precommit avant la preuve** | ⛔ interdit par le protocole | `_build_precommit` (submitter.py:294-303) signe `merkle_root`, dont la feuille lie `commit` = la commitment GRAIL (`protocol/merkle.py:46`). |
| **Copier le volume du marché (8 400)** | ⛔ **négatif** | simulation 882 fen, barres réelles : **−0,05 payée/fen**. Voir §4. |
| **Re-caler la garde pré-flip sur la cadence 190 s** | ⛔ **sans objet** (09/09) | la tranche de prompts est retirée au hasard chaque fenêtre : **0 recouvrement sur 301 paires consécutives**. Le GPU inactif est structurellement inutilisable, pas de la capacité perdue. |
| **`batch_filled` (0,02 → 0,3-0,5/fen)** | ⛔ **sans effet revenu** (09/09) | touche **0 %** de nos entrées <8,4 s ; ne mord que sur la bande 8,4-27 s (rounds 3-9, quasi non payés). |
| **Le parse de `/state` coûte cher** | ⛔ **réfuté** (09/09) | json.loads + pydantic = **16 ms** sur 1,64 Mo. Le poll chaud coûte 0,36 s, gzip déjà actif. |

---

## 4. Le volume : arc complet (rouvert, réfuté, puis optimum trouvé)

### 4.1 Ce qui est vrai et qu'on ignorait
- **Le volume total se sélectionne** : répétabilité par prompt **r = +0,920**
  (split-half, 434 prompts vus ≥4 fois, 11 883 occurrences ; σ inter-prompts 1 616 tok).
  Le volume visé est **obtenu à 1 % près** (prédit 7 813 → réalisé 7 753, etc.).
- ⚡ **2 corrections du 03/09** : « la sélection ne contrôle pas la longueur » est vrai
  pour le **traînard** seulement. Le traînard n'est pas nul non plus : split-half
  **+0,296** par occurrence (+0,456 sur la moyenne), corr(volume, traînard) **+0,616**.
- **Coefficient physique, 2 méthodes concordantes** :
  **+5,5 à +7 s d'arrivée pour 1 000 tokens de traînard** (appariement intra-vague
  n=4 116 ; quintiles de la 1re entrée : traînard 627 → arrivée 7,00 s, 1 036 → 9,90 s).
- **Le mémo est le véhicule** : `payable_memo.best_in_range()` (payable_memo.py:69-77)
  renvoie l'ex-payable **LE PLUS FRAIS** — aucun critère de volume — parmi **65 606
  payables** (~155 candidats par tranche de 5 000). `update()` (engine.py:900) marque
  payable = `in_zone AND n_truncated==0`.
- Coût en zone d'un raccourcissement (ex-payables, quintiles) : re-zone **82,4 %** (Q3,
  nous) → **82,3 %** (Q2, 9 000) → **77,3 %** (Q1, 7 800). Nul jusqu'à 9 000.

### 4.2 Réfutation du levier GLOBAL
- 6 831 de nos entrées admises par quintile de traînard : 654 → 1 170 tok coûte **UN
  round** (6→7) et apporte **+48 % de volume** ⇒ bucket 26 → 36.
- **Test apparié intra-fenêtre** (1 198 fen, notre entrée la plus courte contre la plus
  longue de la MÊME fenêtre) : la longue a un bucket **+6,7** et gagne au paiement
  **230 fois contre 160**.
- Raison : 80 % de nos entrées sont au round 6, où une seconde ne vaut rien.

### 4.3 L'optimum sur les DEUX TÊTES (simulation 882 fen, barres réelles)
`scripts/r2_etude/court.py`

| têtes ramenées à | payées têtes/fen | bucket moy | r≤2/fen | r1/fen | Δ |
|---|---|---|---|---|---|
| réel (~9 900) | 1,10 | 69,6 | 0,90 | 0,00 | — |
| **9 000** | **1,21** | 70,5 | 1,16 | 0,03 | **+0,11** |
| 8 600 | 1,10 | 70,0 | 1,21 | 0,07 | +0,01 |
| 8 400 (marché) | 1,05 | 70,6 | 1,23 | 0,11 | −0,05 |
| 8 000 | 0,96 | 71,7 | 1,27 | 0,21 | −0,13 |
| 7 500 | 0,88 | 73,0 | 1,30 | 0,34 | −0,22 |

**Optimum à ~9 000, gain plafonné à +0,11 payée/fen (~+10 %).**
Pourquoi ça retombe ensuite : le bucket **moyen** monte alors que les payées baissent —
à 7 500 on achète 0,34 entrée au round 1 (bucket ~150 contre une barre à 83), donc on
**sur-paie** le round tout en faisant passer sous la barre des entrées qui la
franchissaient au round 2. Le volume des meneurs est calibré pour LEUR position.

⚠️ **Réserves** : la simulation fait décroître le traînard **proportionnellement** au
volume alors que la corrélation réelle n'est que +0,616 ⇒ **+0,11 est une BORNE HAUTE**.
Elle ne compte que les 2 têtes. Forme du patch le jour venu : trier `best_in_range` par
volume observé le plus proche de **9 000** au lieu de la fraîcheur, **sur les seuls
slots de tête** (les slots tardifs sont insensibles à l'arrivée, le volume y domine).

---

## 5. Piste ouverte non instruite : la vitesse de décodage

Non mesurée directement, à confirmer par un profil sur la box avant tout travail.
Chaînage de deux mesures séparées, donc **à vérifier** :
- décodage effectif ≈ **161 tok/s par séquence** (déduit du coefficient 6,2 s/1000 tok)
  ⇒ un pas ≈ **6,2 ms** ;
- la gate GPU mesure la passe forced-seed à **1,181 ms à 32 séquences**
  ⇒ **~19 % de chaque pas de décodage**.

Si ça se confirme, c'est le plus gros surcoût identifié. Piste concrète : `warp_fast`
(tri restreint au top-k) est **bit-exact** et n'avait été abandonné que pour un
`nonzero` forçant une synchronisation GPU — or un CUDA graph ne peut pas contenir de
sync, donc le graphe actuel tourne forcément sur le tri complet. Réécrire ce chemin
sans sync = chantier avec un prix mesurable.
Réglages jamais testés : `RELIQUARY_VLLM_MAX_NUM_SEQS` **256** (la table de fixes
suggère 96), `VLLM_GPU_FRACTION` 0,76 (mais le cache KV a été mesuré sans effet en
juillet). ⛔ Interdits : quantization et décodage spéculatif cassent le forced-seed.

⚠️ **La vitesse achète de l'ARRIVÉE, pas du VOLUME** — le forced-seed décide de l'EOS ;
décoder plus vite donne les mêmes tokens plus tôt. Les deux leviers se combinent :
le mémo va chercher le volume, la vitesse en paie le temps.

---

## 6. Outils réutilisables

**Scripts** (copiés dans `scripts/r2_etude/`, tous en lecture seule sur R2 + dumps) :

| script | ce qu'il sort |
|---|---|
| `tete_h.py D0 D1` | par heure : payées/fen, arrivée/round/volume/bucket de notre tête, barre, top8, part >barre |
| `marche.py D0 D1` | par heure : cand/fen, entrées et hotkeys au round ≤2, volume top-8, barre |
| `rang.py D0 D1` | classement du marché : payées/fen, r1/f, r2/f, ent/f, arrivée, volume |
| `rounds.py` | entrées PAR ROUND (n/f · payées/f · arrivée) pour nous et 6 concurrents |
| `nouveaux.py` | qui est apparu au round ≤2 entre deux périodes |
| `eres.py` | comparaison de NOS indicateurs sur 5 ères (le tableau du §1) |
| `apparie.py` | test apparié même fenêtre / même prompt_idx contre chaque concurrent |
| `repet.py` | répétabilité split-half du volume par prompt |
| `interne.py` | nos entrées par quintile de traînard + apparié intra-fenêtre |
| `chaine.py` | décomposition horodatée g1 contre g2 (dump `submits_v4.jsonl`) |
| `court.py` | **simulation** têtes ramenées à un volume cible (le tableau du §4.3) |
| `g2_gain.py` | simulation ciblée sur la 2e entrée |
| `vol_eres.py`, `qui.py`, `diag.py`, `vol_cause.py`, `tete.py`, `flip.py` | variantes |

Ils importent `charger()` de `scripts/r2_benchmark_miners.py`. ⚠️ `charger()` décompresse
**tous** les fichiers du cache : pour aller vite, faire un cache filtré par liens durs
(`ln`) sur la plage voulue.

**Caches R2** (`data/`) : `r2_cache_0904` 39275-45126 (5 824 fen) · `r2_cache_eras`
41781-45126 (2 281) · `r2_cache_win` 43900-45308 (1 409).
Rapatriement : `scripts/r2_pull_windows.py --debut N --fin M --out data/r2_cache_0904`
avec `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` depuis `.env.r2`.

**Dump mineur** : `rsync -z -e "ssh -p 20301" root@157.10.162.245:/workspace/submits_v4.jsonl`.
Le corpus `/workspace/samples_v4.jsonl` (277 k groupes) porte **`completion_lens`** =
longueurs par rollout — c'est lui qui donne volume et traînard par prompt.

🪤 **Pièges de mesure vécus dans cette session** :
- dans `submits_v4.jsonl`, un re-tir réécrit la même ligne avec le MÊME `ts` ⇒ compter
  le 1er tir par **ordre de fichier** ;
- ne jamais lire une tendance sur une tranche d'une heure (n≈17) : le « volume de tête
  qui baisse » du 08/09 16h était un artefact d'échantillon, démenti à 3 h ;
- `g1`/`g2` définis par ordre de disponibilité ⇒ « g2 a un traînard plus long » est vrai
  **par construction**, ce n'est pas une découverte.

---

## 7. État du mineur et vigie

- Mineur en vol depuis le **08/09 20:25:07** (déploiement de `6d77f11`), 0 Traceback bloquant :
  `SPRINT_SIZE=2` · `MEMO_HEAD_SLOTS=2` · `BAKE_BATCH_SIZE=5` · `HEAD_FIFO=2` ·
  `MAX_INFLIGHT_FIRES=3` · `LOCAL_TOKEN_AUTH=1` · `STATE_POLL_TIMEOUT_S=10` ·
  `FS_GRAPH=1` · `VLLM_MAX_NUM_SEQS=256` · `VLLM_GPU_FRACTION=0.76`.
- **Point de retour vérifié le 09/09 22:30 : `ops/CONFIG_LIVE_2026-09-09.txt`** — commit,
  md5 des 59 `.py`, des 3 scripts d'exploitation, des 6 fichiers non versionnés et de leur
  sauvegarde, procédure de repli en 5 étapes, et les 50 variables d'environnement exactes.
  Tout est commité et poussé ; `samples_v4.jsonl` (corpus mémo) suit par cron, 17 lignes de retard.
- **Déploiement non consigné retrouvé le 09/09** : `6d77f11` « preuve GRAIL : 48 → 3 transferts
  GPU→hôte par groupe », commité 08/09 20:24:22 et déployé 20:24:50 par une autre session.
  **Jugé le 09/09 : NEUTRE** — preuve p50 1,120 → 1,080 s (p90 1,79 inchangé), chaîne
  prêt→precommit 2,42 → 2,51 s. Sans effet adverse, gardé comme base de repli.
- 🔭 **Upstream : PR #224 mergée dans `main` le 09/09** (`0be0cda`,
  « integration/reliquary-v1-final ») — c'est la bascule qu'on surveillait.
  **Le validateur en vol est ENCORE en v5** (`84dcc571`, profil
  `qwen3-4b-base-dapo-reasoning-v5`, status ok) au 09/09 05:30 UTC. Notre port v6 existe
  (`ops/CUTOVER_V6.md`) mais n'est ni commité ni déployé. À re-vérifier à chaque session :
  `curl -s http://209.20.157.231:8080/health`.
- ⚠️ L'optimisation de timing de la chaîne est traitée dans une **autre session** (08/09).

---

## 8. Prochaines étapes — plan en 4 points, CHIFFRÉ (révisé le 09/09)

Cible : notre bucket (80) est posé SUR la barre (72-81), donc on est à pile ou face.
Le top-8 est à ~99. Le volume est fermé (on est déjà le plus volumineux, et le
simuler à la baisse est négatif, cf. §4). **Il ne reste que le ROUND**, et le
contrefactuel du §1bis en donne le prix exact.

**1 — Localiser les ~0,7 s de détection du flip** · 1 restart · gain **0**
Instrumenter la boucle de déclenchement : durée d'itération + étape, horodatage
réponse `/state` → détection → départ du bake. On sait déjà que ce n'est ni le réseau
(0,36 s chaud) ni le parsing (16 ms). *Porte* : si les 0,7 s se confirment et
s'attribuent → point 2 ; si elles se dissolvent, le levier n'existe pas → point 4.

**2 — Supprimer cette latence** · 1 restart · **+0,30 à +0,35 payée/fen (+34 à +39 %)**
Part d'émission 2,78 % → ~3,8 %, au-dessus du niveau du 07/09. Aucun coût en bucket.
**C'est le point qui vaut le voyage.** *Seuil de repli fixé À L'AVANCE* : arrivée p50
de la tête ne doit pas dépasser **7,5 s**. Verdict R2 à 30-40 fenêtres mûres.

**3 — La chaîne (2,13 s)** · 1 restart · **+0,10 à +0,12/fen** en marginal
`finalize` fusionné dans la preuve (−0,25 s) + round calculé avant le POST (diff déjà
écrit, flag `RELIQUARY_ROUND_AT_POST`). ⛔ Le precommit **ne peut pas** partir avant la
preuve (vérifié en source) : les 1,02 s de preuve sont sur le chemin critique.

**4 — Le décodage (3,4 s)** · chantier · **~+0,10** en marginal
D'abord un **profil d'un pas de décodage sur la box, SANS restart**, pour confirmer les
~19 % de la passe forced-seed — aujourd'hui c'est un chaînage de deux mesures séparées.
Si confirmé : `warp_fast` sans synchronisation GPU, bit-exact, gate forced-seed obligatoire.

⚡ **Le point 2 vaut plus que les points 3 et 4 réunis** (après 0,7 s, aller à 1,7 s
n'ajoute que +0,16 pour deux chantiers lourds). Faire 1 → 2, juger, et ne continuer
que si le gain marginal justifie encore le restart.

🔭 **Raccourci possible** : la bascule v6 apporte `/miner-state` et la randomness
dérivable — elle rendrait les points 1-2 gratuits. Date inconnue, ne pas miser dessus.

**Méthode** : un changement → 30-40 fenêtres mûres → verdict R2 → suivant.

---

## 9. Registre des fixes — déployés, rejetés, restants

Consolidé depuis `CLAUDE.md` (qui reste la source détaillée : chaque ligne y a son
protocole de mesure et son repli). Ici : le verdict et le chiffre.

### 9.1 Déployés et GARDÉS

| date | fix | effet mesuré |
|---|---|---|
| 01/09 | `detokenize=False` + FINAL_ONLY | −2,3 s |
| 01/09 | SPEC_PROOF (grading ∥ preuve GRAIL) | −0,8 s |
| 01/09 | **FS_GRAPH** (CUDA graph du forced-seed) + imputation des timeouts | g1 5,7 → 4,1 s ; **payées 0,54 → 0,96/fen** |
| 01/09 | Prior unique `(1−P(σ₀))⁸ × v5.9` | out_of_zone verdict 12,6 % → **2,1 %** |
| 24/08 | Table de scores pré-calculée + miroir parquet | classement 2,80 s → 0,0 s |
| **04/09 14:21** | **`MEMO_HEAD_SLOTS=2`** (2 têtes prises aux ex-payables) | tête au round ≤2 **37 % → 71 %** ; g1 jeté 36 % → 5 % ; **c'est LE fix qui a réglé la cause n°1** |
| 04/09 16:39 | Prefetch HF hors de la boucle (`to_thread` + `model_info`) | gel de boucle 3,83 s/15 s supprimé ; flips en retard 23 % → ~0 |
| 04/09 19:16 | Mémo apprenant des verdicts validateur | 0 récidive sur les ooz |
| **05/09 23:15** (`2079ecb`) | **Règle grader d'UNE ligne** : statut technique ≠ ok/bad_output ⇒ rollout à 0 | accord validateur **94,6 % → 99,87 %** ; **ooz validateur 105/nuit → 0** |
| **06/09 19:11** (`0db098c`) | **`MAX_INFLIGHT_FIRES` 6→3** + horodatages `t_sign`/`t_precommit_*`/`t_body_*` | **stale de la tête 11,4 % → 6,7 %**, rien perdu sur les entrées 4+ |
| **08/09 11:19** (`eaa70f1`) | **`STATE_POLL_TIMEOUT_S=10`** + échec de poll en WARNING | corrige un flip vu 22,5 s en retard pendant les téléchargements = **3,4 % du revenu** |
| 08/09 | `LOCAL_TOKEN_AUTH` remis à **1** | retour après test OFF non concluant |

### 9.2 Testés et REJETÉS — ne pas re-tester sans raison NEUVE

| date | fix | verdict |
|---|---|---|
| **08/09** | **Sprint 3** (`SPRINT_SIZE=3` + `MEMO_HEAD_SLOTS=3`) | **ÉCHEC NET, replié en 31 min.** g1 prêt 3,30 → 4,50-4,90 s (4,4 σ), sprint livré 4,50 → 6,60. Seuil de repli fixé AVANT : 4,3 s. Point mort de la courbe : 1,0 s. **3e échec de la même famille** (sprint 3 le 02/09, scan_holdoff le 03/09) ⇒ une 3e tête précoce demande une CARTE, pas un réglage. |
| 08/09 | Palier `FS_GRAPH` 48 | mesuré : graphe contre eager = 2 % à n=48 (20-30 ms sur un décodage). Ne vaut ni le restart ni 0,5 Go. |
| 07-08/09 | `LOCAL_TOKEN_AUTH=0` (gate douce OFF) | **NEUTRE, non prouvé** (838 fen vs 537). 708 ombres → **2 payées**, 62 `token_tampered`, 18 fen en `hotkey_proof_debt`. Remis à 1. ⇒ la gate DURE n'a plus lieu d'être testée. |
| 08/09 | `HF_XET_FIXED_DOWNLOAD_CONCURRENCY=2` | **RÉGRESSION, repliée.** Banc faux (97 s) : hf-xet déduplique contre le cache ; en prod 310-364 s contre 57-62 s ⇒ **2 fenêtres perdues par rechargement au lieu d'1**. 🪤 Ne jamais chiffrer un téléchargement HF sur un banc partageant des chunks. |
| 06/09 | Timeout de grading 1,0 → 0,5 s | **0 gain** (0,006 s) : sous SPEC_PROOF le grading est masqué par la preuve. |
| 06/09 | Headroom drand 1,0 → 1,8 s | **NÉGATIF −0,18 à −0,21/fen** : retarde d'un round 25 % des admises pour éviter 1 stale sur 10. |
| 03/09 | Bande de volume (malus vers ~8 800) | **INERTE** (traînard 864 vs 823). ⚠️ visait le traînard, pas le volume total — voir §4. |
| 03/09 | `SCAN_HOLDOFF` 2,5 s | g1 3,6 → 4,8 s pour gagner 1 s sur g3. Contention GPU réelle, pas un artefact d'ordonnancement. |
| 03/09 | Couvre-feu 80 s | **0 payée sur 486 envois tardifs mûrs.** |
| 03/09 | Garde 40/28 (`PREFLIP_GUARD`/`LATE_BAKE_FROM`) | contre-intuitif : têtes lentes 25 % → 40 %, payées 0,66 → 0,50. Repli 50/35. |
| 03/09 | `MEMO_SLOT=0` | non concluant (hash_duplicate 2,7 → 0,7 % ✅ mais payées 0,55 → 0,50). Gardé en fond, à re-trancher. |
| 08/09 | Rechargement de checkpoint (sauver la fenêtre +1) | **ABANDONNÉ (décision utilisateur).** Budget requis 1,5 s ; ni cache AOT (−8 s) ni hot-swap (5 s) ne suffisent — il faudrait un moteur de réserve pré-chauffé. |
| — | Hot-swap des poids | **NO-GO** : la gate FS_GRAPH ne couvre pas `reload_weights_inplace`. |
| — | Plancher de volume sur la tête · filtre de récence du mémo | réfutés par l'audit 4 agents du 03/09 |

### 9.3 Restants — suggérés, chiffrés, NON déployés

| # | fix | gain attendu | état |
|---|---|---|---|
| **N1** | **Tri du mémo par volume cible ~9 000 sur les 2 slots de tête** (au lieu de la fraîcheur) | **+0,11 payée/fen (borne haute)** | **NOUVEAU 08/09**, simulé sur 882 fen (§4.3). Coût en zone nul. Ne pas viser 8 400. |
| **N2** | **Profil d'un pas de décodage** puis, si confirmé, `warp_fast` sans sync | ~19 % du pas de décodage à confirmer | **NOUVEAU 08/09** (§5). Mesure d'abord, code ensuite. |
| 1 | Round calculé avant le POST + cache finalize au re-tir (`RELIQUARY_ROUND_AT_POST`) | +0,02-0,04/fen (150 fen pour le voir) | diff écrit, sûr |
| 2 | Lenteur de `/state` : 37 polls >10 s en 2,3 h, sans lien avec le téléchargement ni le flip | inconnu | **à caractériser AVANT tout correctif** |
| 3 | `stale_round` sur la tête : tir de couverture au round suivant | +0,03/fen max | risqué — vérifier EN SOURCE qu'un 2e precommit même prompt n'est pas un doublon |
| 4 | Résidu de starvation `/state` pendant téléchargement : brider la bande passante (tc/ifb) ou timeout 25-30 s | ≤3,4 %, partiel | ⛔ pas par le nombre de connexions (cf. 9.2) |
| 4c | `HEAD_FIFO` 2→0 | 0 à +0,02/fen | invisible à 30 fen |
| 4d | Bug `_sz_save` (`.tmp` manquant au rename, engine.py:2803) | 0 (propreté) | 10 Tracebacks/jour, bénins |
| 5 | Chaîne GPU : `max_num_seqs` 256→96, finalize fusionné dans la preuve (−0,25 s), marge drand 1,0→0,5 | petits | 1 restart chacun |
| 6 | Alimentation du mémo (solde ≈ −450/nuit) | — | surveiller `memo_hits` |
| 7 | Vigie upstream + port v6 (`ops/CUTOVER_V6.md`) | — | **PR #224 mergée le 09/09**, validateur encore en v5 |
| — | **STRUCTUREL : 2e carte** (mineur MATH ou décodeur dédié) | **+0,6-1,3/fen** | seul levier mesuré pour une 3e tête précoce |

⚠️ **Rappel de répartition** : g1 et g2 font **93,5 % du revenu** (45,5 % et 48,0 %) ;
g3/g4/g5 valent 1,9-2,1 % chacun. **Tout réglage qui ne touche que les slots 3+ pèse
moins de 7 %.**
