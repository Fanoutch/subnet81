# Subnet 81 (Reliquary) — custom miner project

Bittensor GRPO RL training subnet, netuid 81 mainnet (finney).
Wallet `camille81-v2` / hotkey `hotkey81` **ENREGISTRÉE** (uid 167, SS58
`5DvpFN3QEa9iimQiA5jQaRmx8dbW2uxonM53j51Cw3kBva7q`, keyfile régénéré 08-11).

**⚠️ Les affirmations de ce fichier sont des hypothèses** : vérifier contre le
code avant d'asserter un bug/gap (cf. mémoire feedback_verify_code_not_claudemd).

## 📋 TODO — au 25/08 14h, par ordre de valeur

1. **🔴 LE RECHARGEMENT DE CHECKPOINT COÛTE ~11 % DES FENÊTRES** (mesuré 25/08,
   115 fenêtres). **9 rechargements en 3 h 10 — un toutes les 21 min** ; la
   fenêtre qui SUIT l'avancée produit **0 groupe dans 5 cas sur 9**, reprise
   27-109 s. C'est aujourd'hui le plus gros levier mesuré, devant le prior.
   ⛔ **La note « rechargement = 56 s de blocage » est FAUSSE** : le blocage de
   la boucle vaut **0,2 s** (trou d'activité mesuré sur les 9 avancées).
   ✅ **Irréductible** : `checkpoint_hash` entre dans `u_at`, donc tout groupe
   généré sous l'ancien hash est un `SEED_MISMATCH` garanti — le vidage du pool
   (`engine.py:1974-1985`) est imposé par le protocole, pas optimisable.
   ✅ **HYPOTHÈSE CONFIRMÉE 25/08 14h30** : **12 rechargements depuis 10:04**
   (un toutes les **21 min**, cadence identique au 25/08 matin) pour **12 PID
   d'`EngineCore` DISTINCTS** — vLLM se réinitialise donc ENTIÈREMENT à chaque
   fois (poids en VRAM + graphes CUDA). Reste ouvert : vLLM expose-t-il un
   échange de poids à chaud ? **Non vérifié dans la source de vLLM.**
   📏 **Coût mesuré — le « ~11 % » de cette note est CONFIRMÉ à 10,6 %.**
   Décompte sur 838 fenêtres (31755→32593), **47 avancées** (une toutes les
   18 fenêtres), **1,89 fenêtre perdue par avancée** :

   | fenêtre à 0 acceptée | n | part |
   |---|---|---|
   | **invisible** (aucune ligne dans les 2 fichiers) | 48 | 5,7 % |
   | **reprise tardive** (revient, guichet plein) | 31 | 3,7 % |
   | aucune tentative (vue mais muette) | 5 | 0,6 % |
   | tout refusé (`wrong_checkpoint`/`window_mismatch`) | 5 | 0,6 % |
   | *autre cause, hors checkpoint* | *17* | *2,0 %* |
   | **→ imputable au checkpoint** | **89** | **10,6 %** |

   ⚠️ La ligne « reprise tardive » est imputée par PROXIMITÉ (0-3 fenêtres
   après une avancée datée), pas par une trace directe — c'est la seule des
   quatre qui repose sur une heuristique.
   **Séquence type** (32576-32578, 14:12→14:16, log `checkpoint advanced
   mid-iteration (reload stall)` à 14:14:46) : fenêtre en cours **0 acceptée**
   (13 `wrong_checkpoint` + 14 `window_mismatch`) → suivante **sans aucune
   trace** (trou d'activité 121 s) → 3e revient trop tard, `batch_filled`.
   Reprise pleine dès la 4e. Rejoué à l'identique sur 32592-32593 à 14:41.

2. **Pannes de disponibilité — 22 fenêtres sur 115 sans aucune acceptée** :
   12 sans génération (rechargement/blocage), 5 `batch_filled`, 3
   `window_mismatch`/`wrong_checkpoint`, 2 `stale_round`.
   ⚠️ **Timeout chaîne finney le 25/08 12:37 → 11 min, 7 fenêtres perdues.**
   Le watchdog ne l'a pas vu : son seuil de blocage est à **15 min**.
   Descendre à 8-10 min rattraperait ce cas, au prix de faux redémarrages.

3. **🩸 Les échecs de vérification frappent le SPRINT, pas le balayage.**
   0,97 % d'échecs en position 1-4 contre 0,52 % au-delà, et **9 des 13 étaient
   en zone payante** — ~0,9 % des créneaux mais ~3,3 % du REVENU.
   ✅ **MESURE FAITE LE 25/08 — LA LONGUEUR N'EST PAS LE FACTEUR.**
   `max_len` des groupes en échec (n=28) : p10 492 · **méd 780** · p90 1118.
   Groupes acceptés (n=3390) : p10 510 · **méd 745** · p90 1055. **Identique.**
   Taux d'échec plat par tranche de longueur : 0,80 % (<600) · 0,80 % (600-900)
   · 0,97 % (900-1500) · 0 % (>1500, n=71).
   🪤 « Les groupes en échec sont courts » est un ARTEFACT : toute notre
   production est courte. Toujours comparer à la population de référence.
   **Répartition des 28 échecs / 3883 verdicts (0,7 %)** : `logprob_mismatch`
   14 (rangs 6-16, tous en zone payante) · `reward_mismatch` 8 ·
   `bad_termination` 3 · **`seed_mismatch` 2 (rangs 1 et 3)** ·
   `token_tampered` 1. Le coût reste concentré sur les BONS rangs.
   ⚠️ n=28 : le résultat exclut un effet FORT de la longueur, pas un effet léger.
   🔭 Piste restante non explorée : le facteur commun n'est ni la longueur ni la
   position — chercher du côté du contenu du prompt ou du timing GPU.

4. **A/B ENTRELACÉ du sprint** (alterner tous les 3-4 fenêtres, ~4 h). Le seul
   design qui tue le confondant marché.

## ✅ L'ARRIVÉE RESTE LE LEVIER — vérifié par entrée (25/08, n=1 422)

| tokens | arrivée < 8 s | 8-12 s | 12-18 s | > 18 s |
|---|---|---|---|---|
| 0-3 000 | **65 %** (20) | 11 % (18) | 0 % (14) | 3 % (31) |
| 3 000-4 000 | **54 %** (35) | 12 % (33) | 7 % (54) | 0 % (68) |
| 4 000-5 000 | **64 %** (44) | 29 % (42) | 7 % (70) | 1 % (115) |
| 5 000-6 000 | 88 % (16) | 57 % (23) | 15 % (67) | 4 % (124) |
| 6 000-7 000 | 88 % (26) | 66 % (32) | 25 % (83) | 9 % (182) |
| 7 000-9 000 | **100 %** (9) | 89 % (27) | 40 % (94) | **12 %** (172) |

⚡ **À volume fixé, l'arrivée fait ×64 (64 % → 1 %). À arrivée fixée, le volume
fait ×2 (54 % → 100 %).** L'arrivée domine.
⛔ **« Un groupe court ne paie pas » est FAUX** : <3 000 tokens sous 8 s paie
**65 %**, rang médian 14,5 — mieux que 7 000-9 000 tokens après 18 s (12 %).

🪤 **J'ai publié la conclusion INVERSE une heure plus tôt** en croisant, au
niveau FENÊTRE, « première arrivée » et « tokens du plus gros groupe » — deux
agrégats portant sur des entrées DIFFÉRENTES, sur des cases à 4 et 9 obs.
**Ne jamais croiser deux agrégats de fenêtre pour une relation qui vit au
niveau de l'entrée. Refuser toute case sous ~20 observations.**

⛔ **NOS CHAMPS NE RECONSTRUISENT PAS SON CLASSEMENT.** Fen 32749 : l'entrée
rang 1 (7 502 tok, k=1, valeur locale 0,2269, bucket 75) bat celle du rang 4
(4 257 tok, k=6, valeur **0,3026**, bucket **85**) — inférieure sur les DEUX
critères. `opencodeinstruct` est `validator_authoritative_reward=True` : il
**re-note lui-même**, notre `k` et notre `score` ne sont pas les siens.
🪤 Calculer le round par `⌊flip_offset/3⌋+1` est FAUX (suppose l'ouverture à
notre offset 0) — **utiliser `arrival_drand_round`, jamais un round reconstruit.**

## ✅ DÉPLOYÉ 25/08 10:04 — PRIOR v5.8, coupure fenêtre 32439

`predictor_v58.json` + table `prompt_scores_v58.npz` régénérée (13,5 min,
0 erreur, empreinte vérifiée). **Non-régression** : top-8 et top-40 IDENTIQUES
à la notation en direct, écart max 1,3e-08, ×673.
**Repli** : `/workspace/predictor_v51_REPLI.json` (md5 `559f115d…` = ce qui
tournait) + `/workspace/prompt_scores.npz` **jamais écrasée** ; launcher
sauvegardé en `.bak-avant-v58`. Retour = 2 lignes + restart, sans régénération.

### Ce que v5.8 vaut RÉELLEMENT (à 64 fenêtres mûres)
| | v5.1 (nuit) | **v5.8** |
|---|---|---|
| payées/fenêtre mûre | 0,90-0,94 | **1,16** |
| fenêtres mûres payantes | 64 % | **68,8 %** |
| rang ≤20 | 34,5 % | 33,6 % — **identique** |
| rang médian | 30 | 28 |
| arrivée médiane | 20,1 s | 21,6 s |

⛔ **MON MÉCANISME ÉTAIT FAUX.** J'avais projeté +1169 tok/groupe sur le sprint
(t=+19,6 en test apparié intra-fenêtre) donnant −5,1 places et +39 % de payées.
**Mesuré en vol : +187 tok, IC95 [−238 ; +612]** — la borne haute exclut +1169.
Le rang médian et l'arrivée n'ont PAS bougé. Le seul effet qui tient est
l'**in_zone du sprint : 65,6 → 74,2 %** (j'avais prédit −2,3 points).
🪤 **Le gain s'est dégonflé en mûrissant** : +41 % à 21 fenêtres mûres, **+19 %
à 40**, ~+25 % à 64. Encore une fois — exiger 30 fenêtres mûres MINIMUM.

### Pourquoi promouvoir malgré tout (mesuré 25/08, duels propres)
| modèle | Spearman | lift top-10 % | valeur du top-4 |
|---|---|---|---|
| v5.1 EN PROD | +0,246 | 1,11× | 0,1323 |
| **v5.8** | **+0,525** | **1,95×** | **0,1643** |
| *hasard* | — | 1,00× | *0,1328* |

⚡ **v5.1 ne faisait PAS mieux que le hasard** sur les 4 slots du sprint
(0,1323 contre 0,1328 ; in_zone 75,4 % contre 77,1 %). Vérifié aussi sur un
vivier `explore` **que personne n'a filtré** : v5.8 +0,360 / 1,64× contre
v5.1 +0,259 / 1,30×.
⚠️ Réserve non levée : le vivier testé est celui que v5.1 a choisi de BAKER.
v5.8 réordonne mieux un sac donné ; rien ne prouve qu'il REMPLIT mieux ce sac.


## 🔬 A/B DU SPRINT — 24/08 : garder 4, mais **L'A/B NE CONCLUT PAS**

Vérifié par un agent indépendant qui m'a corrigé sur 4 points. Bras datés par
**détecteur de redémarrage** (1er groupe >25 s après l'ouverture, ou <20 groupes
dans la fenêtre) : restarts à **31812** et **31826** (ce dernier certifié par
l'heure du process, 15:42:25, et par le log « 4 prompts de sprint, 4 de
balayage »). ⚠️ Ma borne initiale de 31817 était FAUSSE.

| bras | bornes | config | n | **rang≤20** | IC95 Wilson | rang méd | arrivée |
|---|---|---|---|---|---|---|---|
| A | ≤31811 | sprint=4 | 139 | 17,3 % | [11,9 ; 24,4] | 36 | 18,9 s |
| **B** | **31812-31825** | **sprint OFF** | 41 | **0,0 %** | **[0 ; 8,6]** | 41 | **15,5 s** |
| C | 31826+ | sprint=4 | 107 | 15,0 % | [9,4 ; 22,9] | 38 | 21,2 s |

A et C (deux périodes sprint=4 séparées) sont **indiscernables** → la ligne de
base est 15-17 %, et B est le seul point hors plage.

### ⛔ POURQUOI ÇA NE CONCLUT PAS — mon p=0,07 % était FAUX
Je traitais les entrées comme indépendantes ; elles sont corrélées dans une
fenêtre et entre fenêtres voisines. **Test par blocs contigus** (le seul qui
gère l'autocorrélation) : p ≤ 0,017 au seuil rang≤20, mais **p ≤ 0,052 et
p ≤ 0,069** aux seuils 15 et 25. **Un seuil sur trois.**

Trois obstacles de fond :
1. **TROIS configurations, pas deux.** Le restart de 31826 a réécrit
   `launch_miner_v4.sh`. A et C, tous deux sprint=4, diffèrent ailleurs : le
   groupe le plus volumineux des 8 arrive à **16,9 s en A contre 10,5 s en C**.
   Rien ne garantit donc que 31812 n'a changé QUE `SPRINT_SIZE`.
2. **Effet sans mécanisme.** À bucket apparié le déficit persiste (résidu
   **+7,1 rangs**, t=+2,79 sur `rang ~ arrivée+volume+score`), notre valeur est
   plate (0,143 vs 0,145-0,153) et le marché n'est pas plus grand. Il ne reste
   que « le peloton était meilleur pendant ces 20 minutes » = un **confondant
   temporel**.
3. **9 fenêtres, ~20 min, UN SEUL BLOC.** Un design en blocs ne peut pas
   séparer le réglage de la dérive du marché.

**Pour trancher : A/B ENTRELACÉ** (alterner tous les 3-4 fenêtres pendant ~4 h,
~30 fenêtres propres par bras). C'est le seul design qui tue le confondant.

### ✅ La prémisse du CODE est morte — mais pas comme je le croyais
Cadence médiane (offsets depuis l'ouverture, fenêtres non froides) :

| | g1 | g4 | **g8** | groupes/fenêtre |
|---|---|---|---|---|
| sprint=4 | 6,8 s | 10,7 | **18,7 s** | **47** |
| sprint OFF | 7,1 s | 8,7 | **12,3 s** | **61** |

- **8/8 : le sprint COÛTE ~6 s.** Confirmé.
- ⛔ **1er groupe : ÉGALITÉ** (7,1 vs 6,8 s). Mon « 3,4 contre 4,1 s » n'existe
  pas — **le sprint ne fait PAS sortir la tête plus tôt.**
- ⛔ Le sprint coûte aussi **14 groupes/fenêtre** (47 contre 61).

Le commentaire de `vllm_backend.py:646` (« moins de séquences en vol = décodage
par séquence plus rapide ») est donc faux **dans les deux dimensions**.
⛔ `ops/AB_SPRINT.md` reste CADUC : il visait la baisse à 2, ce qui aggraverait
l'arrivée.

### 🔑 CE QUE FAIT RÉELLEMENT LE SPRINT (code lu, `vllm_backend.py:679-681`)
`n_sprint = max(0, min(sprint_size, n)); if n_sprint >= n: n_sprint = 0` — donc
`SPRINT_SIZE == BAKE_BATCH_SIZE` le désactive. Et `on_group(pos, …)` est appelé
dès que `remaining[pos]==0`, **sans aucune référence à la frontière du sprint** :
un groupe prêt part IMMÉDIATEMENT. Le sprint n'agit donc **ni sur la sélection
ni sur l'envoi**, seulement sur l'**ordre d'admission au moteur** (les 4 têtes
seules, le balayage enfilé à la livraison du dernier ou à `sprint_max_wait_s=20`).

### 🪤 QUATRE ERREURS COMMISES SUR CE SEUL A/B EN UNE JOURNÉE
1. Moyenne sur des populations de tailles différentes → faux « −11 % de valeur ».
2. Borne de bras prise de mémoire (31817 au lieu de 31812).
3. Payées comme métrique (trop bruitées : 0,64/fenêtre).
4. **Entrées traitées comme indépendantes** → p=0,07 % au lieu de p≤0,017.
**Règles** : dater par le détecteur de redémarrage ; juger sur rang≤20 ; test
par blocs contigus ; et vérifier qu'UN SEUL réglage a changé entre les bras.

## 🔁 LE WATCHDOG REDÉMARRE LE MINEUR ~1×/HEURE — ET ÇA NE COÛTE RIEN

**11 redémarrages `FUITE_VRAM` le 25/08** (01:18, 03:05, 04:27, 05:33, 06:29,
07:26, 08:23, 09:23, 11:17, 12:36, 13:43) + 2 `PROCESS ABSENT` (10:04, 12:47).
Seuil `WATCHDOG_VRAM_MAX=135000` MiB (`watchdog.sh:81`) ; les déclenchements
tombent entre **135019 et 135503 MiB** — marge de **0,01 % à 0,4 %**. VRAM au
repos mesurée : **128,5 Go / 143,7 Go**. Ça ressemble à une garde qui mord sur
le PIC NORMAL (vLLM ~109 Go à 0,76 + preuves GRAIL ~27 Go), pas sur une fuite.

⛔ **NE PAS en faire un chantier : le coût est INDÉTECTABLE.** Sur les 12
redémarrages du jour, **8,2 %** des fenêtres qui suivent ont ≤1 acceptée,
contre **12,7 %** de référence toutes fenêtres confondues — c'est-à-dire
*mieux* que la moyenne. Le redémarrage ne coûte rien de mesurable ; c'est le
**rechargement de checkpoint** qui coûte (cf. TODO n°1).
🪤 **Piège vécu le 25/08** : j'ai d'abord attribué une perte de fenêtres au
restart de 13:43:51 parce qu'il tombait au bon moment. **Faux** — la vraie
cause était un rechargement de checkpoint 30 min plus tard. Toujours dater
l'incident AVANT de désigner un coupable qui traîne dans le voisinage.

⚠️ Bug cosmétique : `watchdog.sh:98` crache en boucle `[: 0\n0: integer
expression expected` (`pgrep -fc` renvoie deux valeurs). La garde fonctionne,
mais ça pollue `watchdog.log`.

### 🕳️ 5,7 % DES FENÊTRES SONT INVISIBLES DANS NOS DONNÉES
**47 numéros sur 830** (31755→32584) sont absents **des DEUX** fichiers —
`submits_v4.jsonl` ET `windows_v4.jsonl` : le rechargement avale la fenêtre
avant qu'une seule ligne soit écrite. **Tout compteur fondé sur ces fichiers
les rate silencieusement.** On ne les retrouve qu'en cherchant les **trous dans
la numérotation**, qui est contiguë (vérifié : la fenêtre perdue 32577 apparaît
comme trou, et l'avancée de `checkpoint_n` est enregistrée sur la fenêtre
**SUIVANTE**, 32578 — pas sur la fenêtre perdue).
⛔ C'est cet angle mort qui m'a fait annoncer « 1,2 % de fenêtres perdues au
checkpoint » le 25/08 : chiffre **FAUX par construction**. Le vrai coût est
**10,6 %** (décompte complet en TODO n°1) — la note d'origine avait raison,
c'est ma vérification qui était aveugle.

### 📡 `ReadTimeout` VALIDATEUR : ~5/h, STABLE
`fire task exception … ReadTimeout('')` = notre POST expire, l'entrée est
perdue. Par tranche horaire le 25/08 : 2 · 6 · 5 · 8 (10h→13h). ~5 % des
entrées envoyées. **Pas une dégradation** — ne pas s'en alarmer sans voir la
cadence horaire monter.

## ⛔ NE PAS ATTAQUER LE `stale_round` POUR LUI-MÊME (mesuré 24/08 soir)

Le taux global (21,7 % après les correctifs, 27,3 % avant) est une **moyenne
trompeuse** : il se concentre presque entièrement sur les entrées tardives,
qui ne rapportent rien (1,7 % de payées au-delà de la 4e position).

| arrivée | n | acceptées | **`stale_round`** | `batch_filled` |
|---|---|---|---|---|
| **0-10 s** | 34 | **94 %** | **3 %** | 3 % |
| 10-15 s | 19 | 68 % | 32 % | 0 % |
| 15-20 s | 33 | 64 % | 21 % | 15 % |
| 20-25 s | 23 | 48 % | 35 % | 17 % |
| 25-35 s | 29 | 14 % | 31 % | **55 %** |

Sur les 47 tentatives arrivées **sous 12 s** : **87 % acceptées**, 10,6 % de
`stale_round`, 2,1 % de `batch_filled`.

⚡ **`stale_round` et `batch_filled` sont deux symptômes de la MÊME cause** :
arriver trop tard. À partir de 25 s le `batch_filled` domine (55 %) — le
guichet est fermé.

⛔ **L'alignement sur la grille drand est donc à ÉCARTER** : il coûterait un
round d'arrivée à TOUTES les entrées pour sauver celles qui valent 1,7 %.
Le seul levier reste de faire arriver plus d'entrées **sous 10 s**, où 94 %
passent.

## 🩸 CONTEXTE HISTORIQUE — le `stale_round` avant les correctifs (24/08)

290 sur 1 126. **Purement de la latence d'envoi**, et le mécanisme est net :

| dépassement de round drand | n | taux `stale_round` |
|---|---|---|
| **0** | 481 | **0,0 %** |
| 1 | 400 | 35,2 % |
| 2 | 139 | **66,9 %** |

Le round est calculé au DÉBUT de `_build_signed_request_sync` (`engine.py:3301`),
puis viennent signature, build pydantic, `_build_precommit`, POST precommit,
POST submit. **Si l'ensemble déborde des 3 s du round → entrée perdue.**
Taux d'acceptation actuel : **52,8 %**.

⛔ **Ce n'est PAS le CPU** (banc : 16 `RolloutSubmission` 16 ms, `model_dump_json`
de 877 ko 4 ms, sha256 0,6 ms, sr25519 0,02 ms = **~25 ms**) ni le réseau (RTT
0,19 s). Il reste **1,0-1,5 s non instrumentées** = `_finalize_pool_entry` (GPU)
+ traitement validateur.

## ⚡ LA PREUVE GRAIL EST UNE FILE SUR LE GPU (pas un verrou)

`_proof_rollouts` fait 16 forwards HF sur **la même H200 que vLLM**.
`fork_gpu_guard` est OFF (pas dans le launcher) — ce n'est pas lui.

| preuves qui se chevauchent | 0 | 1 | 2 | 3 | **4** |
|---|---|---|---|---|---|
| durée de la preuve p50 | 0,66 | 1,09 | 1,55 | 3,08 | **5,76 s** |
| durée de l'étage POST p50 | 0,83 | 1,59 | 2,93 | 3,48 | **7,12 s** |

**Aucune dépendance à la longueur** (max_len 500-999 → 0,79 s ; 4000+ → 0,56 s)
⇒ c'est une FILE, pas du travail. `_finalize_pool_entry` (`engine.py:6036`)
remonte 16 tenseurs de hidden states sur ce même GPU et subit le même effet.

⚡ **Correctif à coût nul** : `RELIQUARY_GRADE_CONCURRENCY` **8 → 2-3**. Les 8
groupes d'un bake entrent en preuve ensemble et se sérialisent ; en bornant, les
**PREMIERS** sortent 3-5× plus vite — or seuls les 3 premiers comptent (seal à
+25 s). Variable d'env, réversible, 0 octet changé.

## 🔬 SAVOIR SI C'EST NOUS OU EUX — les verdicts portent LEUR horloge

`verdicts_v4.jsonl` contient **`precommit_arrival_ts`** et **`arrival_ts`** :
les horodatages du VALIDATEUR à notre arrivée. Joint sur `merkle_root`, ça
découpe le délai d'envoi sans aucun biais d'horloge relatif :

| grandeur | ce qu'elle mesure |
|---|---|
| `precommit_arrival_ts − t_proof_end` | **NOUS** : finalize + signature + montée réseau |
| `t_post − precommit_arrival_ts` | **EUX** : traitement validateur + réponse |
| `arrival_ts − precommit_arrival_ts` | aller-retour du petit POST (~0,22 s) |

**Mesuré le 24/08 sur une vague de lenteur (n=98)** :

| période | NOUS | EUX |
|---|---|---|
| calme (31998-32028, n=84) | **0,44 s** | 1,34 s |
| lente (32029-32034, n=14) | 0,66 s | **5,70 s** |

Sur 5 s de dégradation, **4 viennent d'EUX et 0,25 de nous**. Notre part est
remarquablement stable autour de **0,44 s** — c'est le seul chiffre sur lequel
on puisse agir, et il est déjà petit.

⚠️ **Mécanisme de la cascade** : notre client attend la réponse HTTP avant de
libérer le créneau d'envoi. Un serveur lent à 5 s retarde donc AUSSI les
entrées suivantes.

⛔ **Ne jamais affirmer « c'est le validateur » sans ce découpage** —
`t_post − t_proof_end` contient notre `_finalize_pool_entry`, qui utilise le
GPU partagé avec vLLM et peut donc être ralenti par NOUS.

## 🪤 PIÈGES DE MESURE DU CHEMIN D'ENVOI (corrigent mes chiffres du 24/08)

1. ⛔ **Une entrée RE-TIRÉE réécrit une ligne avec le MÊME `_timeline`.** Étage
   POST apparent : **7,57 s p50 sur 281 re-tirs** contre **1,71 s** en 1re
   tentative. **DÉDUPLIQUER par `(prompt_idx, t_pick)`** — sans ça tous les
   chiffres d'envoi sont gonflés.
2. ⛔ **Les médianes ne s'additionnent pas.** J'ai cru à un « trou de 3,7 s » en
   soustrayant des médianes d'étages d'une médiane totale. Les MOYENNES, elles,
   sont additives à 0,01 s près.
3. ⛔ **`groupe N prêt à X s` est relatif au DÉBUT DU BAKE**, pas au flip. Le
   bake démarre à flip+3,7 s.
4. **Sémantique réelle des horodatages** : `t_pick` = début de
   `_pre_bake_entry` (le groupe est DÉJÀ généré) ; `t_gen_end` lit un cache et
   vaut **0,00 s** (ne mesure PAS la génération) ; `t_post` est posé APRÈS le
   retour de la réponse validateur.
5. **Les compteurs `fire_diag` ne s'incrémentent QUE sur un refus.**
   « `sealed` 17,8 et zéro ailleurs » ne dit PAS « la file n'est jamais
   saturée » — ça dit qu'après le 1er `batch_filled` tout est refusé en bloc.

## 💰 CE QUE VALENT LES SECONDES — chiffré, effets fixes fenêtre (24/08)

⛔ **MON SPEARMAN +0,30 ÉTAIT UN ARTEFACT D'AGRÉGATION** (il mélangeait des
fenêtres de difficulté différente). Avec **effets fixes par fenêtre** (on ne
compare que NOS entrées entre elles), n=665, R²=0,74 :

| levier | effet | t |
|---|---|---|
| **arrivée** | **+1,46 place/seconde** (+1,83 hors `explore`) | 27,9 |
| volume | −4,69 places / 1000 tok | −28,6 |

⚡ **1 seconde = 390 tokens.** Je sous-estimais l'arrivée d'un facteur ~3, et
c'est sur cette base que j'ai failli abandonner le chantier.

Mécanisme vérifié : `bucket = volume // (round × 50)`, `round = ⌊arrivée/3⌋+1`,
et **rang ≈ −1,004 × bucket** (R²=0,504). Les deux leviers entrent dans la MÊME
formule, en produit. Ni σ ni k n'ajoutent rien (t = −0,8 et +0,6).

### La falaise est au rang 20 PILE
| rang | n | payées |
|---|---|---|
| 10-14 | 21 | **90,5 %** |
| 15-19 | 63 | **82,5 %** |
| **20-24** | 54 | **11,1 %** |
| ≥30 | 463 | **0,0 %** |

Rang ≤19 → **84,6 %** ; ≥20 → **1,2 %**. Il faut **bucket ≈ 33**.
Notre médiane : volume 6008, arrivée 16,1 s, round 6 → **bucket 20, rang 37**.

### Contrefactuel (532 entrées / 174 fenêtres mûres, calibré à −4 %)
| scénario | payées/fenêtre | gain |
|---|---|---|
| base | 0,44 | — |
| **arrivée −1,5 s** | 0,62 | **+39 %** |
| **arrivée −3 s** | **0,85** | **+91 %** |
| arrivée −5 s | 1,03 | +134 % |
| volume +2000 tok | 0,90 | +105 % |

⚡ **−3 s ≈ +2000 tokens**, un gain qu'aucun levier de volume n'a approché
(μ=0,05 donnait +14,6 %).
⚠️ **NON LINÉAIRE** : depuis 16,1 s, **−1,2 s suffit** à gagner le round 5 et
capture **39 %** du bénéfice ; le palier suivant exige −4,2 s. Viser « 2-3 s »
sans savoir où tombent les frontières de round gaspille la moitié de l'effort.

### Chronologie réelle de notre 1re entrée (180 fenêtres, médianes)
| jalon | offset | delta |
|---|---|---|
| **fin génération / `t_pick`** | **8,47 s** | +8,47 |
| fin grading | 8,60 s | +0,12 |
| fin preuve GRAIL | 9,67 s | +1,07 |
| **precommit reçu par le validateur** | **10,01 s** | +0,34 |
| submit reçu | 10,29 s | +0,28 |
| `t_post` (notre horodatage) | 12,99 s | +2,70 |

**97 % du budget est EN AMONT de la preuve.** ⛔ La note « flip → premier groupe
utilisable = 4,25 s » du CLAUDE.md est **FAUSSE** : c'est **8,47 s**.
Notre biais de mesure (`t_post` vs precommit réel) est de **2,0-2,6 s**, pas 6.

### On est bien en retard (marché, 202 fenêtres de dashboard)
`upload_lag` marché p50 **9,4 s**, min **4,9 s**, ~29 acceptées/fenêtre pour
~72 candidats. Notre MEILLEURE entrée à 12,5 s = **61ᵉ centile** (17/29 mineurs
déjà arrivés) ; notre entrée médiane à 15,8 s = **89ᵉ centile**.

### 🪨 LES SLOTS `explore` SONT DU LEST
133/665 des admises (**20 % du quota**), rang médian **48**, volume 4 887,
**3,0 % payées** contre 15-17 % pour `memo`/`ranked`.

## ✅ DÉPLOYÉ 24/08 20:45 — LATENCE D'AMORÇAGE, coupure fenêtre 31995

**Quatre correctifs**, commits `3bdbd06` (instrumentation) et `0001e97` :

| réglage | avant | après | nature |
|---|---|---|---|
| `RELIQUARY_PROMPT_SCORES` | — | `/workspace/prompt_scores.npz` | **code** |
| `RELIQUARY_PARQUET_LOCAL_ROOT` | — | `/workspace/parquet_mirror` | **code** |
| `RELIQUARY_PARQUET_EXPECTED_LEN` | — | **2481806** (garde) | code |
| `RELIQUARY_GRADE_CONCURRENCY` | 8 | **3** (défaut du code) | env |
| `RELIQUARY_EXPLORE_SLOTS` | 2 | **0** (défaut du code) | env |

### ✅ ACQUIS — chronomètres, indépendants de la maturité
| | avant (91 fen) | après (7 fen) |
|---|---|---|
| **classement de tranche** | **2,80 s** | **0,0 s** (journal du mineur) |
| 1er groupe généré | 6,7 s | **4,7 s** |
| **1re entrée arrivée** | **14,8 s** | **8,3 s** (−6,5 s) |
| preuve → POST | 1,78 s | 1,73 s (inchangé, non touché) |
| entrées/fenêtre | 2,8 | 3,4 |

**Vérification de non-régression du classement** (tranche de 5 000 tirée au
hasard) : notation en direct 2,07 s contre **0,0021 s** par la table (×985),
**top-8 IDENTIQUE, top-40 IDENTIQUE**, écart max de score 1,27e-08.
**Miroir parquet** : `len()` = **2 481 806** des deux côtés, 16 lignes
comparées sans différence, manifest 2,55 s → **0,02 s**.

### ✅ RÉSULTAT CONSOLIDÉ — 27 fenêtres mûres, le chiffre s'est STABILISÉ
| échantillon | rang ≤20 | IC95 |
|---|---|---|
| 7 fenêtres | 45,5 % | [27-65] |
| 15 fenêtres | 40,5 % | [27-56] |
| 23 fenêtres | 39,1 % | [28-51] |
| **47 fenêtres / 136 rangs** | **39,7 %** | **[32-48]** |

Stabilisé à 39-40 %, IC resserré à [32-48], **toujours entièrement au-dessus
de la borne haute d'avant (21 %)**. Contrairement aux trois autres lectures du
jour, celle-ci n'a pas fondu en maturant.

| | avant (84 fen, 55 mûres) | après (47 fen, 27 mûres) |
|---|---|---|
| **rang ≤20** | 16,2 % [12-21] | **39,7 %** [32-48] — **×2,45** |
| rang médian | 34 | **25** |
| **payées/fenêtre mûre** | **0,40** | **1,00** — **×2,5** |

### 💰 PAYÉES PAR HEURE — l'objectif de croisière est DÉPASSÉ
| | avant (2,72 h) | après (0,73 h) |
|---|---|---|
| acceptées | 297 | 85 |
| payées | 33 | 23 |
| **payées/heure** | **12,1** | **31,5** |
| arrivée médiane | 18,0 s | 15,8 s |
| rang médian | 34 | 24 |

Par tranche horaire : 0,35-0,58 payée/fen de 13h à 19h, puis **0,88 à 21h**
(plein effet). La référence « vitesse de croisière » du 21/08 était de
**13,1 payées/h** — dépassée d'un facteur 2,4.

⚠️ **Réserves** : 44 min d'échantillon après, 23 payées, verdicts jeunes ; et
le rang est **positionnel** — si le peloton s'adapte, l'avantage se réduit.
On a gagné une place dans une course, pas une propriété absolue.
⚠️ Trois lectures précoces se sont dégonflées aujourd'hui — **exiger 30
fenêtres mûres** avant d'annoncer un chiffre (`scripts/ab_prior.py
--avant 31900-31994 --apres 31998-`).

**Repli** : vider les 3 variables du launcher + restart. Sauvegardes sur la
box : `backup_avant_latence_2027/`, `launch_avant_latence.sh`.

**Régénérer la table** après un ré-entraînement du prior (sinon elle est
détectée périmée et le mineur retombe seul sur la notation en direct) :
`python3 scripts/precompute_prompt_scores.py --out /workspace/prompt_scores.npz`
(~15 min, 29,8 Mo, n'interrompt pas le minage).

## 🔴🔴🔴 LE GUICHET = UNE COURSE À 64 PLACES (source validateur, 24/08)

⛔ **NOTRE NOTE ÉTAIT FAUSSE** : `batch_filled` ne vient PAS de
`MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW`. Sur le chemin precommit
(`server.py:3143-3150`), tout refus de `try_register_upload_precommit` devient
`BATCH_FILLED`. Le compteur qui mord (`batcher.py:1204-1208`) :

```
if self._upload_precommit_accepted >= MAX_PENDING_UPLOAD_PRECOMMITS_PER_ENV:  # = 64
```

**`_upload_precommit_accepted` est CUMULATIF et JAMAIS décrémenté** — ni à la
révélation, ni à l'expiration ; remis à zéro seulement au batcher suivant.

⚡ **64 precommits par ENV et par FENÊTRE, pour TOUT le marché, premier arrivé
premier servi.** Vérifié en vol (`/health` toutes les 2 s) :

| fenêtre | +5 s | +15 s | +22 s | +33 s |
|---|---|---|---|---|
| 31965 code | 4 | 30 | **64** | |
| 31966 code | 18 | 22 | — | **64** |

**La porte se ferme entre +15 s et +33 s**, alors que `collection_seconds=100`.
Nos `batch_filled` (247 cas) ont un offset médian de 24,8 s contre 18,7 s pour
les `submitted` — concordance exacte.

⚠️ **Question ouverte la plus rentable** : les 64 places sont-elles prises par
~49 mineurs, ou par 2 hotkeys à 32 chacune ? (`MAX_SUBMISSIONS_PER_HOTKEY = 32`
autorise 2 mineurs à tout rafler.) Réponse via `/verdicts` + dashboard marché.

## ⏱️ LE PRECOMMIT NE COÛTE PAS 2,1 s — IL COÛTE 0,22 s

⛔ **Mon erreur** : `t_post − t_proof_end` (2,17 s p50) couvre TOUT le pipeline,
y compris le téléversement de **431 ko** de la phase 2 — qui arrive APRÈS
l'estampille de rang et ne coûte AUCUN rang.

🔑 **Le verdict expose `precommit_arrival_ts`** (horloge du validateur). Joint
sur `merkle_root` (802 entrées) :

| grandeur | p10 | **p50** | p90 |
|---|---|---|---|
| `t_post − t_proof_end` (ce que je mesurais) | 1,17 | **2,17 s** | 12,02 |
| **`precommit_arrival − t_proof_end`** (avant l'estampille) | 0,23 | **0,44 s** | **6,19** |
| `t_post − precommit_arrival` (après, gratuit en rang) | 0,81 | 1,36 s | 6,97 |
| aller-retour du petit POST | 0,20 | **0,22 s** | 0,54 |

Télémétrie du validateur (`/health`) : `/submit/precommit` p50 **19,3 ms**,
`commit_lock_wait_ms` p50 **0,002 ms** → **aucun verrou global**, aucune E/S,
aucun accès chaîne dans le handler.

✅ **Le rang EST estampillé au precommit** — confirmé en source :
`server.py:3856-3858` (`drand_timing_source="precommit_arrival"`) →
`batcher.py:3988` → clé de tri `batcher.py:4333-4345`
`(-value, throughput_rank, arrival_round, …)`. **`arrival_round` est un round
drand quantifié à 3 s** : c'est le round qui compte, pas la milliseconde.

## 🩸 22,5 % DE NOS ENVOIS MEURENT EN `stale_round`
314 sur 1 395, offset médian **19,9 s**. La tolérance arrière du round drand est
**ZÉRO** (confirmé `/health`), et notre délai finalize→precommit atteint
**6,2 s au p90**. Toute traversée d'une frontière de 3 s entre le calcul du round
et l'arrivée du precommit **tue l'entrée**.

⚡ **Réduire les 0,44 s (p90 6,2 s) entre fin de preuve et départ du precommit
est LE levier** : il touche à la fois le `stale_round`, le round d'arrivée
(donc le bucket) et la place dans la course aux 64 slots.

## 🔴🔴 LE GUICHET SE FERME À 25 s — 78 % DE NOTRE PRODUCTION EST PERDUE

**Trouvé le 24/08 par instrumentation** (`_fire_diag`, déployé 18:04). C'est LA
contrainte, et elle rend caduques la plupart des raisonnements de la journée.

| mesure | valeur |
|---|---|
| fermeture du guichet (1er `batch_filled`) | **médiane 24,8 s** (p10 16,5 · p90 36,8) |
| fenêtres recevant un `batch_filled` | **143/145 — 99 %** |
| groupes générés **avant** la fermeture | **9,8**/fenêtre |
| groupes générés **après** | **35,6**/fenêtre |
| **part de production PERDUE** | **78 %** |

**Mécanisme** (`engine.py:3184`) : un rejet `batch_filled` pose
`self._sealed_window = state.window_n`. Ensuite `_maybe_fire_on_append` refuse
tout tir sur cette fenêtre.

### 📊 COMPTEURS CONFIRMÉS — 6 fenêtres consécutives, UN SEUL motif
| motif de blocage | /fenêtre |
|---|---|
| **`sealed`** (guichet déjà fermé) | **17,8** |
| `inflight_saturated` (file pleine) | **0** |
| `budget_exhausted` (quota 32) | 0 |
| `dropped_cooldown` | 0 |
| `dropped_out_of_slice` | 0 |
| `left_no_budget` | 0 |
| `not_open` / `stale_state` / `not_fire_as_ready` | 0 |

⛔ **TROIS hypothèses tombent d'un coup** :
1. « La file d'envoi est saturée » — **FAUX**, `inflight_saturated` = 0.
   `MAX_INFLIGHT_FIRES=6` ne mord JAMAIS. Ne plus raisonner dessus.
2. « Le cooldown/la tranche jettent des groupes » — **FAUX**, 0 et 0.
3. « On bute sur le quota de 32 soumissions » — **FAUX**, 0.

Il ne reste **qu'une** contrainte, et elle est EXTÉRIEURE : le batch du marché
se remplit vers 25 s (16,5 s dans les fenêtres disputées).

### 📉 LA BAISSE DES ACCEPTÉES EST DE NOTRE FAIT, PAS DU MARCHÉ

**Le guichet ne se ferme PAS plus tôt** — fermeture médiane stable toute la
journée : 21,4 s (13h) · 22,6 (15h) · 26,0 (16h) · 24,1 (18h), p10 stable à
16-19 s. ⛔ Ne pas accuser le marché.

**La cascade, mesurée de bout en bout** (l'activation du bonus de volume est à
16h56) :

| heure | générés | avant fermeture | in_zone & avant | tentés | **acceptés** |
|---|---|---|---|---|---|
| 14h | 46,9 | 13,8 | 11,7 | 7,1 | 4,1 |
| **16h** (avant volume) | 46,1 | 11,4 | 10,2 | **9,1** | **5,2** |
| **17h** (après volume) | 41,9 | 9,1 | 7,5 | 6,4 | **3,5** |
| 18h ⚠️ | 36,0 | 8,5 | 6,6 | 5,6 | 2,9 |

⚠️ La tranche 18h contient le redémarrage de l'instrumentation (rodage) — la
comparaison propre est **16h vs 17h**.

Tout part de la PREMIÈRE colonne : des rollouts plus longs = moins de groupes
pour le même temps GPU → moins avant la fermeture → moins d'in_zone utilisables
→ moins de tentatives → moins d'acceptées. Chaque maillon est mesuré.

### ⚖️ VERDICT DU BONUS DE VOLUME : **NEUTRE** (57 fenêtres, 32 mûres)

| | sprint=4 seul | + bonus de volume |
|---|---|---|
| fenêtres / entrées | 34 / 182 | **57 / 219** |
| rang ≤20 | 13,7 % [9-20] | **16,9 %** [12-23] |
| rang médian | 38 | 37 |
| **payées/fenêtre mûre** | **0,50** (14 mûres) | **0,47** (32 mûres) |

**Ni le +44 % annoncé par les agents, ni la perte que je craignais.** La qualité
par entrée monte un peu, le nombre d'entrées baisse (3,5 contre 5,2), le produit
ne bouge pas. IC entièrement recouvrants.

🪤 **LEÇON DE MÉTHODE — trois lectures successives du MÊME réglage** :
| échantillon | rang ≤20 annoncé |
|---|---|
| 50 entrées | 14,0 % (« défavorable ») |
| 83 entrées | 19,3 % (« favorable ») |
| **178 entrées, 32 fenêtres mûres** | **16,9 % (neutre)** |
**Il faut ≥30 fenêtres MÛRES avant qu'un chiffre tienne sur cette métrique.**
Chaque lecture précoce s'est dégonflée — dans les deux directions.

**Recommandation : ne rien changer.** Le réglage est neutre ; un redémarrage
coûte une fenêtre certaine pour un gain indémontrable.

### ⏱️ 6 À 8 SECONDES PERDUES ENTRE « PRÊT » ET « ENVOYÉ » (24/08)

| slot | généré à | **envoyé à** | **délai perdu** |
|---|---|---|---|
| n°1 | 7,6 s | **14,4 s** | **6,8 s** |
| n°2 | 8,6 s | 16,4 s | **7,8 s** |
| n°3 | 11,4 s | 18,7 s | **7,3 s** |
| n°4 | 14,3 s | 20,6 s | **6,3 s** |
| n°5 | 16,0 s | 22,2 s | 6,2 s |
| n°6 | 22,8 s | 25,8 s | 3,0 s |

Cadence de départ : 4 premiers slots partis en **20,6 s**, le 5ᵉ à 22,2, le 6ᵉ à
**25,8 s** — juste APRÈS le guichet (24,8 s), d'où son taux de paiement nul.

⚡ **Le premier groupe EXISTE à 7,6 s et n'arrive qu'à 14,4 s.**

### ⛔ MAIS CE DÉLAI N'EST PAS LE NÔTRE — piste FERMÉE (24/08)
| étape | médiane | marge réelle |
|---|---|---|
| grading | 0,07 s | négligeable |
| preuve GRAIL | **0,91 s** | déjà optimisée (chemin fusé actif par défaut) |
| **precommit** | **2,10 s** | dont **~1,5 s CHEZ EUX** |
| second POST (`/submit`) | ~0 s | rien |

**Pourquoi le precommit n'est pas optimisable de notre côté** :
- Le validateur est en **HTTP, pas HTTPS** → aucune poignée TLS à amortir.
- Mesure réseau depuis la box : `connect` **0,19 s**, aller-retour `/health`
  complet **0,57 s**.
- La charge du precommit est **minuscule** (`_build_precommit` : hotkey,
  window, prompt_idx, merkle_root, checkpoint_hash, payload_bytes/sha256,
  nonce, signature — quelques centaines d'octets). **Les tokens partent dans le
  SECOND POST**, pas celui-ci.
→ 2,10 s pour ~500 octets sur une connexion réutilisée vers un serveur à
0,19 s : **c'est leur traitement**. Gisement réel de notre côté : **~1 s au
mieux**, pas 6.

✅ **Les 4 optimisations de latence v4 sont TOUJOURS ACTIVES** (vérifiées en
source) : `_build_precommit` via `to_thread`, sleep 0,1 s en zone rouge,
`submit_diag` désactivée par défaut (`RELIQUARY_SUBMIT_DIAG`), chemin de preuve
fusé (`RELIQUARY_PROOF_FUSED` défaut 1). **Le passage v4→v5 n'a RIEN changé au
chemin d'envoi** — seul le prompt diffère.

⚠️ **Piège que j'ai refait** : comparer le délai des entrées REJETÉES au
precommit (1 aller-retour) à celui des ACCEPTÉES (2 aller-retours) pour en
déduire la répartition. Ce sont **deux populations différentes** — les
`batch_filled` surviennent plus tard dans la fenêtre. La répartition
precommit/submit reste donc non mesurée.

### 🎯 LE VRAI CHANTIER EST EN AMONT
Les **7,6 s** entre l'ouverture de la fenêtre et le premier groupe généré. Deux
sous-questions jamais séparées : le délai avant le démarrage du bake, et la
durée de génération elle-même. (Les 1,15 s + 3,1 s du CLAUDE.md datent de la v4,
où les complétions étaient 45 % plus courtes.)

**C'est LE levier** : la n°1 paie 21,6 % en arrivant à 14,4 s ; le rang médian
passe de 31 à 15 sur quelques secondes. Récupérer 5-6 s ferait gagner une
position équivalente aux QUATRE premières, qui font 98 % du revenu.

### 🎯 SEULES LES 4 PREMIÈRES ENTRÉES PAIENT (mesuré 24/08, n=604)

Position = ordre de fin de génération dans la fenêtre.

| position | n | arrivée | rang méd | **payées** |
|---|---|---|---|---|
| n°1 | 135 | +16,2 s | 31 | **21,6 %** [15-30] |
| n°2 | 129 | +18,8 s | 33 | 17,8 % [12-26] |
| n°3 | 114 | +20,4 s | 39 | 12,0 % [7-20] |
| n°4 | 89 | +23,3 s | 43 | 7,6 % [4-16] |
| **n°5** | 61 | +24,0 s | 39 | **0,0 %** [0-7] |
| n°6-7 | 56 | +28 s | 45 | 4-6 % |
| 8e+ | 20 | +40,1 s | 50 | **0,0 %** |

| | n | payées | rang méd |
|---|---|---|---|
| **1-4 (sprint)** | 467 | **15,4 %** [12-19] | 35 |
| **5+ (balayage)** | 137 | **1,7 %** [0-6] | 44 |

**Neuf fois moins, et les IC ne se recouvrent PAS** — l'un des rares résultats
franchement tranchés de la journée.

⚡ **Ce n'est PAS une question de qualité** : le balayage produit AUTANT de
tokens que le sprint (6 348 contre 6 211). C'est purement l'ARRIVÉE — la 5ᵉ
entrée arrive à +24,0 s, pile sur le guichet qui ferme à 24,8 s.

**Gradation même dans le sprint** : 21,6 % → 7,6 % de la 1re à la 4e. Chaque
position perdue coûte ~1/3 des chances.

**Conséquence stratégique** : produire PLUS d'entrées ne sert à rien (les
suivantes valent 1,7 %) ; le seul levier est de faire **arriver plus tôt les
toutes premières**. Le balayage ne coûte rien non plus (GPU déjà consommé,
quota de 32 jamais atteint) — inutile de le supprimer.

### ⚡ CONSÉQUENCE : LA BONNE MÉTRIQUE EST « GROUPES AVANT 25 s »
Ni « groupes/fenêtre », ni « tokens/seconde ». Générer plus ou plus longtemps
APRÈS la fermeture ne vaut RIEN.

| configuration | groupes avant 25 s |
|---|---|
| sprint=4 | 9,4 et 10,2 |
| **sprint désactivé** | **13,5** |
| sprint=4 + bonus de volume | **6,7** ← le pire |

⛔ **Ne plus juger un réglage sur les groupes/fenêtre.** C'est ce qui m'a fait
alerter à tort sur la « chute de débit » du bonus de volume, et sous-estimer le
sprint désactivé.

### 🔭 CE QUE ÇA OUVRE
Les 75 s de GPU après la fermeture ne produisent rien d'utilisable (la
génération est liée à la randomness de la fenêtre courante). Le levier n'est
donc pas de générer plus, mais de **faire entrer davantage dans les 25
premières secondes**.

## ✅ DÉPLOYÉ 24/08 16:56 — `volume_v1.json` ACTIVÉ, A/B EN COURS

**Coupure à la fenêtre 31868.** Journal du mineur, 16:57:02 :
`bonus de volume ACTIF: /workspace/volume_v1.json (15232 poids, mu=0.05)`
(avant : « modèle de volume illisible — bonus désactivé »). md5
`afea881f3184365dbd8f57f59fb2ca72`, parité de contrat OK, 0 Traceback.

**Commande du verdict** (écarter 31868-31870, moteur froid) :
`python3 scripts/ab_prior.py --avant 31829-31867 --apres 31871-`

| repère | valeur |
|---|---|
| ligne de base avant | **14,7 %** [9-22] d'entrées à rang ≤20 |
| attendu si le chiffrage tient | **21-26 %** |
| seuil pour conclure | dépasser **~25 %** (sortir de l'IC) |
| volume par groupe | 6 674 → attendu **~7 460** |

⚡ **Le signal RAPIDE est le VOLUME, pas le rang** : il se voit dès les
premières fenêtres et ne dépend pas du peloton. S'il ne monte pas vers 7 400,
le bonus ne mord pas — inutile d'attendre les verdicts.
⚠️ Il faut ~100 entrées par bras (≈2 h) pour que le rang ait un sens, et le
validateur est INSTABLE (nos arrivées 17 → 29 s avant le restart, sans que
notre génération bouge) : si les deux bras ne subissent pas la même
instabilité, la mesure est biaisée.
**Repli** : supprimer `/workspace/volume_v1.json` + restart.

Ce qui contient ce fichier : régression linéaire sac-de-mots sur
`log(tokens du groupe)`, 15 232 poids, 16 071 prompts, validée par **coupe par
prompt** sur 4 018 énoncés jamais vus (Spearman +0,63). Traits = fréquences de
termes L2-normalisées sur les 400 premiers mots + longueur du prompt (poids
+0,33 seulement, donc c'est le CONTENU qui décide). Poids les plus positifs :
`competition`, `challenge`, `quickselect`, `conquer`, `inversion`,
`count_subsets` ; les plus négatifs : `linear_search`, `insertion_sort`,
`compute_average`, `find_max`. Il a appris la **difficulté algorithmique**, pas
un artefact de surface — vérifiable à l'œil dans les poids.

## 🔴 POURQUOI C'ÉTAIT LE LEVIER N°1 (mesuré avant déploiement, 24/08)

**Deux agents indépendants, méthodes différentes, même conclusion.**

Journal du mineur, 15:42:49 :
```
WARNING | modèle de volume illisible
([Errno 2] No such file or directory: '/workspace/volume_v1.json') — bonus désactivé
```
Le launcher exporte pourtant `RELIQUARY_VOLUME_MODEL=/workspace/volume_v1.json`
et `VOLUME_MU=0.05`, et le code sait s'en servir (`engine.py:476-478` :
`score += VOLUME_MU * volume_score(modèle, texte)`). Le tri de tranche est donc
`score_prompt − 0,08 × risk_short`, **terme volume = 0**.
Le fichier existe en local : `data/volume_v1.json` (405 Ko, 20/08).

### La preuve que ça nous coûte : notre sélection = un tirage au sort
Simulation de priors à corrélation croissante avec le volume (copule
gaussienne, 80 tirages/fenêtre, 2 mappings bucket→P) :

| ρ(prior, volume) | P(rang≤20) | vs réel |
|---|---|---|
| **0,00 (aléatoire)** | **0,126-0,133** | **×0,93-0,98** |
| 0,50 | 0,189-0,225 | ×1,4-1,7 |
| **0,63 (= volume_v1 hors éch.)** | **0,208-0,255** | **×1,5-1,9** |
| 1,00 (oracle) | 0,290-0,369 | ×2,1-2,7 |

**ρ=0 reproduit exactement notre résultat réel (0,133 simulé vs 0,136 mesuré).**

### Volume contre valeur : le volume paie 3×, la valeur 1,5×
Plafond rétrospectif (n=339, arrivées inchangées) :

| tri | volume méd | P(rang≤20) |
|---|---|---|
| réel | 5 963 | 0,136 |
| oracle **valeur** | 6 704 | 0,202 (**×1,5**) |
| oracle **volume** | 8 958 | **0,449 (×3,3)** |

Le volume est **prédictible** : répétabilité du même prompt r=+0,785 (n=360),
R² de l'identité du prompt = **90 %**. Et il coûte peu d'arrivée :
**+0,63 s / 1 000 tokens** (intra-rafale ; décorrélé au niveau fenêtre, r=+0,03).

**Gain attendu : payées 0,39 → 0,6-0,75 par fenêtre.**
**Déploiement** : `scp data/volume_v1.json` → `/workspace/` + restart. Coût nul,
repli = supprimer le fichier (le code retombe sur « bonus désactivé »).

### 💰 CHIFFRAGE END-TO-END (2e agent, méthode indépendante)
Simulation de sélection sur 4 251 groupes v5 / 88 fenêtres, vivier médian 48
candidats, puis modèle de rang calibré + courbe payée(rang) :

| changement | Δ rang | payées/fen (top-8) | relatif |
|---|---|---|---|
| prior v5.1 → **v5.7** | −1,7 place | 1,07 → 1,30 | +21 % |
| **v5.1 + `volume_v1`** | **−3,9 places** | 1,07 → **1,54** | **+44 %** |
| v5.7 + `volume_v1` | −4,0 places | 1,07 → 1,55 | +45 % |

⚡ **Une fois le bonus de volume actif, changer de prior ne vaut plus que
~1 point relatif.** Le bonus apporte **+788 tok** contre **+352** pour le
meilleur prior. Et **~92 % du gain de rang de v5.7 passe par le VOLUME**, pas
par la valeur — v5.7 gagne surtout en choisissant incidemment des prompts longs.

**Test de robustesse décisif** : sur un vivier **indépendant du prior**
(explore+memo, ~17/fen), le gain volumique de v5.7 **s'effondre à −22 tok** au
top-8, alors que celui de `volume_v1` TIENT (+290 / +630). Le gain du prior est
donc en partie un artefact du vivier qu'il a lui-même filtré ; celui du modèle
de volume ne l'est pas.

`volume_v1.json` prédit le volume à **Spearman +0,576**, contre +0,198 pour
v5.1 et +0,346 pour v5.7 : c'est un modèle spécialisé sur la bonne variable.
Contrepartie mesurée : in_zone 95 % → 90 % (µ=0,05 ; µ=0,15 n'ajoute rien et
coûte de la valeur). Le prior sert alors surtout à protéger l'in_zone.

⚠️ **Fragilité n°1 du chiffrage** : le coefficient −4,94 places/1000 tok est
**transversal**, ajusté à marché figé. `canonical_rank` étant positionnel,
l'appliquer comme effet causal suppose que le peloton ne bouge pas. Le coût GPU
de générer plus long (moins de groupes bakés) n'est **pas** modélisé.

### ⚠️ CE QUI RECADRE TOUT : 85 % DE LA MATIÈRE NAÎT TROP TARD
Sur 47 groupes/fenêtre, **seuls ~7,2 (15 %) naissent assez tôt** pour être
postés avant la falaise (~24,5 s). Offset de génération médian : **+14,6 s pour
les envoyés contre +61,6 s pour les jamais envoyés**.
⛔ **« On génère 40 groupes et on n'en envoie que 4 » est une ILLUSION
COMPTABLE** que j'ai propagée toute la journée. Le reste n'était jamais
postable. Aucun prior ne récupère ça — c'est du DÉBIT et du TIMING.

### 🔑 Il n'y a AUCUNE sélection à l'envoi (code lu)
`_maybe_fire_on_append` (`engine.py:4073`) tire dès l'append ; `_fire_for_window`
draine le pool **dans l'ordre**, sans tri, filtré seulement par
cooldown/tranche/budget. **La seule sélection est l'ORDRE DE BAKE**, donné par
le prior. Réordonner la file d'envoi est donc structurellement un no-op —
ce que `scripts/test_traine_leviers.py` avait déjà constaté sans l'expliquer.

**Réserves de l'agent, à garder** : le mapping bucket→P repose sur 46 positifs
sur 339 et le bin décisif a n=15 (d'où les fourchettes) ; le contrefactuel
pioche dans le pool que le prior ACTUEL a produit, or un prior orienté volume
irait chercher d'autres prompts de la tranche, de distribution inconnue ; la
pénalité de 0,63 s/1000 tok est estimée sur le mix actuel, pas sur un régime
où toute la rafale serait longue.

## 🎯 LE PRÉDICTEUR EN PRODUCTION EST FAIBLE ET PÉRIMÉ (mesuré 24/08)

⛔ **PIÈGE D'ÉTIQUETAGE MAJEUR — le champ `score` du dump N'EST PAS une
prédiction.** Le code de la box écrit `"score": sigma * (1.0 - _mean)`
(`engine.py:1050`) = la **valeur d'enchère RÉALISÉE**, calculée après coup à
partir des rewards. Toute corrélation « score ↔ k » est donc une tautologie.
J'ai publié un « Spearman −0,775, le prédicteur est bon » qui ne mesurait que
ça, et un « les 8 têtes d'un bake sont déjà bonnes » circulaire (je les
classais par leur valeur réalisée). **Les deux étaient faux.**

### La vraie mesure : duel sur 2 618 groupes v5, INCONNUS des deux modèles
(données postérieures à l'entraînement des deux → aucun biais pro-incumbent,
le défaut qui avait invalidé 7 duels le 17/08)

| modèle | Spearman vs valeur réelle | lift top-10 % | k moyen du top-10 % |
|---|---|---|---|
| **v50 — EN PRODUCTION, 18/08 21:10** | **+0,226** | 1,32× | 8,3 |
| **v5.7 — 24/08 05:30** | **+0,507** | 1,45× | 8,0 |

**Les 5 candidats classés** (mêmes 2 618 groupes) :

| modèle | Spearman | lift top-10 % | k moy top-10 % |
|---|---|---|---|
| **v50 — EN PRODUCTION** | +0,226 | 1,32× | 8,3 |
| v5.2 (19/08) | +0,218 | 1,28× | 9,1 |
| v5.3 (20/08) | +0,245 | 1,34× | 8,7 |
| v5.4 (21/08) | +0,445 | 1,43× | 8,3 |
| **v5.5 / v5.6 / v5.7** | **+0,507** | **1,45×** | **8,0** |

**Cause de l'écart : la TAILLE DU CORPUS**, pas une dérive de modèle.

| modèle | corpus | holdout | lift (méta) |
|---|---|---|---|
| **v50 — EN PRODUCTION** | **3 807** | 0,39 | 1,21 |
| v5.3 | 18 331 | 0,427 | 1,28 |
| v5.4 | 40 113 | 0,443 | 1,49 |
| v5.5 → v5.7 | **63 376** (figé) | 0,515 | 1,63 |

Le v50 a été entraîné le 18/08 juste après une reconstruction de box, quand le
corpus venait d'être reconstitué : **16× moins de données** que le candidat.
Le plateau v5.5→v5.7 n'est PAS une panne d'entraîneur — le corpus est figé
parce que **le mineur était arrêté du 22/08 au 24/08**. Les md5 diffèrent bien.
Maintenant que le mineur tourne, la nuit prochaine aura des données v5.

Gain net sur la prod : corrélation **doublée**, valeur du top-10 % **+10 %**.
Reproduire : `scripts/duel_prior_v5.py`.

### ✅ v5.8 ENTRAÎNÉ LE 24/08 16:23 — le meilleur disponible, PAS DÉPLOYÉ
`python3 scripts/train_prior_v50.py` (inerte : aucun ssh/scp/restart, lit
`data/samples_v4.jsonl`, écrit `data/predictor_v5.8_2026-08-24_1623.json`).
Moins d'une minute.

| | v5.1 (en prod) | **v5.8** |
|---|---|---|
| corpus | 3 807 | **67 627** (dont 6,3 % v5) |
| Spearman (duel propre, n=11 113) | 0,299 | **0,525** |
| lift top-10 % | 1,21 | **1,69** |
| P(vedette \| top-20 % prédit) | — | **33,6 %** (base 14,3 %) |

Duel **symétrique et sans fuite** : échantillon postérieur aux DEUX
entraînements, prompts jamais vus. Verdict `CANDIDAT_GAGNANT`.

⛔ **NE PAS re-tester v5.8 sur nos groupes v5** (`scripts/duel_prior_v5.py`) :
ils sont DANS son corpus → biais pro-candidat, exactement le défaut qui avait
invalidé 7 duels le 17/08. Pour la même raison **v5.7 et v5.8 ne sont pas
départageables aujourd'hui** : toute donnée postérieure à v5.7 est déjà dans
le corpus de v5.8.

⛔ **BUG PIRE QUE PRÉVU dans `retrain_prior_nightly.sh:22`** :
`NEW=$(ls -t data/predictor_v50_*.json | head -1)` ne rate pas seulement les
nouveaux candidats — il pointe sur le **vieux fichier du 18/08**, qui serait
copié sur la box comme « candidat » si le duel était gagné. Promouvoir ce
candidat REMETTRAIT l'ancien modèle. Corriger le motif en `predictor_v5.*_*`.

**Déploiement (restart requis, go utilisateur)** :
`scp data/predictor_v5.8_2026-08-24_1623.json` → `/workspace/predictor_v50.json`
puis redémarrer. Repli : `data/predictor_v5.1_2026-08-18_2110.json` (md5
`559f115d…` = ce qui tourne aujourd'hui, vérifié).

### ⚠️ Pourquoi la prod tourne un modèle de six jours : un BUG de motif
`scripts/retrain_prior_nightly.sh:22` fait
`ls -t data/predictor_v50_*.json` alors que l'entraîneur écrit
`predictor_v5.3_*` … `predictor_v5.7_*`. **Le motif ne correspond JAMAIS**, donc
rien n'est promu depuis le 20/08. Cinq candidats dorment sur la dev box
(v5.3 → v5.7). L'entraînement nocturne tourne pour rien chaque nuit.

⚠️ Les deux modèles portent `protocol: 4` — entraînés sur l'ancien prompt.
Un ré-entraînement sur corpus v5 ferait probablement mieux encore.

**Déploiement** : copier le candidat en `/workspace/predictor_v50.json` +
redémarrer (le launcher lit `RELIQUARY_PROMPT_PREDICTOR`, défaut ce chemin).
Repli : le v50 actuel est sauvegardé en `data/predictor_v50_2026-08-18_2110.json`.

### 📈 Le prédicteur ne dérive PAS avec les checkpoints
Mesuré sur les checkpoints 524 → 609 : le pouvoir discriminant tient. Le
volume, lui, a bondi de **+45 %** (4 345 → 6 362 tok) entre le dernier
checkpoint v4 et le premier v5 — c'est **l'effet du prompt v5**, pas une
dérive du modèle, et il est plat à l'intérieur de v5. ⚠️ La note « le modèle
RACCOURCIT, −27 tok/checkpoint » ne vaut que pour l'ère v4.

## 📉 CE QUI A ÉTÉ MESURÉ ET REJETÉ LE 24/08

- **`MAX_INFLIGHT_FIRES` 3 → 6** : attente livraison→POST **inchangée** (4,19 →
  4,46 s). La file n'était PAS le goulot tant que le sprint étalait les
  livraisons. ⚠️ MAIS quand les 8 groupes arrivent groupés (sprint off), le
  POST tombe de 3,66 à 1,56 s — donc le réglage est GARDÉ comme filet, il ne
  coûte rien.
- **`SHORT_RISK_LAMBDA=0`** : gain réel **0,38 s** sur 2,65 s de classement
  (14 %). Le malus est bien ACTIF (journal : « malus anti-court ACTIF … 40659
  tokens »). Marginal, non déployé.
- **Sauter les sha256 / `json.dumps` du classement** (recommandé par un agent) :
  **0,07 s seulement**, pas les 1,5 s supposées. La notation pure (score +
  risk) ne pèse que **0,75 s sur 2,65 s** — les 1,9 s restantes ne sont PAS
  identifiées (mon banc de lecture parquet s'est révélé non exploitable :
  19 ms/prompt contre 0,53 mesuré en prod, cache froid).
- **Augmenter `BAKE_BATCH_SIZE`** : inutile. On bake 8 groupes, on en place 4,
  et **0,42 seulement arrive sous 12 s**. Ajouter des candidats les envoie au
  balayage, qui n'arrive JAMAIS sous 12 s (0 sur 30 mesurées).

## ⏱️ OÙ PASSE LE TEMPS ENTRE « GÉNÉRÉ » ET « ENVOYÉ » (mesuré 24/08, 345 envois)

| étape | médiane | moyenne | p90 | part |
|---|---|---|---|---|
| grading | 0,07 s | 0,34 s | 0,10 s | négligeable |
| **preuve GRAIL** | **1,17 s** | **1,98 s** | 5,69 s | **33 %** |
| **POST validateur** | **1,95 s** | **4,09 s** | 10,75 s | **67 %** |
| total | 4,38 s | 6,06 s | | |

**Mais tout ce délai ne coûte PAS de l'arrivée.** Test apparié à l'intérieur
d'une même fenêtre (contrôle la charge du validateur) :

| variable | paires | effet sur l'arrivée |
|---|---|---|
| délai total généré→envoyé | 961 | **0,08 s par seconde** (pente nulle) |
| **preuve GRAIL seule** | 624 | **0,63 s par seconde** |

⚡ **Le POST est HORS du chemin critique** — `MAX_INFLIGHT_FIRES=6` le
parallélise, donc un POST lent ne retarde pas les autres entrées. Le réglage
que j'avais jugé « inutile » le matin même est en fait ce qui neutralise les
4 s du validateur. **NE PAS le retirer.**

⚡ **La preuve GRAIL est SUR le chemin critique** : ~1,98 s dont 63 % se
répercutent = **~1,25 s d'arrivée récupérables**. C'est le seul poste de
latence encore attaquable après l'envoi. ⚠️ Ne pas confondre avec
`RELIQUARY_SPEC_PROOF` (rejeté : il parallélise la preuve avec le *grading*,
qui ne coûte que 0,07 s).

**Le validateur est lent en ce moment** : 3 `httpx.ReadTimeout` sur le
precommit (15h51, 15h53) et son API dashboard met **29 s** à répondre.

## 🎯 CE QUI PAIE VRAIMENT EN v5 — mesuré, sans référence à v4

**Le rang décide, et le seuil est net** (nos entrées v5, 202 admises décidées) :
| rang | payées |
|---|---|
| 13-16 | **100 %** |
| 17-20 | 50 % |
| 21-25 | 14 % |
| 26-34 | 5 % |
| **35+** | **0 %** |

Notre rang médian est **35-37** : pile à la frontière du zéro.

**Le volume prime sur la vitesse** :
| quartile de volume | volume | arrivée | payées |
|---|---|---|---|
| Q1 | 4 558 | +19,8 s | **4 %** |
| Q4 | **7 735** | +21,4 s | **29 %** |

Portrait-robot : une entrée **payée** a 7 267 tok et arrive à +12,8 s ; une
**non payée** a 5 818 tok et arrive à +19,6 s. Les deux comptent, mais le
volume n'est pas sacrifiable — et la VALEUR d'enchère passe avant les deux.

⚠️ **La falaise d'admission est à ~24,5 s en v5**, pas 12 s comme en v4 : tout
le monde génère plus long, donc tout le monde arrive plus tard. **L'admission
n'est PAS le problème** (58 %), c'est le RANG.

## 🚨 ÉTAT AU 24/08 — VALIDATEUR EN v5, MINEUR PORTÉ, EN ATTENTE DE BOX

**Le validateur tourne `protocol_version 5` depuis le 23/08** (PR #190, image
`cba84ce`, profil `qwen3-4b-base-dapo-reasoning-v5`, status `ok`).
⛔ **Relancer en v4 = 100 % de rejet** — le prompt fixe les tokens et les
tokens le forced-seed.

### ✅ LE PORT v5 EST FAIT, TESTÉ, POUSSÉ — commit `9cc2070`
Le SEUL changement qui nous concerne est **le PROMPT**, désormais rendu via un
template versionné publié dans le `generation_contract` :

| | v4 | v5 |
|---|---|---|
| prompt code | `énoncé + contrat` (172 car.) | **enveloppé** (321 car.) |
| préambule | — | `Solve the following programming problem step by step.` |
| consigne finale | — | `…provide the final implementation in the last fenced Python code block.` |

**Vérifié IDENTIQUE à v4** (donc rien d'autre à porter) : sampling (16
rollouts, T=1,0, top_p 1,0, top_k 0), `max_new_tokens` 8192, `bft=null`,
`token_cap` 8192, `collection_seconds` 100, `upload_grace` 33. Les ajouts de
`constants.py` upstream ne touchent que leur trainer détaché (PR #189).

**Parité vérifiée contre le validateur LIVE**, pas contre leur code :
sha256 de nos templates == ceux de `/health` (`opencodeinstruct` 47f2d9e1…,
`openmathinstruct` 7f234305…). `reliquary/protocol/profiles.py` (créé) +
1 ligne dans `environment/opencodeinstruct.py`. 12 tests, **0 régression**
(44 échecs avant, 44 après — vLLM/GPU/fixtures pré-existants).
**Repli** : `RELIQUARY_PROTOCOL_VERSION=4` → chemin legacy byte-exact, testé.

✅ `_extract_python` prend déjà le **dernier** bloc fenced (`matches[-1]`) —
conforme à la nouvelle consigne v5, et non modifié upstream. Rien à faire.

### 🔭 HYPOTHÈSE À VÉRIFIER DÈS LES PREMIÈRES FENÊTRES
Le prompt v5 demande de **raisonner étape par étape** → les complétions
devraient RALLONGER. Or le rang est `tokens // (rounds × 50)` et notre médiane
était tombée à 3 867 tokens, juste sous le seuil de payabilité au round 3.
⚠️ Deux réserves : ça vaut pour TOUT LE MONDE (effet relatif possiblement nul),
et générer plus long coûte du temps, ce qui pousse vers le round 4. **Mesurer,
ne pas supposer.** (Le modèle, lui, RACCOURCISSAIT : −27 tok/checkpoint.)

### 🔑 ÉTAT DE LA BOX ET DE LA HOTKEY
- **Hotkey TOUJOURS ENREGISTRÉE** : uid 167, vérifié le 24/08 sur
  `https://www.reliqua.ai/api/miners` (status `offline`, rang 166 — dégradé
  par l'inactivité, mais la place est gardée).
- **Box `38.255.28.21` PERDUE** : port 20098 fermé. Le 20100 répond en SSH mais
  **refuse notre clé** (autre conteneur) ; le 20099 ne parle pas SSH.
  **PREMIER GESTE : récupérer port + accès chez Lium**, puis
  `ops/RECONSTRUCTION_BOX.md` (9 étapes, ~40 min).
- ⚠️ Le launcher versionné contient désormais `PROTOCOL_VERSION=5`,
  `MIN_ROLLOUT_LEN=0` et le retrait de `HF_XET`. Ces deux derniers n'avaient
  JAMAIS été commités : ils vivaient dans `/workspace` de la box perdue.

## 🔬 A/B EN COURS — retrait du gate anti-rollout-court (À CONCLURE)

Déployé au redémarrage de **17:15:21**, coupure à la **fenêtre 30177**.
Changement : `RELIQUARY_MIN_ROLLOUT_LEN` 32 → **0** (+ retrait de
`HF_XET_HIGH_PERFORMANCE`, inerte). Rien d'autre n'a bougé.

**Commande pour conclure** :
`python3 scripts/ab_gate.py --comparer 30177 --depuis 30182`
(`--depuis` écarte les 3 fenêtres de rodage : ne JAMAIS juger un moteur froid.)

**Données disponibles** : 18 fenêtres de bras B, **70 admises dont 56 verdicts
décidés (80 %)**. Sous-dimensionné — il en faudrait ~100 pour trancher l'écart
observé — mais analysable.

**Ce que la mesure disait à l'arrêt** :
| échantillon | écart sur les admises/fen |
|---|---|
| 5 fenêtres | +0,73 |
| 7 fenêtres | +0,38 |
| **10 fenêtres** | **+0,03** — IC95 [−1,00 ; +1,06] |

L'estimation **converge vers zéro** quand l'échantillon grossit : signature d'un
effet qui n'existait pas. Mon pronostic à l'arrêt : **non concluant, voire
légèrement négatif.**

**✅ LA SÉCURITÉ EST ÉTABLIE** : 9 entrées courtes envoyées, 7 verdicts décidés,
**ZÉRO `logprob_mismatch`, ZÉRO fenêtre à dette**. Le contrôle à couverture
complète de PR #188 passe bien sur nos groupes. Cette question-là est close.

**Rendement des entrées courtes : 1 payée sur 7 (14 %)**, contre 25-30 % pour
les entrées normales. Si ça se confirme, les réadmettre revient à remplacer des
entrées ordinaires par des entrées moins bonnes.

⚠️ **POURQUOI le gain était nul, compris trop tard** : on bake 100-140 groupes
par fenêtre et on n'en place que 3-5. **La contrainte est le TEMPS, pas le
stock** — la porte se ferme vers 10 s. Les entrées courtes se SUBSTITUENT aux
autres au lieu de s'y ajouter. Mon chiffrage initial de +1,5 entrée/fenêtre
supposait l'inverse : il était faux.

## 📉 LA BAISSE DU 21/08 — CE QU'ELLE EST ET CE QU'ELLE N'EST PAS

**Elle commence à 14h47**, deux heures et demie AVANT notre changement :
| tranche | payées/h |
|---|---|
| 08h47 | **13,1** |
| 12h47 | 9,8 |
| **14h47** | **5,6** ← décrochage, avant le fix |
| 16h47 | 5,5 ← après le fix, identique |

⛔ **Ne PAS attribuer la baisse au retrait du gate.** Et ne pas comparer des
agrégats « avant/après » : le « avant » moyenne toute la journée et fabrique
un faux écart. C'est le piège qui m'a fait accuser PR #188 à tort.

**Cause établie par 4 agents** : le peloton a gagné **6 secondes** en 24 h
(lag marché p50 17,1 → 11,0 s, 8 heures consécutives hors amplitude). Le rang
étant `tokens // (rounds × 50)`, passer de 6 à 4 rounds leur donne +50 % de
bucket sans changer un token. Notre rang glisse de **+0,26 à +0,45 place par
heure à volume ET arrivée constants** (t = 2,7 à 5,2).
Le peloton **ne s'épaissit PAS** : 49 mineurs avant/après, et le compteur du
validateur donne 59 → 56 candidats par fenêtre (en légère BAISSE).

**Notre pipeline est intact** : génération plate, rejets internes plats
(40-46 %), file d'envoi améliorée, 0 Traceback / 0 OOM sur 32 000 lignes.

**Artefact à connaître** : le validateur a raccourci son cycle de fenêtre vers
08h (273-303 s → 186-225 s, **+30 % de fenêtres/heure**). Tout ce qui se compte
« par fenêtre » baisse donc mécaniquement. **COMPTER PAR HEURE.**

## 🚀 OBJECTIF : RETROUVER LA VITESSE DE CROISIÈRE

**Référence à reconquérir : 13,1 payées/heure** (21/08 08h47, 27 fenêtres).
Au moment de l'arrêt : 5,5/h. Il manque donc **plus de la moitié**.

⚠️ **LE RETOUR ARRIÈRE NE SUFFIRA PAS.** La chute se produit à 14h47, deux
heures et demie AVANT le retrait du gate, et la tranche qui suit le fix (5,5)
est identique à celle qui le précède (5,6). La configuration d'avant produisait
déjà 5,6 : y revenir ne ramène pas les 13,1.

**Ce qui a changé, mesuré** : le peloton a gagné 6 secondes (lag p50 17,1 →
11,0 s). Comme le rang est `tokens // (rounds × 50)`, passer de 6 à 4 rounds
leur donne +50 % de bucket sans un token de plus. Notre volume et notre arrivée
n'ont pas bougé — c'est donc notre POSITION RELATIVE qui s'est dégradée, de
0,26 à 0,45 place par heure.

**Il n'y a donc qu'une famille de réponses : gagner un round nous aussi.**
À 3 rounds il faut ~3 750 tokens pour être payable, à 4 rounds il en faut 5 000.
Notre médiane est à ~4 000 : on est payable au round 3, jamais au round 4.
**Gagner un round vaut plus de 1 000 tokens** — et aucun levier de volume ne
donne 1 000 tokens (μ=0,05 en donne ~600, et coûte de l'arrivée).

### Les trois candidats, par ordre de valeur attendue
1. **SPRINT** (`ops/AB_SPRINT.md`) — le seul déjà codé qui vise le round.
   BAISSER `SPRINT_SIZE` de 8 à 4 ou 2. Mesure : 7 à 22 fenêtres par bras.
2. **Le délai flip → 1er groupe utilisable = 4,25 s** (1,15 s avant le bake,
   3,1 s de génération). Chantier de fond, pas une variable.
3. **La fenêtre perdue à chaque checkpoint** — 1 à 2 par heure, structurel
   (rechargement 56 s contre un batch qui ferme à 12 s).

### Ce qui NE ramènera PAS la vitesse de croisière (mesuré aujourd'hui)
Le bonus de volume, les slots d'exploration, les trois leviers de traîne,
`SPEC_PROOF`, la queue du POST, et le retrait du gate lui-même. Tous mesurés,
tous insuffisants ou nuls. **Ne pas y retourner sans raison nouvelle.**


## 🎯 CE QU'IL RESTE À FAIRE — par ordre de valeur

1. **Conclure l'A/B du gate** (commande ci-dessus) et décider : garder à 0, ou
   remettre 32. Le gate ne coûte PAS de rang (testé : corrélations −0,09), il
   ne joue que sur le nombre d'entrées.
2. **LE SPRINT — le seul levier qui vise le ROUND.** Neutralisé depuis le
   18/08 : le launcher exporte `SPRINT_SIZE=8` = `BAKE_BATCH_SIZE=8`, or le
   code fait `if n_sprint >= n: n_sprint = 0`. **Il faut le BAISSER, pas
   l'augmenter** : 4 → sprint sur 4 prompts = 64 séquences en vol au lieu de
   128 ; 2 → 32 séquences. Protocole complet : `ops/AB_SPRINT.md`, métrique le
   bucket (7 à 22 fenêtres par bras selon l'effet). C'est le levier qui répond
   au durcissement, puisque **gagner un round vaut plus de 1 000 tokens**.
3. **Le délai flip → premier groupe utilisable : 4,25 s** (1,15 s avant le
   bake + 3,1 s de génération). C'est le vrai chantier de fond, pas une
   variable d'environnement.

## 📊 LES ÉTUDES DU 21/08 — scripts prêts, verdicts rendus

| script | question | verdict |
|---|---|---|
| `scripts/ab_gate.py` | le retrait du gate paie-t-il ? | **à conclure** |
| `scripts/ab_sprint.py` + `ops/AB_SPRINT.md` | le sprint paie-t-il ? | **non lancé** |
| `scripts/test_volume_mu.py` | le bonus de volume ? | **+2,4 % à μ=0,01, +14,6 % à 0,05 — il en faut +32 %. Insuffisant.** |
| `scripts/test_traine_leviers.py` | la traîne est-elle exploitable ? | **3 leviers morts** (file d'envoi no-op, prédiction +0,142, hash −0,010) |
| `scripts/etude_bake_ordre.py` | pourquoi le gros groupe arrive tard ? | le plus volumineux sort en position 6-8 dans **87 %** des bakes |
| `scripts/train_volume_v2.py` | ré-entraîner le volume ? | aucune fenêtre glissante ne bat le v1 figé |
| `scripts/train_traine.py` | prédire la traîne ? | Spearman +0,142, abandonné |
| `scripts/watch_rollouts_courts.py` | vigie du fix | ABANDON au 1er `logprob_mismatch` |

**Autres résultats à ne pas re-dériver** : `SPEC_PROOF` est inutile (le grading
coûte 0,06 s, rien à paralléliser) ; le chemin de preuve fusé est déjà actif ;
la queue du POST coûte 1,4 fenêtre sur la période ; les slots d'exploration
2→1 donnent +6,3 % de volume mais **aucun effet sur le paiement**.


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

⚠️ **PANNE VÉCUE LE 24/08 — `poll_dashboard.sh` muet depuis 13h39** : l'API
`reliqua.ai/api/miners` a ralenti à **29 s** de réponse, or le script utilisait
`--max-time 10` → 100 % de timeouts, **silencieusement** (`|| true`). Corrigé à
45 s (sauvegarde `.bak-2026-08-24`). **Conséquence coûteuse** : aucune donnée de
marché sur les fenêtres 31812-31827, donc la chute de rang de cette période est
**définitivement non diagnosticable** — on ne peut pas distinguer « on a
régressé » de « le peloton a accéléré ». ⛔ Vérifier que le fichier GROSSIT, pas
seulement que le tmux tourne.

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
