# CONSTAT MONITOR — session du 2026-08-25

Relevé des anomalies détectées pendant la surveillance en direct du mineur
(box H200 `root@157.10.163.2:20299`, uid 167).

**Période surveillée : fenêtres 32574 → 32630, 14:05 → 15:53 UTC (1 h 48).**
53 fenêtres, 406 tentatives, **197 acceptées** (3,7/fenêtre), 178 verdicts
décidés, **48 payées = 26,7/heure**.

| motif de rejet | n |
|---|---|
| `stale_round` | 117 |
| `batch_filled` | 63 |
| `window_mismatch` | 14 |
| `wrong_checkpoint` | 13 |
| `precommit_invalid` | 2 |

⚠️ **Ces chiffres incluent deux vagues de lenteur du validateur** (§4), dont la
seconde a dégénéré en **panne de 28 min** (§4bis). Ne pas les prendre comme
ligne de base d'un réglage. La surveillance s'est poursuivie après la reprise
(fenêtre 32631 : 8 acceptées à 7-23 s, régime normal retrouvé).

---

## 0. 📋 INVENTAIRE DE **NOS** ERREURS — 9 POSTES, 2 QUI COÛTENT

Récapitulatif côté mineur uniquement (le côté validateur est au §4quater).

### Ce qui coûte
| # | notre erreur | volume | coût mesuré |
|---|---|---|---|
| **1** | **Durée du rechargement de checkpoint** | 12/jour · 121 s en cache, 365 s avec téléchargement | **~97 payées = 9,8 % du revenu** |
| **2** | **Échecs de conformité** | **32 / 3 895 verdicts (0,82 %)** | **~15 payées** |

Détail du poste 2 — les seuls vrais défauts de notre code :

| motif | n | dont zone payante (rang ≤20) |
|---|---|---|
| `logprob_mismatch` | 14 | **12** |
| `reward_mismatch` | 10 | 0 |
| `token_tampered` | 3 | 2 (rangs 6 et 10) |
| `bad_termination` | 3 | 2 |
| **`seed_mismatch`** | 2 | **2 (rangs 1 et 3)** |
| **total** | **32** | **18** |

⚠️ **Estimation corrigée** : d'abord annoncée à ~9 payées en pondérant par la
BANDE D'ARRIVÉE, elle passe à **~15,3** en pondérant par le **RANG réellement
obtenu** — estimateur plus direct puisque ces entrées avaient un rang.
Taux de paiement mesuré : **97,8 % au rang ≤5**, 93,8 % au rang ≤15, 51,1 % au
rang ≤20, **3,5 % au-delà de 20**. Un échec sur une entrée bien classée est
presque toujours une payée perdue.

### Ce qui ne coûte RIEN — vérifié, ne pas y toucher
| # | notre « erreur » | mesure |
|---|---|---|
| 3 | 12 redémarrages `FUITE_VRAM` + 2 `PROCESS ABSENT` | **coût indétectable** (8,2 % vs 12,7 % de référence) |
| 4 | 1 `WEDGE` inutile pendant la panne validateur | aucune conséquence |
| 5 | Bug `watchdog.sh:98` | cosmétique, pollue le journal |
| 6 | `checkpoint pull failed` horaire | bénin, snapshot déjà à jour |
| 7 | 4 défauts de l'outillage de surveillance | **corrigés ce jour** (§9) |
| 8 | 1 faux positif du détecteur 🥶 | **corrigé** (2 fenêtres de recul) |

### À nous, mais NON EXPLIQUÉ
| # | | |
|---|---|---|
| 9 | **Variance de la 1re arrivée** (4,7 s → 31,9 s) | ni l'envoi, ni le vidage du pool (testés et réfutés). Le retard naît **entre le flip et le premier groupe utilisable** — chantier n°3 du CLAUDE.md, levier le plus rentable identifié (1 s ≈ 1,46 place). |

🎯 **Deux postes comptent : la durée du rechargement vLLM (~10 % du revenu) et
les 32 échecs de conformité (~15 payées).** Tout le reste est sans coût
mesurable ou déjà corrigé.

---

## 0bis. 🔀 CHANGEMENT DE RÉGIME DU MODÈLE — 25/08 ~21h00

**Le modèle s'est remis à produire des rollouts très longs.** Ce n'est pas la
longueur TYPIQUE qui a changé, c'est la **QUEUE de distribution**.

| période | n | médiane | p90 | **part > 1500 tok** |
|---|---|---|---|---|
| avant (19:00-21:19) | 404 | **612** | 880 | **1,5 %** |
| entre les 2 restarts (21:21-22:09) | 36 | **1114** | 1859 | **22,2 %** |
| après restart 2 (22:09→) | 33 | **1144** | 1968 | **24,2 %** |

⚠️ **Chiffres corrigés.** Ma première lecture donnait « médiane 798, 8,1 % » —
elle découpait par heure, donc la tranche « 21h » contenait 20 minutes
d'ANCIEN régime. En bornant sur l'événement réel : **médiane × 1,9** et part de
très longs **× 15** (1,5 % → 24 %). C'est ce qui explique enfin pleinement des
bakes passés de 11 s à 20-57 s.

⚡ **Pourquoi la queue compte plus que la médiane** : un bake attend le rollout
**le plus long de ses 128 séquences en vol** (8 groupes × 16). À 1 % de très
longs, un bake sur trois en contient un ; à 8 %, presque chaque bake en contient
plusieurs. **Le temps de bake suit la QUEUE, pas la médiane.**

### Conséquences mesurées
| | avant 21:21 | après |
|---|---|---|
| bake complet (groupe 8/8) | **11 s** (médiane stable 17h→21h) | **20-57 s** |
| groupes générés / fenêtre | ~47 | **15-27** |
| acceptées / fenêtre | **6,2** | **2,0** |
| 1re arrivée | 6-9 s | 10-31 s |

### ✅ MAIS le rang par entrée s'AMÉLIORE
Le volume entre au numérateur du bucket. À arrivée égale, on gagne des places :
rang **9 payé pour une arrivée à 21,9 s** (`fen 32808`), **8 à 14,3 s**
(`fen 32805`), là où ces offsets valaient le rang 45-50 le matin.
### ⚖️ SOLDE NET — bien moins mauvais qu'annoncé d'abord
| période | fen mûres | acceptées/fen | **payées/fen** |
|---|---|---|---|
| avant 21:21 (2 h) | 32 | 4,9 | **0,56** |
| après 21:21 | 9 | **1,8** | **0,44** |

**2,7× moins d'entrées, mais seulement −21 % de payées par fenêtre.** Le taux de
conversion passe de **11 % à 24 %** : les entrées survivantes sont plus
volumineuses donc mieux classées.

⛔ **Mon premier chiffrage (« ~20 payées/h → ~6 ») était FAUX** : il comparait
une heure pleine à 30 min, en comptant les verdicts indécis comme non payés.
Sur **fenêtres mûres des deux côtés**, l'écart tombe à 0,56 → 0,44.
⚠️ 9 fenêtres mûres — le seuil du projet est 30. L'écart n'est pas
distinguable du bruit à cet effectif.

### ✅ VERDICT À 29 FENÊTRES MÛRES — aucun effet démontrable sur le revenu
| | ancien (49 fen) | **nouveau (29 fen)** |
|---|---|---|
| acceptées/fenêtre | 4,92 | **2,28** |
| **payées/fenêtre** | **0,84** | **0,59** |
| **rang ≤20** | 22,0 % | **28,8 %** |
| 1re arrivée | 8,9 s | 16,6 s |
| `max_len` médian | 631 | **1122** |

**Écart −0,25 payée/fenêtre, p = 0,307 (permutation, unité = fenêtre) — NON
SIGNIFICATIF.** Malgré 2,2× moins d'entrées et +7,7 s d'arrivée, le revenu par
fenêtre n'est pas distinguable du bruit, et **la qualité par entrée MONTE**
(rang ≤20 : 22,0 % → 28,8 %).

⚠️ **La ligne de base est très instable** : le bras « avant » vaut 0,56
payée/fen sur 2 h et **0,84 sur 3 h**. Le point de coupure change le résultat
de 50 % — aucune des deux lectures ne permet d'affirmer une perte.

🎯 **DÉCISION : NE RIEN CHANGER.** La piste `BAKE_BATCH_SIZE` interviendrait sur
un problème non prouvé et risquerait de **casser le gain de rang** qui compense
la perte de débit. Le changement déplace l'équilibre (moins d'entrées, mieux
classées), il ne détruit pas le revenu.

### 🆕 EFFET SECONDAIRE — `HTTP 408 submission_body_timeout` (26/08 00h29)
Les complétions plus longues font grossir le payload de soumission :

| | n | médiane | p90 | p99 | max |
|---|---|---|---|---|---|
| avant 21:21 | 691 | 391 ko | 489 ko | 681 ko | 862 ko |
| **après 21:21** | 286 | **592 ko** | **749 ko** | **1072 ko** | **1257 ko** |

**+51 % en médiane**, maximum 862 ko → **1,26 Mo**. C'est exactement quand
apparaissent les **premiers `HTTP 408` de tout l'historique** (4 occurrences,
00:29 et 00:31) : le validateur abandonne pendant la réception du corps.

⛔ **Ne PAS croire le message du mineur** : `submitter.py:354-359` étiquette
« miner-side payload bug » **tout** code HTTP ≥ 400 sur le precommit. C'est
juste pour un 400/422 (payload malformé), **faux pour un 408** qui signifie
`Request Timeout` — un téléversement lent, pas un payload invalide. Réseau
vérifié sain au même moment (`connect` 0,19 s, aller-retour 0,59 s).

⚠️ **À surveiller** : ce motif frappe sélectivement nos **plus gros groupes**,
donc les mieux classés. C'est le seul des trois effets du régime long qui
touche préférentiellement nos meilleures entrées.

### ⛔ Ce que ce N'EST PAS
| hypothèse | vérification |
|---|---|
| retour au modèle de base | `checkpoint_n` **continue de monter** (659 → 660), dépôt inchangé |
| **le redémarrage du mineur** | ⚡ **FACTEUR CONFONDANT LEVÉ** — un restart watchdog `FUITE_VRAM` a eu lieu à **21:19:54**, 2 min avant. Mais un **second restart à 22:09** n'a **rien changé** (médiane 1114 → 1144, part >1500 22,2 % → 24,2 %). Le régime est une propriété du CHECKPOINT, pas un état du processus. |
| cache KV réduit après rechargement | **identique** : 96,77 GiB / 704 656 tokens aux 4 derniers inits |
| rodage JIT | compilations Triton datées de **21:21:15 uniquement**, aucune depuis |
| mineur malade | GPU 72 %, charge 1,24, groupe 1/8 toujours prêt en 5-7 s |
| marché ralenti aussi | **NON** — lag marché p50 **stable à 8 s**, 32 acceptées/fenêtre |

⚠️ Le marché n'étant pas affecté, **le guichet ne se décalera pas** : c'est une
perte de position relative tant que le régime dure.

🔭 **Levier identifié, NON appliqué** : on bake 8 groupes en parallèle ; à queue
épaisse, les 8 finissent trop tard. Baisser `BAKE_BATCH_SIZE` sortirait les
PREMIERS groupes plus tôt — or seuls les 3-4 premiers paient. Variable pure,
réversible. **À ne pas appliquer sans mesure** : un redémarrage coûte une
fenêtre certaine, et le régime peut disparaître au checkpoint suivant.

---

## 1. 🥶 RECHARGEMENT DE CHECKPOINT — 10,6 % DES FENÊTRES

**C'est l'anomalie la plus coûteuse de la session, et elle est structurelle.**

### Hypothèse de longue date CONFIRMÉE
**12 rechargements depuis 10:04** (un toutes les **21-27 min**) pour **12 PID
d'`EngineCore` DISTINCTS** → vLLM se réinitialise **entièrement** à chaque
avancée de checkpoint (poids en VRAM + graphes CUDA). L'hypothèse était notée
« non vérifiée » au CLAUDE.md depuis le 25/08 matin ; elle l'est maintenant.
❓ **Reste ouvert** : vLLM expose-t-il un échange de poids à chaud ? Non
vérifié dans leur source.

### Séquence type — une avancée coûte 2 à 3 fenêtres
Observée à l'identique **4 fois en direct** (14:14, 14:41, 15:08, 15:35) :

| temps | fenêtre | ce qui se passe |
|---|---|---|
| 1 | N | **0 acceptée** — `wrong_checkpoint` puis `window_mismatch` |
| 2 | N+1 | **aucune trace**, trou d'activité 121 s (vLLM redémarre) |
| 3 | N+2 | le moteur revient, guichet déjà plein → `batch_filled` |
| 4 | N+3 | reprise pleine |

Log témoin : `checkpoint advanced mid-iteration (reload stall); skipping fire
for stale window=32576` à 14:14:46.

### Coût chiffré — 838 fenêtres, 47 avancées, **1,89 fenêtre perdue par avancée**

| fenêtre à 0 acceptée | n | part |
|---|---|---|
| **invisible** (aucune ligne nulle part) | 48 | 5,7 % |
| **reprise tardive** (guichet plein) | 31 | 3,7 % |
| aucune tentative (vue mais muette) | 5 | 0,6 % |
| tout refusé | 5 | 0,6 % |
| *autre cause, hors checkpoint* | *17* | *2,0 %* |
| **→ imputable au checkpoint** | **89** | **10,6 %** |

⚠️ La ligne « reprise tardive » est imputée **par proximité** (0-3 fenêtres
après une avancée datée), pas par une trace directe. C'est le maillon faible
du décompte.
⛔ **SURESTIMÉE — règle resserrée le 25/08 22h.** Cas `fen 32815` : marquée
« reprise après rechargement » alors que l'avancée datait de 3 fenêtres plus
tôt **et que la fenêtre intercalaire 32814 était SAINE** (3 acceptées, 1 payée).
La règle n'impute plus qu'une **CHAÎNE ININTERROMPUE de fenêtres à 0 depuis
l'avancée**. Elle sous-compte plutôt qu'elle ne gonfle — le bon sens de l'erreur
pour la ligne la plus fragile.
👉 **Le total « 10,6 % imputable au checkpoint » est donc un PLAFOND**, pas une
valeur centrale. Le noyau dur (invisible + aucune tentative + tout refusé) vaut
**6,9 %**.

### ⚠️ DEUX RÉGIMES DE RECHARGEMENT — le coût n'est PAS fixe
| cas | trou d'activité | fenêtres perdues |
|---|---|---|
| checkpoint **déjà en cache** (réinit vLLM seule) | ~121 s | **2** |
| checkpoint **à télécharger** depuis HF | **365 s** | **3** |

Cas mesuré le 25/08 à 16:41 (`checkpoint 649 -> 650`, snapshot `b70559ec`) :
silence total de 16:35:44 à 16:41:40, puis reprise en 9 s (groupe 1/8 à 3,5 s).
⚠️ **La perte commence AVANT l'événement journalisé** : le log ne date
l'avancée qu'à la fin du téléchargement (16:41:40) alors que l'activité s'est
arrêtée à 16:35:44. Une attribution par proximité de `checkpoint_n` étiquettera
donc la première fenêtre du trou en « cause inconnue » — c'est un faux négatif
attendu, pas un défaut du détecteur.

✅ **Irréductible côté protocole** : `checkpoint_hash` entre dans `u_at`, donc
tout groupe généré sous l'ancien hash est un `SEED_MISMATCH` garanti. Le vidage
du pool est imposé, pas optimisable.

---

## 2. 🕳️ 5,7 % DES FENÊTRES SONT INVISIBLES DANS LES DONNÉES

**48 numéros sur 830** (31755→32584) sont absents **des DEUX** fichiers —
`submits_v4.jsonl` ET `windows_v4.jsonl`. Le rechargement avale la fenêtre
avant qu'une seule ligne soit écrite.

⛔ **Tout compteur fondé sur ces fichiers les rate silencieusement.**
C'est cet angle mort qui m'a fait annoncer « 1,2 % de fenêtres perdues au
checkpoint » en début de session — **faux par construction**, le vrai chiffre
est 10,6 %.

**Comment les retrouver** : chercher les **trous dans la numérotation**, qui est
contiguë. Et attention — **l'avancée de `checkpoint_n` est enregistrée sur la
fenêtre SUIVANTE**, pas sur la fenêtre perdue (vérifié : fenêtre perdue 32577
→ avancée notée en 32578 ; idem 32559 → 32560).

---

## 3. 🔁 WATCHDOG — 11 REDÉMARRAGES `FUITE_VRAM` DANS LA JOURNÉE

Horaires : 01:18, 03:05, 04:27, 05:33, 06:29, 07:26, 08:23, 09:23, 11:17,
12:36, 13:43, **17:32** — plus 2 `PROCESS ABSENT` (10:04, 12:47) et 1 `WEDGE`
(16:10, pendant la panne validateur, cf. §4bis). **Soit ~1 redémarrage/heure.**

Seuil `WATCHDOG_VRAM_MAX=135000` MiB (`watchdog.sh:81`). Les déclenchements
tombent entre **135003 et 135503 MiB** : une marge de **0,002 % à 0,4 %**.
⚡ Le 12ᵉ redémarrage du jour (17:32:00) s'est déclenché à **135003 MiB — 3 MiB
au-dessus du seuil**. On ne mesure plus une fuite, on mesure du bruit.
VRAM au repos mesurée : **128,5 Go / 143,7 Go**. Ça ressemble à une garde qui
mord sur le **pic normal** (vLLM ~109 Go à 0,76 + preuves GRAIL ~27 Go), pas
sur une fuite.

⛔ **NE PAS en faire un chantier — le coût est INDÉTECTABLE.** Sur les 12
redémarrages du jour, **8,2 %** des fenêtres qui suivent ont ≤1 acceptée,
contre **12,7 %** de référence toutes fenêtres. C'est-à-dire *mieux* que la
moyenne.

### ⚠️ Bug : `watchdog.sh:98`
```
/workspace/watchdog.sh: line 98: [: 0
0: integer expression expected
```
En boucle dans `watchdog.log`. `pgrep -fc code_grader_driver` renvoie deux
valeurs sur une ligne. La garde fonctionne, mais le journal est pollué.

---

## 4. 🐌 DEUX VAGUES DE LENTEUR DU VALIDATEUR

Découpage par l'horloge du validateur (la seule méthode valable) :
`NOUS = precommit_arrival_ts − t_proof_end` · `EUX = t_post − precommit_arrival_ts`.

| moment | **NOUS** | **EUX** | effet |
|---|---|---|---|
| régime normal | 0,26-0,44 s | **1,3 s** | — |
| **vague 1 — 14:35→14:50** | 0,63 s | **7,66 s** (p90 8,32) | `stale_round`/fen 1,6 → 3,8 |
| **vague 2 — 15:50→** | 0,59 s | **25,27 s** | fen 32630 : **23 `stale_round`, 1 acceptée sur 17** |

Sur la fenêtre 32630, le délai preuve→POST fait **15,6 s de médiane, jusqu'à
62,6 s**. À ce régime chaque envoi franchit plusieurs frontières de round drand
de 3 s : l'entrée est morte d'avance.

🔑 **Notre part est restée saine (0,26-0,63 s) tout du long.** Ce n'est pas nous.

🔑 **`/health` répond en 0,58 s, `status ok`, protocole 5** pendant les deux
vagues → leur serveur est debout, c'est le **chemin de soumission** qui est
engorgé. Ne pas conclure « le validateur est down » sur la base de `/health`.

⚠️ **Mécanisme de cascade** : notre client attend la réponse HTTP avant de
libérer le créneau d'envoi. Un POST à 25 s retarde donc AUSSI les entrées
suivantes — d'où les fenêtres où toutes les entrées arrivent au même offset
(ex. 32629 : quatre entrées à 34,3 s, rangs 43-58).

---

## 4bis. 🔴 PANNE DU VALIDATEUR — 28 MIN, ~9 FENÊTRES PERDUES (15:52 → 16:20)

**La vague 2 n'était pas une vague : c'était le début d'une panne.**

| jalon | heure | fait |
|---|---|---|
| dernière soumission acceptée | **15:52:14** | — |
| `/state` figé | 15:52 → 16:20 | `window_n` **bloqué à 32630**, même `randomness` (`612337cd…`) pendant 28 min |
| `/health` | ~16:06 | passe de `ok` à **`degraded`** ; le watchdog voit `validateur HTTP 000` |
| watchdog | 16:10:28 | `WEDGE (dernier groupe 950s)` → **restart inutile** (voir plus bas) |
| reprise | **~16:20** | `window_n` 32631, `status ok`, reprise spontanée du mineur |

### Les 3 motifs de rejet ne sont qu'UNE cause
Sur la seule fenêtre 32630, en cascade au fil de l'aggravation :
`stale_round` (29) → `precommit_invalid` (5) → **`precommit_expired` (26)**.
Le dernier apparaît quand le precommit dépasse les 33 s de grâce. **Trois
symptômes, un seul mal : le validateur ne répond plus dans les temps.**

### ⛔ Ce qu'il ne faut PAS faire
- **Ne pas redémarrer le mineur.** Le watchdog l'a fait à 16:10 pour rien : le
  moteur n'était pas coincé, il n'avait plus de fenêtre à travailler. Le
  détecteur de wedge ne distingue pas « mineur bloqué » de « validateur muet ».
  🔭 **Piste** : conditionner le wedge à un `/state` qui AVANCE.
- **Ne pas juger un réglage sur cette période.** Toutes les métriques de
  15:50 → 16:20 sont contaminées.

### ✅ Ce que la panne prouve sur nous
Notre part de latence (`NOUS`) est restée à **0,26-0,63 s du début à la fin**,
y compris au pire de l'engorgement. Le mineur a repris **seul**, sans
intervention : bake de 8 groupes en 18,2 s, groupe 1/8 prêt à **1,8 s**.
Bonus : le changement de checkpoint (`ee39b6ba…` → `4c9f36ad…`) a été absorbé
**pendant** la panne, donc **aucun 🥶 de rechargement à la reprise**.

---

## 4bis-2. ⚡ SECONDE PANNE — 502 FRANC, ~2 MIN (20:28)

Profil **différent** de celle de 15:52 : pas de dégradation progressive, une
coupure nette.

| fait | valeur |
|---|---|
| `precommit HTTP 502` | **43 sur la seule minute 20:28**, 2 à 20:29, puis zéro |
| `/health` | renvoie lui-même **502**, pas seulement lent |
| durée | **~2 min** (retour HTTP 200 à 20:30) |
| coût | 1 fenêtre au plus |

⚡ **Effet de bord favorable, à connaître** : la panne a bloqué TOUT le marché.
Au retour, les 64 places étaient libres et notre arriéré est passé en entier —
`fen 32768` a pris **24 acceptées avec des arrivées jusqu'à 88 s**, du jamais vu.
⛔ **Sans valeur** : seules les 3 entrées sous 12 s ont payé (rangs 2, 3, 15) ;
les 21 autres (28-88 s) finissent aux rangs 34-64. Le guichet ouvert ne change
rien au rang.
🪤 C'est cette traîne de 88 s qui a produit le faux positif du détecteur (§9).

**Bilan validateur du jour** : 2 vagues de lenteur (14:35, 15:50) + 1 panne de
28 min (15:52) + 1 coupure de 2 min (20:28). **Leur infrastructure est instable
aujourd'hui**, et c'est la première source de perte (§4quater).

---

## 4ter. 💰 RÉPARTITION DES SLOTS PERDUS PAR CAUSE

Comptage sur **889 fenêtres** (31755→32644), **6 455 lignes → 5 116 entrées
logiques**.

### ⚠️ Deux pièges qui faussent tout comptage naïf
1. **26,2 % des entrées sont RE-TIRÉES.** Un `stale_round` n'est pas une perte :
   `_fire_for_window` remet l'entrée en file. **67 % des `stale_round` finissent
   acceptés** (923 rattrapés sur 1 378). Compter les rejets bruts surestime la
   perte d'un facteur ~6 sur ce motif.
   ⛔ Et dédupliquer par `(prompt_idx, t_pick)` en gardant la **première** ligne
   fait l'erreur inverse : on garde le rejet et on jette l'acceptation. **Il faut
   l'issue FINALE de chaque entrée.**
2. **Tous les slots ne valent pas pareil.** Taux de paiement observé par bande
   d'arrivée — un slot perdu à 30 s ne coûte quasiment rien :

   | arrivée | <10 s | <15 s | <20 s | <25 s | <30 s | ≥30 s |
   |---|---|---|---|---|---|---|
   | payées | **71,1 %** | 27,7 % | 11,6 % | 7,8 % | 10,7 % | 5,4 % |

### Issue finale des 5 116 entrées
| issue | n | part |
|---|---|---|
| **acceptée** | 3 688 | **72,1 %** |
| `batch_filled` | 1 030 | 20,1 % |
| `stale_round` (définitif) | 239 | 4,7 % |
| `wrong_checkpoint` | 49 | 1,0 % |
| `window_mismatch` | 46 | 0,9 % |
| `precommit_expired` | 37 | 0,7 % |
| `window_not_active` | 17 | 0,3 % |
| `precommit_invalid` | 10 | 0,2 % |

### 💰 Le même bilan converti en PAYÉES (pondéré par la bande d'arrivée)
| cause | entrées perdues | payées perdues | **part du revenu potentiel** |
|---|---|---|---|
| **encaissé** | — | — | **75,8 %** |
| **guichet plein** (`batch_filled`) | 1 031 | ~101 | **11,0 %** |
| **fenêtres mortes** (checkpoint) | 102 fen | ~84 | **9,1 %** |
| `stale_round` définitif | 239 | ~20 | 2,2 % |
| `wrong_checkpoint` + `window_mismatch` | 95 | ~13 | 1,4 % |
| `precommit_expired`/`invalid`/`not_active` | 64 | ~4 | 0,5 % |

**Revenu potentiel ≈ 922 payées, encaissé 699.**

### 🎯 Ce que ce tableau change
- **`batch_filled` est la première fuite (11,0 %)** — la course aux 64 places
  du marché, pas un défaut de chez nous. Confirme que la métrique utile est
  « groupes avant la fermeture du guichet ».
- **Le checkpoint est second (9,1 %)** et c'est le seul poste où *nous* avons
  une prise technique (§1).
- ⛔ **`stale_round` ne pèse que 2,2 %** malgré ses 1 378 occurrences brutes —
  **confirmation indépendante de la consigne « ne pas l'attaquer pour
  lui-même »** déjà inscrite au CLAUDE.md. Les rejets sont rattrapés, et ceux
  qui ne le sont pas concernent des entrées tardives sans valeur.
- Tous les motifs « inquiétants » de la session (`precommit_expired`,
  `precommit_invalid`, `window_not_active`) pèsent **0,5 % à eux trois**.

⚠️ **Limites** : les 84 payées des fenêtres mortes sont une **estimation**
(102 fenêtres × 0,82 payée/fenêtre vivante) — on ne sait pas ce qu'elles auraient
produit. Et les taux par bande sont transversaux, à marché figé : ils supposent
qu'un slot sauvé serait arrivé dans la même bande, ce qui n'est pas garanti.

---

## 4quater. ⚖️ EUX OU NOUS — LE PARTAGE DU REVENU PERDU

Même base que le §4ter (889 fenêtres, 5 116 entrées), mais imputée par côté.

| poste | payées | part | côté |
|---|---|---|---|
| **encaissé** | 756 | **76,0 %** | — |
| guichet plein + lenteur + panne | 111 | **11,2 %** | **EUX** |
| rechargement de checkpoint | 97 | **9,8 %** | **NOUS** |
| `stale_round` définitif | 22 | 2,3 % | partagé |
| **échecs de vérification** | **9** | **0,9 %** | **NOUS** |

### Ce qui est vraiment à nous
1. **Le rechargement (9,8 %)** — mais l'OBLIGATION de recharger est imposée par
   le protocole. Ce qui nous appartient, c'est la **DURÉE** : 121 s en cache,
   365 s avec téléchargement, parce que vLLM se réinitialise ENTIÈREMENT (§1).
   **C'est le seul poste avec une prise technique réelle.**
2. **Les échecs de vérification (0,9 %)** — 28 groupes / 3 883 verdicts, seuls
   vrais défauts de conformité du code : `logprob_mismatch` 14 ·
   `reward_mismatch` 8 · `bad_termination` 3 · `seed_mismatch` 2 ·
   `token_tampered` 1. Ni la longueur ni la position ne les expliquent (§1 du
   CLAUDE.md, TODO n°3). Ils frappent les BONS rangs (jusqu'au rang 1).

### ⛔ Ce qui RESSEMBLE à une erreur de notre côté et n'en est pas
| apparence | mesure |
|---|---|
| 12 redémarrages VRAM/jour | coût **indétectable** (8,2 % vs 12,7 % de référence) |
| `checkpoint pull failed` horaire | bénin — notre snapshot était déjà à jour |
| 1 378 `stale_round` | **67 % rattrapés** par re-tir → 22 payées, pas 196 |
| notre latence d'envoi | **0,26-0,63 s** toute la session, jamais dégradée |
| `precommit_expired`/`invalid` | symptômes de LEUR lenteur, pas de notre code |

---

## 4quinquies. 📉 BAISSE DU RANG À ARRIVÉE CONTRÔLÉE (à suivre)

**Le seul signal statistiquement soutenu de la session.** Mesuré vers 19h50 UTC.

Part de nos entrées arrivées **sous 12 s** qui atteignent le rang ≤20 :

| bras | fenêtres | entrées | rang méd | rang ≤20 |
|---|---|---|---|---|
| AVANT (T-6h..-2h) | 58 | 114 | 13 | **75,1 %** |
| APRÈS (T-2h..0h) | 45 | 91 | 20 | **56,3 %** |

**−18,8 points, p = 0,019** (test de permutation, **unité = la fenêtre**).
⚡ **Le test par ENTRÉE donnait des IC recouvrants et concluait à tort au bruit.**
Les entrées d'une même fenêtre sont corrélées : il faut agréger par fenêtre.

### Ce que ce n'est PAS (vérifié)
| hypothèse | mesure |
|---|---|
| notre latence | `NOUS` 0,33-0,42 s, plate |
| notre volume | **en hausse** : `max_len` méd 556 → 596 |
| peloton plus rapide | lag marché p50 **8,45 → 8,17 s** (négligeable) |
| peloton plus nombreux | 156 → 150 rollouts, **256 mineurs** des deux côtés |

### 🔭 MÉCANISME CANDIDAT — nos complétions RACCOURCISSENT aux ckpt 656-657
⛔ Ma première piste (« le modèle allonge, le seuil monte ») était **à l'envers**.
Mesure par checkpoint :

| checkpoint | n groupes | `max_len` médian |
|---|---|---|
| 650-655 | 68-95 chacun | **655-695** |
| **656** | 99 | **593** |
| **657** | 26 | **568** |

**−15 % en deux crans**, alors que la dérive de fond mesurée sur 53 checkpoints
(572→657) n'est que de **−3,0 tokens/checkpoint**. Coïncide dans le temps avec
la baisse du rang :

| tranche | rang ≤20 | `max_len` médian |
|---|---|---|
| 13h | 31,4 % | 706 |
| 17h | 25,8 % | 677 |
| 18h | 24,7 % | 644 |
| 19h | 21,1 % | 604 |

### 🔑 TEST DU RANG À BUCKET CONSTANT — le marché n'a PAS raccourci
Le dashboard n'expose aucun volume de tokens des autres mineurs. Détour : notre
**rang à bucket constant**. Si tout le monde raccourcissait ensemble, le rang
étant positionnel, ce rang resterait stable.

| tranche | n | bucket-proxy méd | rang méd | **rang à bucket > 40k** |
|---|---|---|---|---|
| T-6h..-4h | 213 | 63 820 | 30 | **26,0** (n=182) |
| T-4h..-2h | 305 | 64 569 | 28 | **26** (n=251) |
| **T-2h..0h** | 332 | **57 240** | **34** | **30,0** (n=252) |

**Deux dégradations INDÉPENDANTES et simultanées :**
1. notre volume **−11 %** (conséquence du raccourcissement ckpt 656-657) ;
2. **à bucket comparable, on perd 4 places** (26 → 30) — donc le peloton a
   allongé ou s'est densifié en haut pendant qu'on raccourcissait.

⚠️ **Réserves** : `payload_bytes / round` est un **PROXY** du bucket (le
validateur utilise `Σ completion_lens`, absent de nos données), et « bucket
> 40k » écarte le bas de distribution sans fixer le bucket. **Indication forte,
pas démonstration.**
✅ **Les payées par fenêtre mûre NE suivent PAS** : 1,21 (17h) · 0,80 (18h) ·
**1,05 (19h)** — dans la bande normale du jour. C'est ce qui empêche de parler
de perte de revenu à ce stade.

⚠️ **Ce n'est pas un A/B, ce sont deux PÉRIODES.** Le bras AVANT contient la
panne validateur (§4bis) et les deux vagues de lenteur (§4), le bras APRÈS non.
Tout ce qui a changé entre-temps est confondu avec l'effet. Le résultat dit
« ça a baissé », **pas** « voici pourquoi ».
### ✅ REFAIT 2 H PLUS TARD — ce n'est PAS une chute continue, c'est un PALIER
⛔ **Ma formulation initiale était fausse dans sa dynamique.** Le test refait à
l'identique donne : AVANT (4 h, 64 fen) 65,1 % vs APRÈS (2 h, 40 fen) 57,9 %,
écart +7,2 pts, **p = 0,39 — NON significatif**. L'écart de 18,8 pts s'est
dégonflé, comme les quatre autres lectures précoces du jour.

Série complète par tranches de 2 h (unité = fenêtre) :

| tranche | fen | rang ≤20 |
|---|---|---|
| T-12h..-10h | 39 | 82,3 % |
| T-10h..-8h | 35 | 65,0 % |
| T-8h..-6h | 34 | 77,2 % |
| T-6h..-4h | 25 | 73,7 % |
| **T-4h..-2h** | 39 | **59,7 %** |
| **T-2h..0h** | 40 | **57,9 %** |

Testé comme un **PALIER** et non comme une tendance :
**71,7 % → 58,8 %, écart 13 pts, p = 0,039 — SIGNIFICATIF.**

### ⛔⛔ PISTE ABANDONNÉE — le palier ne survit pas à la résolution horaire
Série complète, entrées sous 12 s, par HEURE :

| h | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|
| rang ≤20 | 55,6 | 76,9 | 59,3 | 79,3 | 65,2 | 70,0 | 64,7 | 54,8 | **44,4** | 58,6 |
| rang méd | 19 | 13 | 18 | 11 | 15 | 12 | 12 | 17 | 25 | 15 |
| `max_len` | 679 | 626 | 610 | 536 | 578 | 551 | 588 | **642** | 615 | 584 |
| arrivée | 8,9 | 8,1 | 9,0 | 8,0 | 7,5 | 6,9 | 7,5 | 9,1 | 8,5 | 9,5 |

**Le niveau actuel (58,6 %) est celui de 11h (55,6 %) et de 13h (59,3 %).**
La série oscille entre 44 et 79 % toute la journée, sans direction. Le palier
« 71,7 → 58,8 %, p = 0,039 » dépend **entièrement de l'endroit où on coupe** :
décaler la frontière d'une heure le fait disparaître.
⛔ **Aucune variable ne suit** : `max_len` est PLUS HAUT à 18h (642) qu'à 16h
(551) alors que le rang y est pire ; l'arrivée ne bouge que d'1 s (~1,5 place
attendue) pour un rang médian passant de 12 à 25.

🪤🪤 **LA VRAIE LEÇON — trois formulations sur la même donnée en 2 h** :
« tendance » → « palier » → « rien ». Découper une variable bruitée de
plusieurs façons jusqu'à trouver un p intéressant **fabrique** de la
significativité. Le bon réflexe : fixer le découpage AVANT de regarder, et
exiger qu'un mécanisme mesurable accompagne l'écart. Ici il n'y en avait aucun.
✅ **Ce qui reste vrai et mesuré** : les complétions raccourcissent bien de 15 %
aux ckpt 656-657 (fait), et les payées par fenêtre mûre n'ont pas bougé (fait).
Le lien entre les deux n'est PAS établi.

---

## 5. 📡 `ReadTimeout` — LA FORME EXTRÊME DE LA MÊME CAUSE

`fire task exception … ReadTimeout('')` = notre POST expire, l'entrée est perdue.

| tranche | 10h | 11h | 12h | 13h | 14h | 15h |
|---|---|---|---|---|---|---|
| n | 2 | 6 | 5 | 8 | **9** | 7 |

~5 % des entrées envoyées. **Pas une dégradation de fond** — la pointe de 14h
correspond exactement à la vague 1. Ne pas s'en alarmer sans voir une tranche
horaire complète dépasser nettement 9.

---

## 6. 🔍 `precommit_invalid` — RARE, MAIS EN RAFALE SOUS LA LENTEUR

**5 occurrences sur tout l'historique** (fenêtres 31918, 32386, 32477, 32618),
soit une entrée perdue toutes les ~200 fenêtres — donc **pas un bug**.
⚠️ **MAIS** la fenêtre 32630 en a produit **5 à elle seule**, pendant la vague 2.
C'est le même engorgement vu d'un autre angle : un precommit traité 15-60 s
après son envoi n'est plus valide. **Symptôme, pas cause.**

---

## 6bis. 🩸 ÉCHECS DE VÉRIFICATION — 0,8 %, MAIS SUR LES MEILLEURS RANGS

**31 échecs sur 3 883 verdicts (0,8 %)** — les seuls vrais défauts de
conformité de notre code.

| motif | n | rangs touchés |
|---|---|---|
| `logprob_mismatch` | 14 | **6 à 16 — tous en zone payante** |
| `reward_mismatch` | 8 | — |
| `bad_termination` | 3 | 10, 12, 50 |
| **`seed_mismatch`** | **2** | **1 et 3** |
| `token_tampered` | 1→2 | 22, puis **10** (fen 32726) |

⚡ **Une payée quasi certaine perdue le 25/08** : `fen 32686`, rang **1**,
valeur d'enchère 0,477, arrivée 5,4 s → rejetée `seed_mismatch` au stade
`auction_seal`.

### ✅ LA LONGUEUR N'EST PAS LE FACTEUR (mesure demandée par le TODO n°3)
| | p10 | médiane | p90 |
|---|---|---|---|
| groupes **en échec** (n=28) | 492 | **780** | 1118 |
| groupes **acceptés** (n=3390) | 510 | **745** | 1055 |

Taux d'échec **plat** par tranche : 0,80 % (<600) · 0,80 % (600-900) · 0,97 %
(900-1500) · 0 % (>1500, n=71).
🪤 « Les groupes en échec sont courts » est un **ARTEFACT** : toute notre
production est courte. Toujours comparer à la population de référence.
⚠️ n=28 : exclut un effet FORT de la longueur, pas un effet léger.

### 📊 Pas de grappe
Échecs par centaine de fenêtres : 2 · 1 · 2 · 3 · 3 · 5 · 2 · 3 · 5 · 2 · 3.
Stable sur toute la plage — aucune dégradation dans le temps.

🔭 **Piste restante** : ni la longueur ni la position ne l'expliquent. Chercher
du côté du contenu du prompt ou de la contention GPU au moment de la preuve.

---

## 6ter. 💵 LE RANG NE SUFFIT PAS — LA VALEUR D'ENCHÈRE DÉCIDE AUSSI

Le seuil de paiement est **stable toute la journée** — p90 des rangs payés :

| tranche | 12h | 13h | 14h | 15h | 16h | 17h |
|---|---|---|---|---|---|---|
| p90 rangs payés | 17 | 16 | 17 | 15 | 17 | 17 |

⛔ **Mais un bon rang ne garantit pas le paiement** : chaque heure compte **5 à
14 entrées non payées alors qu'elles sont au rang ≤20**. Observé en direct :
rang 6 non payé pendant que le rang 8 de la même fenêtre l'était (`fen 32747`),
rang 17 non payé (`fen 32696`), rang 21 **payé** (`fen 32623`).

**La valeur d'enchère du groupe entre dans la sélection**, pas seulement le
débit. ⚠️ Ne jamais conclure « on a perdu une payée » d'un rang ≤20 non payé,
ni « le seuil se durcit » d'une série de rangs 17-21 non payés.

---

## 7. ❓ VARIANCE DE LA PREMIÈRE ARRIVÉE — NON EXPLIQUÉE

Notre première arrivée oscille fortement d'une fenêtre à l'autre :

| fen | 32612 | 32613 | 32614 | 32615 | 32616 | 32617 | 32619 | 32620 | 32621 | 32623 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1re arrivée | 5,2 s | 9,1 | **4,7** | **20,8** | 4,9 | **18,2** | 6,0 | **22,8** | 14,6 | **31,9** |

- ⛔ **Ce n'est pas notre chemin d'envoi** : `NOUS` reste à 0,25-0,45 s sur les
  bonnes comme sur les mauvaises fenêtres.
- ⛔ **Ce n'est pas le vidage du pool au flip** — testé et **réfuté** : 32614 a
  vidé 31 entrées pour une 1re arrivée à 4,7 s, 32615 n'en a vidé que 10 pour
  20,8 s. Aucune corrélation.
- ✅ La génération est saine : groupe 1/8 prêt à **3,2-4,7 s**, groupe 8/8 à
  10,3-14,4 s, GPU 71-99 %.

**Le retard naît donc entre le flip et le premier groupe utilisable.** C'est le
chantier n°3 du CLAUDE.md, et le plus rentable identifié au projet
(**1 s = 1,46 place**). Mérite une étude dédiée, pas des sondages au fil de l'eau.

---

## 8. 🎲 LE GUICHET DES 64 PLACES EST TRÈS VARIABLE

| fenêtre | comportement |
|---|---|
| **32615** | guichet ouvert **jusqu'à 48 s** — 14 acceptées, 1 seul `batch_filled` |
| **32618** | guichet fermé tôt — **16 `batch_filled`** |

Conséquence directe sur le rang, qui est **positionnel** : une arrivée à 21,8 s
a décroché le **rang 11 payé** en 32615 (marché lent), tandis qu'une arrivée à
9,7 s n'a valu que le **rang 46** en 32601 (marché rapide).
⛔ **Ne jamais juger un réglage sur une poignée de fenêtres** — le rythme du
peloton domine le signal à cette échelle.

---

## 9. 🛠️ DÉFAUTS DE L'OUTILLAGE DE SURVEILLANCE — CORRIGÉS CE JOUR

| défaut | effet | correctif |
|---|---|---|
| `live_monitor_tick.py` ne comptait que les lignes **nouvelles** du tick | une même fenêtre rapportée 2× avec des décomptes partiels (« 7 rejets » puis « 6+14 ») | ré-agrégation de **toutes** les lignes de la fenêtre |
| alerte d'erreur : `grep -cE "ERROR\|Traceback"` | une exception comptait 2-3 « erreurs », et le message affiché était `Traceback (most recent call last):` — inexploitable | compte les seules lignes `\| ERROR \|`, affiche le message réel + horodatage |
| aucune détection des fenêtres perdues | 5,7 % des fenêtres invisibles | **détecteur 🥶 à 3 formes** (tout refusé / aucune tentative / aucune trace) + attribution de cause par `checkpoint_n` |
| `live_monitor.sh` en dur sur une box morte | — | `BOX`/`PORT`/`STATE` en variables |

### ⚠️ VERDICTS NON DÉDUPLIQUÉS DANS LE MONITEUR — corrigé
`verdicts_v4.jsonl` **réécrit la même ligne à chaque poll** : **13 551 lignes
pour 4 513 merkle uniques**, jusqu'à **6 réécritures** pour une seule entrée.
Le tick du moniteur ne dédupliquait pas → une entrée pouvait être affichée et
comptée deux fois quand deux réécritures tombaient dans le même tick de 60 s.
**Impact vérifié sur 10 fenêtres annoncées : 8 justes, 2 gonflées d'une unité**
(`fen 32700` 2 réelles au lieu de 3, `fen 32819` 1 au lieu de 2).
✅ **Les tableaux chiffrés de ce document ne sont PAS touchés** — ils viennent
de scripts qui dédupliquent par `merkle_root`. Seul l'affichage temps réel
l'était.
🔧 Correctif : `_v[merkle_root] = ligne` avant usage, on garde la plus récente.

### ⚠️ FAUX POSITIF DU DÉTECTEUR 🥶 — corrigé
`fen 32768` marquée « aucune trace du tout » a finalement produit **24
acceptées**, avec des arrivées jusqu'à **70 s**. Le mineur tirait encore sur
elle alors que la fenêtre suivante avait démarré.
**Audit des 22 marquages de la session : 21 corrects, 1 faux positif.**
🔧 Correctif : une fenêtre n'est déclarée perdue qu'avec **DEUX fenêtres de
recul** (`dernier - 1`), le temps que les écritures tardives arrivent.
🪤 **Leçon générale** : dans ces JSONL, « la fenêtre N+1 existe » ne veut PAS
dire « la fenêtre N est finie ». La traîne d'envoi peut dépasser une fenêtre
entière. Tout comptage par fenêtre doit prévoir cette marge.
⚠️ **RÉCIDIVE (fen 32828)** : le correctif n'avait été appliqué qu'à la
détection des fenêtres INVISIBLES, pas aux deux autres formes. `32828`,
marquée « guichet déjà fermé », a finalement placé **7 acceptées à 32-61 s**.
Le recul de 2 fenêtres est désormais exigé par **les trois formes** du 🥶.
👉 Le problème s'aggrave avec le régime long du §0bis : les arrivées vont
maintenant jusqu'à **99 s**, soit bien au-delà d'une fenêtre.

Sauvegardes : `live_monitor.sh.bak-2026-08-25`,
`live_monitor_tick.py.bak-2026-08-25`, `CLAUDE.md.bak-2026-08-25-1430`.

---

## 🪤 PIÈGES DE MESURE — CHACUN A PRODUIT UNE CONCLUSION FAUSSE AUJOURD'HUI

0. **Compter les rejets bruts comme des pertes.** `stale_round` affiche 1 378
   occurrences… dont **923 finissent acceptées après re-tir**. Sa perte réelle
   est de 239 entrées, soit **2,2 % du revenu** au lieu des ~21 % qu'un
   comptage brut suggère. **Raisonner sur l'ISSUE FINALE de chaque entrée**
   (§4ter), jamais sur la ligne de rejet.
0bis. **Comparer `/state` à une LIGNE DE LOG.** J'ai cru à un décalage de
   checkpoint (`fb78f170` attendu contre `9ca92010` chargé) — la ligne de log
   datait de 30 min plus tôt et un rechargement était en cours. **Le bon test
   est `ls` des snapshots sur disque**, qui donne l'état réel et non
   l'historique. Fausse alerte évitée de justesse.
0ter. **⚠️⚠️ Grouper par heure sans la DATE — COMMIS TROIS FOIS.**
   `strftime("%Hh")` mélange hier et aujourd'hui. Trois tableaux horaires
   publiés agrégeaient deux journées, dont l'analyse de la queue de
   distribution du §0bis (la tranche « 21h » mélangeait 24/08 et 25/08).
   **Toujours `"%m-%d %Hh"`.** L'avoir écrit une fois n'a pas suffi.
1. **Médianes par fenêtre sans dédupliquer les re-tirs.** J'ai annoncé « notre
   latence bondit à 3,3-4,6 s » sur des médianes de 2 à 5 entrées non
   dédupliquées. Sur données dédupliquées : **plate à 0,27-0,37 s**, p90 même
   *amélioré*. **Toujours dédupliquer par `(prompt_idx, t_pick)`.**
2. **Accuser le coupable qui traîne dans le voisinage.** J'ai imputé une perte
   de fenêtres au redémarrage de 13:43:51 parce qu'il tombait au bon moment.
   La vraie cause était un rechargement de checkpoint **30 min plus tard**.
   **Dater l'incident AVANT de désigner un coupable.**
3. **Compter sur des fichiers qui ne voient pas tout** (§2) → « 1,2 % » au lieu
   de 10,6 %.
4. **Un chiffre sur 6 fenêtres mûres ne vaut rien.** La tranche 14h donnait
   0,67 payée/fenêtre mûre — dans le bruit d'une journée à 0,76-1,38. Le seuil
   du projet reste **30 fenêtres mûres minimum**.

---

## ✅ CE QUI EST SAIN — vérifié, ne pas re-chercher

- **Notre chemin d'envoi** : `NOUS` à 0,26-0,63 s sur toute la session, y
  compris pendant les deux vagues de lenteur.
- **La génération** : groupe 1/8 à 3,2-4,7 s, groupe 8/8 à 10,3-14,4 s.
- **Le checkpoint** : notre snapshot local `ee39b6ba3da61808efa7ed3b6fccb9b6864a7a9c`
  == la révision attendue par `/state`. Le `checkpoint pull failed`
  (`httpx.ConnectError`) qui apparaît **toutes les heures pile** (12:12, 13:11,
  14:12) est **sans conséquence** : c'est un rafraîchissement périodique, et
  « keeping local » est le bon comportement puisque le local est à jour.
- **Le watchdog** : il redémarre souvent, mais sans coût mesurable (§3).

---

## 🔧 PISTES À INSTRUIRE — établies le 25/08 après-midi

### 1. ⚡ LE RECHARGEMENT COÛTE 18 s DE RECOMPILATION INUTILE (cause trouvée)

⛔ **La note « rechargement = 56 s de blocage » est FAUSSE.** Le blocage de la
boucle mineur vaut **0,2 s** (trou d'activité mesuré sur les 9 avancées de
10:04→13:14). Le coût est ailleurs, et il est entièrement dans vLLM :

| étape | durée |
|---|---|
| chargement des poids | **1,19 s** |
| modèle en VRAM | 2,08 s |
| transformation Dynamo | 7,00 s |
| **compilation du graphe** | **11,65 s** |
| **`torch.compile` total** | **22,1 s** |
| *le même cache réutilisé* | ***4,4 s*** |

**Cause** : `model` et `revision` ne sont PAS dans les `ignored_factors` de
`ModelConfig`. Le chemin du checkpoint change à chaque publication → le
`config_hash` change → nouveau répertoire → recompilation intégrale.
**43 répertoires, 2,0 Go de cache pour UN SEUL graphe.**

**Preuve que le graphe est indépendant du checkpoint** :
- `cache_key_factors.json` de deux caches : **211 facteurs, UN SEUL diffère**
  (`config_hash`) — `env_hash`, `code_hash`, `compiler_hash` identiques.
- `computation_graph.py` : **IDENTIQUE octet pour octet**.
- Les artefacts binaires diffèrent de **240 octets sur 7 Mo** (taille d'un
  chemin de fichier embarqué). ⚠️ Leur équivalence fonctionnelle n'est PAS
  prouvée — seule celle du graphe source l'est.

**Correctif** (`compilation/backends.py:1053` : `if not cache_dir:` → le hash
n'est calculé QUE s'il est absent) — 3 lignes dans `vllm_backend.py` avant le
`LLM(**kwargs)` de la ligne 235 :
```python
cache = os.environ.get("RELIQUARY_VLLM_COMPILE_CACHE_DIR")
if cache:
    kwargs["compilation_config"] = {"cache_dir": cache}
```
⛔ **Aucune variable vLLM ne suffit** — vérifié sur `envs.py`. `VLLM_CACHE_ROOT`
ne déplace que la racine, le sous-répertoire haché reste calculé.
⚠️ **Mettre la version de vLLM dans le chemin** (`/workspace/vllm_compile_v0.24.0`) :
en figeant le répertoire on perd l'invalidation sur mise à jour du moteur.
✅ **Filet** : un graphe faux donnerait du `SEED_MISMATCH` massif dès la
première fenêtre — bruyant, détecté en 4 min, repli = vider la variable.

### 2. 🏃 A/B `SPRINT_SIZE` 4 → 5 (variable pure, aucun code)

**Juger sur les GROUPES AVANT 25 s**, jamais sur les payées à 1 h : c'est une
mesure de débit, indépendante du peloton, lisible sur 10 fenêtres.
Repères : sprint=4 → **9,4 et 10,2** · sprint OFF (8) → **13,5**.

**Ce qui soutient l'élargissement** (mesuré 25/08, 396 fenêtres) — la qualité
de v5.8 ne s'effondre PAS en profondeur :

| slot | valeur v5.1 | **valeur v5.8** |
|---|---|---|
| 1 | 0,1517 | **0,2333** |
| 5 | 0,1565 | **0,2106** |
| 10 | 0,1532 | **0,1892** |

⚡ **v5.1 est PLAT** (0,1517 au slot 1 contre 0,1532 au slot 10) : son
classement ne porte aucune information. **Le 10ᵉ choix de v5.8 vaut mieux que
le MEILLEUR de v5.1.** Élargir le sprint n'aurait ajouté que du bruit sous
v5.1 ; sous v5.8 c'est cohérent.
Coût cumulé de l'élargissement : top-4 **+47,6 %** de valeur → top-5 +44,9 %
→ top-8 +38,0 % → top-10 +34,1 %.

⚠️ **Ne PAS superposer** ce test à un autre changement, et ne pas conclure sur
les payées avant **30 fenêtres mûres**.

### 3. ⏰ SEUIL DU WATCHDOG — 15 min, trop haut

Timeout chaîne finney le 25/08 à 12:37 → **11 min de blocage, 7 fenêtres à
zéro groupe** (32525-32531). Le watchdog ne l'a pas vu : son seuil de blocage
est à 15 min. Descendre à 8-10 min rattraperait ce cas, au prix de faux
redémarrages (chacun coûte une fenêtre).

### 4. ✅ v5.8 — CONCLU, gain RÉEL mais mécanisme RÉFUTÉ

80 fenêtres mûres : **1,09-1,16 payée/fenêtre mûre contre 0,90-0,94** sous
v5.1, fenêtres payantes 67,5 % contre 64 %. Soit **+18 à +21 %**.
⛔ **Mon mécanisme était faux** : je projetais +1169 tok/groupe sur le sprint
(t=+19,6 en apparié) donnant −5,1 places et +39 %. Mesuré en vol : **+187 tok,
IC95 [−238 ; +612]**. Rang médian et arrivée INCHANGÉS. Le seul effet qui tient
est l'in_zone du sprint, **65,6 → 74,2 %**.
🪤 **+41 % à 21 fenêtres mûres, +19 % à 40, ~+20 % à 80.** Encore une lecture
précoce dégonflée — le seuil de 30 fenêtres mûres n'est pas négociable.

---

## 🔄 5. ⛔ « LA CONTRAINTE A CHANGÉ DE CAMP » — CONCLUSION RÉFUTÉE LE JOUR MÊME

**Mesuré le 25/08 sur 175 fenêtres mûres** (depuis le restart 32439, v5.8).

| | fenêtres PAYÉES | NON payées |
|---|---|---|
| n | 109 | 66 |
| **tokens du meilleur groupe** | **6 422** | **5 262** |
| tokens du groupe le plus gros | **7 603** | 6 409 |
| arrivée de la meilleure | 8,5 s | 13,4 s |
| première arrivée | 8,1 s | 11,4 s |
| entrées acceptées | 5,3 | 4,1 |

### Le croisement départage les deux facteurs

| 1re arrivée ＼ tokens max | < 5 000 | 5 000-7 000 | **> 7 000** |
|---|---|---|---|
| **< 10 s** | 44 % (9) | 66 % (32) | **83 %** (53) |
| 10-15 s | **0 %** (4) | 24 % (17) | **82 %** (28) |
| > 15 s | — | 10 % (10) | **55 %** (22) |

⚡ **Le volume protège, la vitesse ne rattrape pas.** À >7 000 tokens on paie
82-83 % **même en arrivant à 10-15 s**, et encore 55 % au-delà de 15 s. À
l'inverse, arriver sous 10 s avec <5 000 tokens ne donne que 44 %, et la case
`10-15 s × <5 000` est à **0 % sur 4 fenêtres**.

### ⛔⛔ CETTE SECTION EST FAUSSE — CORRIGÉE PAR LA MESURE PAR ENTRÉE

Le croisement ci-dessus est au niveau **FENÊTRE**, en croisant « première
arrivée » et « tokens du plus gros groupe » — deux agrégats qui ne portent PAS
sur la même entrée — avec des cases à 4 et 9 observations.

**Refait au niveau ENTRÉE (n=1 422), le sens s'INVERSE :**

| tokens | arrivée < 8 s | 8-12 s | 12-18 s | > 18 s |
|---|---|---|---|---|
| 0-3 000 | **65 %** (20) | 11 % (18) | 0 % (14) | 3 % (31) |
| 3 000-4 000 | **54 %** (35) | 12 % (33) | 7 % (54) | 0 % (68) |
| 4 000-5 000 | **64 %** (44) | 29 % (42) | 7 % (70) | 1 % (115) |
| 5 000-6 000 | 88 % (16) | 57 % (23) | 15 % (67) | 4 % (124) |
| 6 000-7 000 | 88 % (26) | 66 % (32) | 25 % (83) | 9 % (182) |
| 7 000-9 000 | **100 %** (9) | 89 % (27) | 40 % (94) | **12 %** (172) |

⚡ **L'ARRIVÉE DOMINE, et de loin.**
- À volume fixé (4 000-5 000) : l'arrivée fait passer de **64 % à 1 %** — ×64.
- À arrivée fixée (<8 s) : le volume fait passer de **54 % à 100 %** — ×2.

⛔ **« Un groupe court ne paie pas » est FAUX** : <3 000 tokens sous 8 s paie
**65 %**, rang médian 14,5 — mieux que 7 000-9 000 tokens après 18 s (12 %).
✅ **La note du CLAUDE.md « l'arrivée est LE levier » reste donc VALIDE.**

🪤 **LEÇON** : ne jamais croiser deux agrégats de fenêtre (min de l'un, max de
l'autre) pour conclure sur une relation qui vit au niveau de l'entrée. Et
refuser toute case à moins de ~20 observations.

### ⛔ Ce que ça corrige dans le CLAUDE.md
La note « 1 seconde = +1,46 place = 390 tokens, l'arrivée est LE levier » a été
mesurée **AVANT** les correctifs de latence du 24/08. Depuis que l'arrivée est
passée de 14,8 s à 8,3 s, **ce goulot est desserré et le volume est devenu
contraignant**. La contrainte a été déplacée, pas supprimée.

### 🪤 Le contre-exemple qui interdit la conclusion simple
« Les fenêtres non payées sont celles où les rollouts sont courts » est vrai en
tendance mais FAUX comme règle :

| fenêtre | arrivée | tokens | résultat |
|---|---|---|---|
| 32746 | 6,6 s | **4 274** | ❌ rien |
| 32749 | 5,8 s | **4 257** | ✅ **rang 4, payé** |

Même volume à 17 tokens près, issue opposée. C'est le PRODUIT arrivée × volume
qui décide, via `bucket = tokens // (round × 50)`.

### ⛔ NOS CHAMPS NE RECONSTRUISENT PAS SON CLASSEMENT
Fenêtre 32749, champs bruts du validateur (`arrival_drand_round` confirme nos
offsets — aucun glissement) :

| offset | round rel | tokens | k | valeur LOCALE | bucket | **rang** |
|---|---|---|---|---|---|---|
| 5,8 s | 1 | 4 257 | 6 | **0,3026** | **85** | 4 |
| 7,2 s | 2 | **7 502** | **1** | 0,2269 | 75 | **1** |

⚠️ L'entrée rang 1 a un bucket **inférieur** ET une valeur locale **inférieure**
— elle devrait sortir derrière sur les deux critères. Explication la plus
probable : `opencodeinstruct` est `validator_authoritative_reward=True`, donc
**le validateur re-note lui-même** ; notre `k` et notre `score` ne sont pas les
siens. **Ne pas prétendre expliquer un rang avec nos champs.**
🪤 Piège commis ce jour-là : calculer le round par `⌊flip_offset/3⌋+1` suppose
que la fenêtre s'ouvre à notre offset 0. Le round réel du validateur montrait
un décalage — toute une colonne de buckets était fausse. **Utiliser
`arrival_drand_round`, jamais un round reconstruit.**
