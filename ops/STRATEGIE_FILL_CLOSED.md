# Stratégie fill-closed — où va la compétition, et où investir

**Établi le 2026-09-09** contre `origin/main @ 0be0cda` (PR #224,
`integration/reliquary-v1-final`). Compagnon de `ops/CUTOVER_V6.md` : celui-ci dit
**quoi optimiser et pourquoi**, l'autre dit **comment basculer le jour J**.

> ⚠️ Trois niveaux de confiance dans ce document, jamais mélangés :
> **[CODE]** lu dans la source mergée · **[MESURÉ]** chiffré sur notre box ou sur R2 ·
> **[MODÈLE]** arithmétique dérivée des deux, non observée. Ne jamais promouvoir un
> [MODÈLE] en fait sans mesure.

---

## 1. En un coup d'œil : l'inversion

| | v5 aujourd'hui | fill-closed |
|---|---|---|
| ce qui décide | **le rang** (bucket = tokens ÷ (rounds × 50)) | **l'ordre d'arrivée** dans une file FIFO |
| horizon qui compte | la **seconde** (frontière de round à 3 s) | la **minute** |
| contrainte qui mord | le **temps** | le **quota (32)** |
| notre handicap | bucket 81 posé sur la barre 81 | **n'existe plus** |
| ce qu'il faut optimiser | la latence de la chaîne | **le taux de groupes qui passent** |

**Le mineur doit passer d'une machine à débit à une machine à sélection.**

---

## 2. Ce n'est PAS armé — et ce n'est pas « l'économie par token »

**[CODE]** La capacité échoue fermée et exige **deux** conditions simultanées :

```python
FILL_CLOSED_ENABLED = (
    PROTOCOL_PROFILE_ID == "qwen3-4b-base-dapo-fill-closed-v6"
    and _FILL_CLOSED_REQUESTED        # env var, défaut "0"
)
```

**[CODE]** ⛔ **Correction d'une note longtemps portée par CLAUDE.md** : « paiement PAR
TOKEN, admission par TAUX » est **faux** pour ce qui a mergé. Les deux modules le
disent eux-mêmes :

- `token_rewards.py` — *« the disabled fill experiment […] the **Reliquary 1 target
  retains selected-slot rewards and does not activate this policy** »*
- `admission_priority.py` — *« **Reliquary 1 does not use this priority** for ticket
  admission or final ranking »*

⇒ Reliquary 1 garde les **sièges sélectionnés**. La branche par token / par débit est
conservée pour rejeu, éteinte.

**[CODE]** La suite d'environnements neuve (logic, verifiable, records, outils)
appartient au profil `qwen3-4b-reliquary-verifiable-v6-dev1`, décrit comme
*« isolated infrastructure/frontier profile »*, **hors lignée Math+Code**. Ce n'est pas
la bascule production.

**[CODE]** Le profil de bascule, lui, ne change **rien** à la génération :
*« Same model, same sampling, same budgets, same prompts as v5. v6 changes when a
window ends and who gets admitted, **never what a miner generates** — so the generation
contract is v5's, field for field. »* Mêmes deux environnements.

---

## 3. Le régime, chiffré

### Côté validateur **[CODE]**

| | valeur | d'où |
|---|---|---|
| durée max de fenêtre | **1 800 s** | `FILL_CLOSED_MAX_SECONDS` |
| deadline precommit | **1 767 s** | `MAX − UPLOAD_GRACE (33)` |
| groupes prouvés qui ferment un env | **256** | `16 × B_BATCH` |
| émissions portées par la fenêtre | **16** | `= CHECKPOINT_PUBLISH_INTERVAL_WINDOWS` |
| budget d'admission / env | **512** | `2 × 256` |
| budget de démarrage de grading / env | **1 024** | `2 × 512` |
| 1er pick | **30 s**, puis cadencé par le curseur du trainer, profondeur 2 |
| départage | **aucun** (`throughput_tiebreak=None`) |

**[CODE]** L'admission est un simple compteur, sans score :

```python
def may_admit(self, environment): return self._admitted[name] < self._budgets[name]
```

… et il est **monotone** : *« a group that fails its proof already spent real grading
cost […] does not refund the budget »*. **Un déchet brûle une place définitivement.**

**[CODE]** Le beacon de génération est `window_open_drand_round` : **randomness figée à
l'ouverture**, et `set_prompt_range()` est appelée *« once randomness is assigned »* —
donc **une seule tranche pour toute la fenêtre**, jusqu'à 30 min.

**[CODE]** `FILL_CLOSED_EMISSIONS_PER_WINDOW = CHECKPOINT_PUBLISH_INTERVAL_WINDOWS` :
une fenêtre = 16 pas d'optimiseur = **exactement un cycle de publication de
checkpoint**. ⇒ **l'avancée tombe à la frontière de fenêtre, plus jamais en plein vol.**
Cela supprime deux postes mesurés : la fenêtre perdue par rechargement (~6 % du revenu)
et la starvation `/state` pendant le téléchargement (3,4 %).

### Notre quota **[CODE]**

`MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW = 2 × B_BATCH = 32` pour tout protocole ≥ 4,
**donc aussi sous v6**. Aucune surcharge fill-closed (3 lecteurs, tous dans
`server.py`). Clé de quota = `(miner_hotkey, window_start)` ⇒ **32 au total pour la
fenêtre, les deux environnements confondus.**

### Notre capacité **[MESURÉ]**

| | valeur | échantillon |
|---|---|---|
| durée d'un bake (5 groupes) | **12,5 s** | n=1 782 bakes |
| débit de génération | **0,4 groupe/s** | dérivé |
| bridé aujourd'hui (`PREFLIP_GUARD=50`) | 20 groupes/fenêtre | n=461 fen |
| **débridé sur 1 800 s** | **~720 groupes** | [MODÈLE] |
| coût de preuve locale | 1,06 s/groupe | n=469 |

⇒ **22× plus de capacité que de cartouches.** Le GPU n'est PAS la contrainte.
**Augmenter le batch ne sert à rien** : on ne peut pas soumettre plus de 32.

### Le temps disponible pour choisir **[MESURÉ → MODÈLE]**

Cadence de preuve du validateur, env code, n=110 fenêtres : **16 groupes prouvés en
66 s de temps mur = 0,24 groupe/s**. Pour les 256 qui ferment un env : **~17,7 min**.

⇒ La fenêtre tournerait ~18 min sur un plafond de 30, **bornée par le débit de preuve
du validateur**, pas par la vitesse des mineurs. Il y a du temps pour être sélectif.
⚠️ [MODÈLE] : extrapole une cadence v5 vers des budgets fill-closed non tracés.

---

## 4. Le trou de conception à combler

**[CODE]** Aujourd'hui il n'y a **aucune sélection à l'envoi** :
`_maybe_fire_on_append` tire dès qu'un groupe entre au pool, `_fire_for_window` draine
**dans l'ordre**, filtré seulement par cooldown / tranche / budget. **La seule sélection
est l'ordre de bake**, donné par le prior. C'est optimal aujourd'hui — arriver tôt est
tout.

Sous fill-closed, tirer les 32 **premiers prêts** au lieu des 32 **meilleurs sur ~720**
gâche le levier principal. Il faut un **étage de sélection avant le tir**, qui
aujourd'hui serait un no-op structurel.

**Conséquence non évidente — inverser l'ordre du pipeline.** On prouve actuellement
*avant* de décider (SPEC_PROOF sur les slots de tête). À 1,06 s/groupe, prouver 720
groupes coûterait **763 s sur 1 800** pour en jeter 96 %. Sous fill-closed il faut
**sélectionner sur des signaux bon marché** (zone, grading, mémo, prior) et ne prouver
que les ~32 retenus.

**Le compromis à instrumenter** : retenir pour choisir = arriver plus tard dans la file
FIFO. Avec ~18 min de remplissage il y a du mou, mais **c'est une mesure à faire, pas
une hypothèse à tenir**.

---

## 5. Ce qui meurt, ce qui prend de la valeur

### ⛔ À retirer / neutraliser à la bascule (le port en neutralise déjà une partie)

`PREFLIP_GUARD_S` · `LATE_BAKE_FROM` · `FIRE_CURFEW_S` · le réglage de
`MAX_INFLIGHT_FIRES` · `DRAND_MIN_HEADROOM_S` · **et tout le chantier « gagner des
secondes sur la chaîne »**, y compris le fix de preuve du 08/09 (48→3 transferts,
−0,060 s : réel mais sans objet quand le round disparaît).

### ✅ Ce qui devient le cœur

| levier | pourquoi il change de statut | état |
|---|---|---|
| **qualité du filtre de zone local** | il jette 51 % des groupes ; sous fill-closed un rejet à tort coûte une cartouche, un maintien à tort **brûle une place d'admission sans remboursement** | à re-calibrer |
| **tri du mémo** | on prendrait **top 32 sur ~155** au lieu de top 2 → le critère devient déterminant. `best_in_range()` trie par **fraîcheur seule** | à ré-écrire |
| **alimentation du mémo** | moins de soumissions/unité de temps ⇒ moins de verdicts ⇒ apprentissage plus lent. Solde déjà **≈ −450/nuit** | passe de « à surveiller » à dépendance critique |
| **prior** | seul instrument qui ordonne le vivier avant génération | à ré-entraîner sur l'ère courante |
| **rationnement des 32 tirs** | ressource rare : ne pas les cramer dans les 2 premières minutes | **n'existe pas** |

---

## 6. Le chantier, par ordre de valeur

| # | chantier | pourquoi | état | comment juger |
|---|---|---|---|---|
| 1 | **Étage de sélection avant le tir** (choisir 32 sur ~720) | le levier n°1 du régime ; structurellement absent | à concevoir | banc hors ligne sur corpus réel : taux de passage des 32 retenus contre les 32 premiers prêts |
| 2 | **Preuve APRÈS sélection** | 763 s de GPU pour jeter 96 % du travail | à concevoir | temps GPU de preuve par fenêtre |
| 3 | **Rationnement du quota** | 32 cartouches pour ~18 min | à concevoir | distribution des tirs dans la fenêtre |
| 4 | **Tri du mémo par autre chose que la fraîcheur** | top 32 sur 155 ; coefficients déjà mesurés (re-zone 77,3 % au Q1 contre 88,5 % au Q5) | coefficients dispo | taux de re-zone des 32 retenus |
| 5 | **Re-calibrer le filtre de zone** | asymétrie des coûts inversée par le budget monotone | mesures 04/09 réutilisables | faux positifs / faux négatifs contre les verdicts R2 |
| 6 | **Alimentation du mémo** | solde négatif aggravé par le régime | non traité | `memo_hits`, solde/nuit |
| 7 | Génération continue 30 min | les gardes sautent ; jamais testé en continu | inconnu | thermique, VRAM, contention preuve↔décodage |

---

## 7. Ce qui reste à instruire — à faire AVANT de coder les chantiers

1. **Un *pick* remet-il le compteur de quota à zéro ?** La clé est
   `(hotkey, window_start)` donc *a priori* non — mais **toute la section 3 en dépend**.
   Si oui, c'est 32 × 16 = 512 tirs et l'analyse change du tout au tout.
2. **La fenêtre ferme-t-elle vraiment en ~18 min** sous les budgets fill-closed, ou plus
   vite si le marché est rapide ? Décide s'il y a du mou pour sélectionner.
3. **`checkpoint_identity`** est importé par le mineur de référence
   (`checkpoint_identity_from_state`). Exigé par le validateur, ou organisation interne ?
   Seul fichier neuf de #224 qui touche notre chemin.
4. **Combien de mineurs saturent leur quota ?** Détermine si les 512 places d'admission
   sont disputées. [MODÈLE] : ~35 mineurs × 32 = 1 120 pour 1 024 places ⇒ tout juste
   sursouscrit.
5. **`same_prompt_superseded`** vaut 4,4 % de nos admises sur 100 s. Sur 30 min avec une
   tranche figée, tous les mineurs ont le temps de converger vers les mêmes prompts.
   Le mémo rejoue délibérément des ex-payables — **c'est le profil qui se fait
   supplanter**. À chiffrer avant de garder le mémo tel quel.

---

## 8. Pièges de mesure propres à ce régime

- **Une fenêtre = 16 émissions.** Tout ce qui se compte « par fenêtre » change d'échelle
  d'un facteur 16 : **compter PAR HEURE**, jamais par fenêtre, pour comparer les deux
  régimes. (Le même piège a produit une fausse chute le 21/08.)
- **La randomness est figée 30 min** ⇒ un prompt raté est raté pour toute la fenêtre.
  Aucun deuxième essai, contrairement à l'intuition qu'une longue fenêtre laisse du
  rattrapage.
- **Le budget d'admission ne se rembourse pas** ⇒ un `out_of_zone` ne coûte plus 1 slot
  sur 32 sans dette (mesure du 04/09), il coûte une place. Les arbitrages du régime v5
  sur le filtre local sont **tous à refaire**.
- **La perte d'une fenêtre coûte 30 min**, pas 100 s. Les incidents deviennent rares
  mais graves : le watchdog et le repli doivent être relus sous cet angle.

---

## 9. Statut de la branche

`port/v6-fill-closed`, basée sur `fix/course-2026-08-27` (= la box), poussée.
Contient les 4 correctifs de prod du 07-09/09. Inerte sous v5 : tout est gaté par
`state.fill_closed` ou `PROTOCOL_VERSION>=6`.

**Ce qu'elle fait déjà** : wire v6, veto de tir (phase / fraîcheur / cutoff−40 s /
budget env / quota 32), gardes neutralisées, heartbeat, backoff 503, bloc launcher v6.

**Ce qu'elle ne fait pas encore** : rien de la section 6. Le port rend la bascule
**possible** ; il ne rend pas le mineur **compétitif** dans le nouveau régime.
