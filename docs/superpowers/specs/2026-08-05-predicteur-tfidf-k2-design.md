# Prédicteur de difficulté TF-IDF ciblé k=2 — design

**Date** : 2026-08-05
**Statut** : design validé (brainstorming), prêt pour plan d'implémentation
**Contexte** : subnet 81, migration 4B/v3 live. Le prédicteur 2B (word-prior,
AUC 0.605) doit être re-ciblé pour le contrat 4B/v3 et l'auction par débit.

## Problème

Sous le 4B/v3, l'auction paie `std·(1-mean)` du vecteur de reward (8 rollouts),
avec un **tie-break par débit**. Le prédicteur existant
(`reliquary/miner/prompt_predictor.py`) prédit la *moyenne* de reward et classe
par proximité de 0.5 — mauvaise cible. On veut un prédicteur qui **détecte les
prompts k≈2** (les plus payables) à partir du **texte seul**, pour classer les
5000 prompts de la tranche courante et baker les meilleurs.

Décision produit : garder l'approche **prior TF-IDF par mot** (interprétable,
CPU, stdlib), dont un sous-produit est un **classement des mots par impact**.

## Cible d'apprentissage

Cible par prompt = **`std(rewards)·(1 - mean(rewards))`** (le score d'auction),
prédite directement. Ce score **pique à k=2** (mean 0.25) et **s'effondre à k=0
et k=8** :

| k (succès/8) | mean | std·(1-mean) |
|---|---|---|
| 1 | 0.125 | 0.290 |
| **2** | **0.25** | **0.325** ← max |
| 3 | 0.375 | 0.303 |
| 4 | 0.50 | 0.250 |
| 8 | 1.0 | 0.0 |

« Viser k=2 » = « prédire le score d'auction et prendre les plus hauts ».

⚠️ **NE PAS** prédire la moyenne puis trier vers 0.25 : mesuré PIRE sur le 2B
(auction 0.126 vs 0.191) — le prédicteur imprécis déborde sous 0.25 dans le k=0
(unanime raté → hors-zone → 0 payé). La cible auction évite ce débordement par
construction (elle pénalise k=0).

## Alignement critique : validité des labels (probe ↔ mineur)

**Le probe DOIT générer dans les conditions EXACTES du mineur en prod, sinon les
labels sont invalides.** Config prod = `ops/launch_miner.sh` + `PLAN_REPRISE_GPU.md` :

- **`RELIQUARY_MAX_NEW_TOKENS=2600`** — cap opérationnel délibéré (choix DÉBIT).
  Étude §5 : les réponses code gagnantes font 600-1000 tokens ; cap court = 6×
  moins de gaspillage + 2× débit agrégé. Sous tie-break v3, un k=2 lent est pire
  qu'un drop + régénération de prompts rapides.
- **`RELIQUARY_MAX_TRUNCATED_CODE=0`** — un seul rollout tronqué (cap sans EOS)
  jette tout le groupe (`too_many_truncated`, `engine.py:485`).
- Sampler protocole (T0.6 / top_p0.95 / top_k20), `TOP_P_PROTO`/`TOP_K_PROTO`.

**Conséquence — soumissibilité gratuite** : un prompt dont la solution dépasse
2600 tokens → rollouts tronqués (pas d'EOS) → reward ~0 → sort en **k=0** dans le
label → le prédicteur apprend **tout seul** à l'éviter. Le cap court encode donc
la stratégie débit directement dans les labels ; **aucune modélisation de
longueur n'est nécessaire**. Le seul filtre de drop est la non-terminaison (pas
d'EOS), PAS un seuil de longueur — au cap 2600 les deux coïncident.

⚠️ **Hypothèse rejetée** : « prompt dur = plus de tokens » n'est PAS mesuré sur
le 4B code. On ne l'assume pas. On aligne sur 2600 point.

## Composants

### 1. `prompt_predictor.py` (modifié, stdlib-only)
- `tokenize` : inchangé (unigrams + bigrams).
- `train_word_priors` : inchangé mécaniquement (empirical-Bayes k=10) mais **la
  cible passe de mean à `std·(1-mean)`** (calculée en amont, cf. `auction_score`).
- `score_prompt` : inchangé (moyenne des priors pondérée idf) → prédit le score
  d'auction.
- **Sélection simplifiée** : `selection_score` devient l'identité sur le score
  prédit (plus haut = mieux) ; `select_eligible` classe décroissant. On retire
  le `-|pred-0.5|`.
- `auction_score(rewards) -> float` : nouvelle fonction pure = `std·(1-mean)`.

### 2. `word_impact_report(model, min_df=N)` (nouveau)
Pour chaque mot avec `df >= min_df` : sort `(mot, prior, df, idf, écart_global)`.
- Trié décroissant → **mots-signal payables** (associés à k≈2).
- Trié croissant → **mots unanimes** (k=0/k=8, sans valeur).
Export JSON/CSV inspectable. C'est la **réponse empirique** à « les mots
portent-ils un signal sur le 4B, et lesquels ».

### 3. Évaluation / gate de déploiement
Split held-out. Métriques :
- **(a)** Spearman(score prédit, vrai score auction).
- **(b)** Valeur d'auction RÉELLE moyenne du top-N sélectionné vs tirage
  aléatoire (métrique décisionnelle, cf. pilote 0.202 vs 0.191).
- **(c)** % du top-N tombant dans k∈[2,6].
**Gate** : on ne câble dans le mineur que si (b) bat l'aléatoire d'une marge
nette. Sinon, résultat négatif honnête (les mots ne portent pas de signal 4B) →
décision séparée.

### 4. Probe (`scripts/difficulty_probe.py`, ajustements mineurs)
- Générer à **cap 2600**, `MAX_TRUNCATED_CODE=0`, sampler protocole, env code,
  checkpoint 4B `ReliquaryForge/qwen3.5-4b-reliquary-v4`, backend vLLM.
- Enregistrer par rollout : reward + **terminé (EOS) / tronqué** (l'info est déjà
  disponible via `first_eos_index`) — utile au diagnostic, pas au label (le k=0
  des tronqués suffit).
- Format de sortie inchangé (`prompt` / `rewards` / `in_zone`), concaténable avec
  d'éventuels dumps prod.
- Séparé du mineur (cf. préférence : probe dédié, pas de collecte en prod).
- Volume : commencer petit (~2-3k prompts valides à 2600) et regarder la courbe ;
  monter si le signal progresse.

### 5. Câblage runtime (derrière flag, off par défaut)
Le mineur charge le JSON, score la tranche de 5000 via `score_prompt`, classe
décroissant, passe le top-N à `selector.next(eligible=set(...))`. Activé
seulement après passage du gate.

## Flux

```
[GPU H100]  probe 4B @ cap 2600  → dump.jsonl {prompt, rewards[8], eos_flags}
[CPU]       auction_score → train_word_priors(cible=auction)
            → word_impact_report → eval/gate (Spearman, top-N vs random)
[GPU]       si gate OK : miner charge JSON, classe la tranche, bake le top-N
```

## Travail réalisable AUJOURD'HUI (sans lancer le GPU)
Tout le code + tests sur données synthétiques (CPU, dev box) : `auction_score`,
la bascule de cible dans le training, `word_impact_report`, les 3 métriques
d'eval, et les ajustements probe (cap/flags/enregistrement EOS). Prêt à tourner
dès que le dump 4B @ 2600 arrive.

## Caveat (non bloquant)
Le probe génère en **sampling libre** protocole ; le mineur sous enforcement
génère en **forced-seed** (tokens forcés aux positions confiantes → rollouts
moins divers). Le forced-seed n'est PAS reproductible offline (dépend de la
randomness par-fenêtre `u_at`). Le sampling libre protocole est le meilleur
proxy — biais uniforme sur le `std`, préserve probablement le classement. À
valider un jour sur des prompts réellement servis en prod.

## Hors scope (YAGNI)
- Features structurelles / char n-grams / embeddings / arbres boostés : écartés
  au profit du prior TF-IDF interprétable. Réévaluer seulement si le gate échoue.
- Modélisation explicite de la longueur / soumissibilité : inutile (le cap 2600
  la replie dans le label k=0).
