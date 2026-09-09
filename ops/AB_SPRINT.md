# Protocole A/B — réactiver le SPRINT

**État : PRÊT, NON LANCÉ. Demande un redémarrage du mineur, donc un go explicite.**

## Le constat

`vllm_backend.generate_forced_phase1_multi_stream` implémente un « sprint » :
les `sprint_size` premiers prompts du classement décodent **seuls**, avec moins
de séquences en vol, donc plus vite par séquence. Sa docstring : « 4 par défaut :
32 séquences en vol au lieu de 128 → décodage par séquence ~2× plus rapide → le
meilleur candidat arrive des rounds plus tôt ».

Il est **neutralisé** aujourd'hui :

```python
n_sprint = max(0, min(int(sprint_size or 0), n))
if n_sprint >= n:
    n_sprint = 0        # sprint = tout le lot : aucun sens, tout part
```

Le launcher exporte `RELIQUARY_BAKE_BATCH_SIZE=8` **et** `RELIQUARY_SPRINT_SIZE=8`.
`n_sprint >= n` → remis à zéro. Les 8 prompts partent ensemble, 128 séquences.

Ce réglage vient du commit `16ab655` (18/08), délibéré : « sprint complet — tout
le bake en un vol de génération », justifié par « ~5 slots/fenêtre perdus
post-seal, seal à 10-40 s ». **Cette prémisse ne tient plus** : 65 % de nos
envois partent après 12 s et 14 % seulement sont admis.

## Ce que la mesure prédit

Mesuré le 21/08 sur 1 097 bakes : dans un lot de 8, les groupes sortent par
ordre croissant de longueur — position 1 à 2,8 s (2 909 tok), position 8 à
6,8 s (6 055 tok). Le plus volumineux sort en position 6-8 dans **87 %** des cas.

Sur les 620 entrées réellement admises, le volume paie net du péage d'arrivée
(Spearman volume↔bucket = +0,471), mais nos meilleures entrées plafonnent à
**bucket 28**, juste sous le seuil de 29 au-delà duquel 81 % des fenêtres paient.
Un round gagné les porterait à **40**.

## Métrique de décision

**Bucket de la meilleure entrée de la fenêtre** — `tokens // (rounds × 50)`.
Ligne de base : moyenne **27,4**, écart-type 8,3, médiane 28, 48 % de fenêtres
au-dessus de 29.

| effet à détecter | fenêtres par bras | durée (14 fen/h) |
|---|---|---|
| +5 bucket | 43 | 3,1 h |
| +7 bucket | 22 | 1,6 h |
| +12 (prédit) | 7 | 40 min |

⛔ **NE PAS trancher sur les payées/fenêtre** : moyenne 2,44 pour un écart-type
de 2,66 → détecter +0,30 demanderait **1 234 fenêtres par bras, soit 88 h**.
C'est ce piège qui a fait juger « sprint 8 bon puis mauvais » par le passé.

## Déroulé

1. **Bras A (témoin)** — relever la ligne de base sur la configuration actuelle :
   `python3 scripts/ab_sprint.py --depuis "<T0>" --jusqu-a "<T0+1h30>"`
2. **Basculer** : `RELIQUARY_SPRINT_SIZE=4` dans `ops/launch_miner_v4.sh`, puis
   redémarrer le mineur. ⚠️ En profiter pour retirer `HF_XET_HIGH_PERFORMANCE=1`
   (inerte, posé sur une prémisse fausse).
3. **Attendre 15 minutes** avant de compter quoi que ce soit — règle d'or : ne
   jamais juger une configuration sans contrôler l'âge du moteur.
4. **Bras B** : `python3 scripts/ab_sprint.py --comparer "<A>" "<B>"`.
   Le script sort l'écart, son erreur-type, l'IC 95 %, et refuse de conclure
   quand l'écart n'est pas distinguable du bruit.

## Règles de mesure (chacune payée d'une erreur)

1. **Âge du moteur** — écarter les 15 premières minutes après un redémarrage.
2. **Fenêtres de checkpoint écartées** — le rechargement en coûte 1 à 2, sans
   rapport avec le réglage testé (~1 avancée par heure).
3. **Verdicts mûrs seulement** — une fenêtre dont `rewarded` est encore `None`
   n'est pas « zéro payée », elle n'est pas décidée.
4. Le round drand est reconstruit depuis `flip_offset_s` : proxy exact à 84 %
   contre `arrival_drand_round`, écarts de ±1. Bon pour l'agrégat, pas pour
   juger une entrée isolée.

## Repli

Remettre `RELIQUARY_SPRINT_SIZE=8` et redémarrer. Aucun état persistant n'est
touché : le sprint ne concerne que l'ordre d'enfilement des requêtes vLLM.

## Réserve honnête

La **magnitude** n'est pas mesurée. L'estimation « 2× plus rapide par séquence »
vient de la docstring et d'un banc ancien (102 tok/s/seq à 32 séquences ; ~55
observés à 128, sur un autre checkpoint). C'est bien l'objet de l'A/B.
