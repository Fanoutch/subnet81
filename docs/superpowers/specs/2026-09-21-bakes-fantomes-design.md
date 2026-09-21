# Bakes fantômes — étiqueter des prompts pendant le temps mort GPU (21/09)

## Pourquoi
Sous V1, un cycle de fenêtre dure ~687 s (médiane) et le mineur ne produit que
~263 s : le GPU dort ~62 % du temps, pendant le trou 503. La variable qui décide
du hors-zone est « ce prompt a-t-il été mesuré récemment, et avec quel
verdict » : un prompt dont la dernière mesure était en zone est re-jeté à ~21 %,
un prompt jamais mesuré à ~39 %, un prompt déjà hors zone à ~60 % (8 134 picks
classés, 67 fenêtres). On utilise le temps mort pour fabriquer ces mesures.
Gain estimé (inféré, à confirmer en vol) : +0,8 à +1,1 payé/fenêtre en plus de
la liste noire à 20 000 fenêtres.

## Branchement
Dans `_generator_loop`, branche `_bake_paused` (celle qui ne faisait que
`_pause_wait(1.0)` pendant le trou 503) : si `_ghost_ready(now)`, on exécute UN
lot fantôme à la place de l'attente, dans la même coroutine que les bakes de
production. L'exclusion mutuelle avec les vrais bakes est donc garantie par
construction.

## Sécurité (deux garde-fous indépendants)
1. **Fenêtre de tir** : `T_MIN ≤ now − _window_open_ts ≤ T_MAX` (330 / 537 s par
   défaut). `_window_open_ts` est l'ouverture exacte publiée par le validateur.
   Sur 349 fenêtres, le cycle n'est jamais descendu sous 627 s : on s'arrête
   90 s avant le flip le plus précoce observé.
2. **Interruption au flip** : le `should_abort` passé au moteur devient vrai dès
   que la randomness de fenêtre change (le flip `/miner-state` la met à jour
   immédiatement) ou que `T_MAX` est dépassé. Le moteur le vérifie à chaque pas
   et annule les requêtes : le verrou `_VLLM_CALL_LOCK` est rendu en ~un pas.
Conditions supplémentaires : trou 503 réel (dernier `/state` 200 plus vieux que
`_V6_MAX_STATE_AGE_S`), poids synchronisés, backend vLLM et table de scores
présents.

## Ce qu'on génère
Lots de `RELIQUARY_GHOST_LOT` prompts (défaut 8) tirés dans la bande
p95-p99 du score du prior sur l'univers entier (~99 000 prompts, l'équivalent
des rangs 51-250 d'une tranche). Exclus : liste noire active, ex-payables du
mémo mesurés il y a moins de 250 fenêtres, prompts déjà étiquetés en fantôme il
y a moins de 250 fenêtres. Randomness SYNTHÉTIQUE (hash préfixé, ne peut pas
coïncider avec une vraie), checkpoint courant.

## Étiquetage et alimentation
Notation par `grade_group_parallel_ex`, décision `_skip_for_out_of_zone` sur
`timeout_imputed_for_zone` : exactement le filtre de production. Dump dans
`RELIQUARY_GHOST_DUMP`. Si `RELIQUARY_GHOST_FEED=1` :
- en zone et aucun rollout tronqué → `payable_memo.update(…, True, …)` ;
- hors zone → `_sz_note` (liste noire, 20 000 fenêtres).
Jamais touchés : `_pool`, `_phase1_cache`, `_cached_randomness`, preuve, tirs.

## Déploiement
- **Étape A** : `GHOST_BAKE=1`, `GHOST_FEED=0`. Vérifier l'absence de
  perturbation : 1er bake de chaque fenêtre toujours à +1,2-1,5 s, groupe 1
  prêt inchangé ; compter lots/heure et interruptions.
- **Étape B** : `GHOST_FEED=1`. Juger sur plusieurs jours : réserve mémo fraîche
  par tranche, jetés du 1er bake, payés (R2, contrôle marché).
Repli : `RELIQUARY_GHOST_BAKE=0` (défaut) = comportement historique.
