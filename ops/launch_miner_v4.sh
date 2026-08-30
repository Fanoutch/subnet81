#!/bin/bash
# Launcher v4 compatible restart/watchdog (balayage 18/08 G1) — à déployer sur
# la box en /workspace/launch_miner_v4.sh, puis activer la chaîne de relance :
#   echo /workspace/launch_miner_v4.sh > /workspace/.miner_launcher
# (restart_miner.sh lit ce marqueur ; sans lui il relance launch_miner.sh = v3
# = 100 % GENERATION_CONTRACT_MISMATCH après le premier restart du watchdog.)
# AUTONOME : ne passe PAS par ops/launch_miner.sh (ses défauts v3 — cap 2600,
# checkpoint ReliquaryForge v3, batch 40 — sont du poison en v4) ; l'infra
# éprouvée (venv, env vLLM, garde GPU-vide) est reproduite ici.
set -uo pipefail

# ── Garde anti-course d'init (incident 2026-08-13) : au restart un EngineCore
# zombie peut tenir la VRAM plusieurs minutes → l'init vLLM brûle ses 5
# tentatives. On attend le GPU vide (le kill zombie de vllm_backend couvre le
# reste en régime).
for i in $(seq 1 24); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ -z "$used" ] || [ "$used" -lt 2000 ] && break
  echo "launch_v4: GPU encore occupé (${used} MiB), attente ($i/24)..."
  sleep 5
done

export PYTHONPATH=/workspace/reliquary-miner-priv
export HF_HOME=/workspace/hf
export GRAIL_ATTN_IMPL=sdpa

# ── Réseau — AUTO-DÉTECTION egress (18/08 : impossible de distinguer « box
# filtrée » de « validateur down » tant qu'il ne répond pas ; on teste au
# lancement). Direct OK → direct ; sinon tunnel inverse 127.0.0.1:8080
# (monté DEPUIS la dev box : tmux tunnel81, boucle ssh -N -R auto-retry).
# ATTENTE au lieu d'ABORT (fix 20/08) : le 20/08 à 18:21 le validateur est
# tombé en 502 ; le launcher a abandonné et le mineur est resté MORT 37 min
# (le watchdog ne relançait pas un process absent — corrigé aussi). On teste
# désormais direct puis tunnel en boucle, avec un plafond généreux : une panne
# validateur ne doit jamais nous laisser hors ligne.
if [ -z "${RELIQUARY_VALIDATOR_URL:-}" ]; then
  _try=0
  _max=${RELIQUARY_EGRESS_WAIT_TRIES:-240}   # 240 x 15 s = 1 h
  while [ "$_try" -lt "$_max" ]; do
    _code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 4 \
            http://209.20.157.231:8080/health 2>/dev/null)
    if [ "$_code" = "200" ]; then
      export RELIQUARY_VALIDATOR_URL=http://209.20.157.231:8080
      echo "launch_v4: egress DIRECT vers le validateur"
      break
    fi
    _code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 4 \
            http://127.0.0.1:8080/health 2>/dev/null)
    if [ "$_code" = "200" ]; then
      export RELIQUARY_VALIDATOR_URL=http://127.0.0.1:8080
      echo "launch_v4: egress via TUNNEL inverse (dev box)"
      break
    fi
    _try=$((_try + 1))
    [ $((_try % 4)) -eq 1 ] && \
      echo "launch_v4: validateur injoignable (HTTP $_code) — attente ${_try}/${_max}" >&2
    sleep 15
  done
  if [ -z "${RELIQUARY_VALIDATOR_URL:-}" ]; then
    echo "launch_v4: ABORT — validateur injoignable depuis 1 h" >&2
    exit 1
  fi
fi
# secureweb3 exclu : injoignable (gel 15-20 s/tirage, flips ratés 28968-70) ;
# les 4 miroirs re-testés OK depuis CETTE box le 2026-08-18.
export RELIQUARY_DRAND_URLS=${RELIQUARY_DRAND_URLS:-"https://api3.drand.sh,https://drand.cloudflare.com,https://api.drand.sh,https://api2.drand.sh"}

# ── Bascule v4 : LE flag + les ceintures ────────────────────────────────────
# BASCULE v5 (24/08) : le validateur tourne `qwen3-4b-base-dapo-reasoning-v5`
# depuis le 23/08 (PR #190, image cba84ce). Le seul changement qui nous touche
# est le PROMPT, desormais rendu via un template versionne — porte dans
# reliquary/protocol/profiles.py, parite sha256 verifiee contre /health.
# Repli : remettre 4 (le chemin legacy reste byte-exact, teste).
export RELIQUARY_PROTOCOL_VERSION=${RELIQUARY_PROTOCOL_VERSION:-5}
export RELIQUARY_AUCTION_MIN_SCORE=${RELIQUARY_AUCTION_MIN_SCORE:-0}
export RELIQUARY_RANKING_BUDGET_S=${RELIQUARY_RANKING_BUDGET_S:-12}
export RELIQUARY_SAMPLE_DUMP=${RELIQUARY_SAMPLE_DUMP:-/workspace/samples_v4.jsonl}
# Slot mémo : ON avec store FRAIS (le mémo s'amorce depuis SAMPLE_DUMP — le
# chemin v4 neuf garantit zéro contamination v3 ; il se remplit tout seul).
# MEMO_MIN_SCORE calibré sur l'étude offline 18/08 (128 groupes code réels) :
# payable = 95,3 % → une table de « payables » ne discrimine rien ; le seuil
# 0.23 ≈ p75 des scores d'enchère observés (p90 0.261, max 0.312) fait du
# mémo une table de VEDETTES — et H1 (corr 0.824, P(pay→pay) 100 %) dit
# qu'une vedette mesurée le reste. Re-calibrer sur les données live à H+24.
export RELIQUARY_MEMO_SLOT=${RELIQUARY_MEMO_SLOT:-1}
export RELIQUARY_MEMO_MIN_SCORE=${RELIQUARY_MEMO_MIN_SCORE:-0.23}
# Fix vitesse 18/08 soir : hold-off balayage (preuves vedettes sans contention)
# Vedette memo rapide : bande de longueur mesuree (meta gagnant 2900-7200 tok)
# Couvre-feu de bake (etude drain 18/08) : pas de nouveau bake apres T+95s
# (BAKE_CURFEW_S / SCAN_HOLDOFF_S / MEMO_FAST_BAND retirés le 30/08 : exports
#  ORPHELINS, aucun lecteur dans le code — audit timing, grep intégral.)
# ⚠️ OUBLI DU 1er LANCEMENT (18/08 16h16-16h50, trouvé car 0 hit mémo) : sans
# ce flag, prompt_range=None → picks NON confinés à la tranche de fenêtre ET
# slot mémo jamais armé. Le prod v3 l'a toujours eu — toute la stratégie de
# tranche (mémo, classement, retombées) en dépend.
export RELIQUARY_PROMPT_RANGE_FROM_WINDOW=${RELIQUARY_PROMPT_RANGE_FROM_WINDOW:-0}
# Instrumentation étude v4 (etudev4.md §B) : chaque fenêtre minée sans ces
# logs = de l'étiquetage gratuit perdu (H1-H11). Rapatriés par pull81.
export RELIQUARY_VERDICTS_DUMP=${RELIQUARY_VERDICTS_DUMP:-/workspace/verdicts_v4.jsonl}
export RELIQUARY_SUBMIT_DUMP=${RELIQUARY_SUBMIT_DUMP:-/workspace/submits_v4.jsonl}
export RELIQUARY_WINDOW_DUMP=${RELIQUARY_WINDOW_DUMP:-/workspace/windows_v4.jsonl}
# ARCHITECTURE 2 MINEURS (décision 18/08) : un env par box, pleine puissance
# chacun — le quota 32/fenêtre est PAR ENV (un batcher par env côté
# validateur), même hotkey OK, zéro collision (espaces de prompts disjoints).
#   Box 1 (prod, la première lancée) : opencodeinstruct — ce défaut.
#   Box 2 (2e H200, quand louée)     : RELIQUARY_ACTIVE_ENVS=openmathinstruct
#     (l'étude offline du 18/08 a montré le math TRÈS minable en v4 :
#      boxing 91,5 %, payable 94,5 %, répétabilité corr 0.881).
# Un seul mineur PEUT faire les 2 (dual-env alterné) mais à débit partagé —
# fallback si une seule box : openmathinstruct,opencodeinstruct.
export RELIQUARY_ACTIVE_ENVS=${RELIQUARY_ACTIVE_ENVS:-opencodeinstruct}
# PURGE des réglages v3 hérités (K_MIN/K_MAX 2/6, MAX_NEW_TOKENS 2600/16384,
# prédicteurs v3, sprint 90 s, seuils q10 v3) : ne PAS les poser ici.
unset RELIQUARY_K_MIN RELIQUARY_K_MAX RELIQUARY_MAX_NEW_TOKENS \
      RELIQUARY_PROMPT_PREDICTOR_2 \
      RELIQUARY_MIN_LOCAL_Q10 RELIQUARY_MIN_LOCAL_MEDIAN \
      RELIQUARY_SPRINT_MAX_WAIT_S RELIQUARY_MAX_TRUNCATED_CODE 2>/dev/null || true
# ── PRIOR v5.0 (câblé 18/08 ~19h, go utilisateur) : entraîné 100 % données v4
# (2 156 groupes, cible = score d'enchère), holdout propre Spearman 0.313,
# P(vedette|top-20%) 36,5 % vs 25,1 %. Vérifié live : seules les vedettes
# paient en fenêtre disputée (29400 : k=3/0.255 payé, 9 picks aléatoires
# rangs 26-45 perdus). Ré-entraîner ~quotidien (scripts/train_prior_v50.py).
export RELIQUARY_PROMPT_PREDICTOR=${RELIQUARY_PROMPT_PREDICTOR:-/workspace/predictor_v59.json}
# 2 slots explore = labels non biaisés pour les ré-entraînements (obligatoire
# dès qu'un prior influence les picks — leçon v4.3/mémorisation).
export RELIQUARY_EXPLORE_SLOTS=${RELIQUARY_EXPLORE_SLOTS:-0}

# ── vLLM (calibré au BANC v4 H200 18/08, gate parité PASS des 2 modes) ─────
# Banc (4B-Base, M=16, 1024 tok, fs ON) : coût forced-seed ~4,7 % (full-support
# = plus de tri top_p) ; CUDA graphs ×2,19 ; prefix caching neutre-négatif
# (OFF) ; courbe longueur PLATE jusqu'à 8192. Frontière graphs :
#   32 seqs 5134 tok/s (160/seq) · 64: 8469 (132) · 128: 12525 (98) · 256: 16028 (63)
# Gate forced-seed v4 : PASS eager 0.9572/0.9123, PASS GRAPHS 0.9674/0.9388.
export RELIQUARY_VLLM_FORCED_SEED=1       # sans lui : boucle HF sync = débit mort
export RELIQUARY_VLLM_CUDA_GRAPHS=${RELIQUARY_VLLM_CUDA_GRAPHS:-1}  # ×2,19, parité PASS 0.9674
export RELIQUARY_VLLM_GPU_FRACTION=${RELIQUARY_VLLM_GPU_FRACTION:-0.76}  # post-OOM 16/08 (0 OOM au banc)
export RELIQUARY_VLLM_MAX_NUM_SEQS=${RELIQUARY_VLLM_MAX_NUM_SEQS:-256}  # couvre 16 prompts×16 ; 512 = captures/VRAM pour rien
# M=16 rollouts en v4 : batch de B prompts = B×16 séquences en vol.
# 8 = 128 séquences = 12,5k tok/s agrégés à 98 tok/s/seq (mesuré graphs).
# Arbitrage rang vs couverture : par-groupe = per-seq×16 → sprint étroit
# (2-3 prompts, 160-140/seq) pour le rang, scan large (8) pour la couverture.
export RELIQUARY_BAKE_BATCH_SIZE=${RELIQUARY_BAKE_BATCH_SIZE:-8}
# ── Fix seal 18/08 (contrefactuel : ~5 slots/fenêtre perdus post-seal, seal à
# 10-40 s ; concurrence médiane 0.250 aux rangs 4-9 confirmée) : tout le bake
# en UN vol de génération + grading concurrent → les 8 groupes soumis <15 s.
# SPRINT ramene au DEFAUT DU CODE (4) le 24/08. A 8 il valait BAKE_BATCH_SIZE,
# donc `if n_sprint >= n: n_sprint = 0` le desactivait : les 8 prompts partaient
# ensemble, 128 sequences en vol (~55 tok/s/seq contre 102 a 32 au banc H200).
# A 4 : les 4 tetes de classement decodent seules, 64 sequences, elles sortent
# plus tot — et le ROUND decide (il vaut >1000 tokens). Le reglage 8 datait du
# 18/08 et se justifiait par « ~5 slots perdus post-seal » ; premisse tombee,
# 65 % de nos envois partaient apres la fermeture du batch.
# Bras suivant a tester : 2 (32 sequences). Protocole : ops/AB_SPRINT.md.
# 4 -> 8 le 24/08 : a 8 (= BAKE_BATCH_SIZE) le code fait `n_sprint = 0`, donc
# le sprint est DESACTIVE et les 8 prompts partent ENSEMBLE (128 sequences).
# POURQUOI : mesure sur 97 admises — le balayage (positions 5-8) n'arrive
# JAMAIS sous 12 s (0 sur 30), or la bande sous 12 s est la seule qui paie
# (55 % contre 6 % au-dessus). Le balayage n'est enfile qu'a la livraison du
# DERNIER groupe du sprint (~6,7 s) : c'est un retard d'ORDONNANCEMENT, pas de
# generation. 72 % de nos fenetres n'ont aucune entree sous 12 s.
# CRITERE DE LECTURE DIRECTE, une fenetre suffit — ligne `groupe 8/8 pret a X`:
#   X <= 9 s  -> la cadence tient a 128 seqs, on garde (et monter le sprint a 6
#                serait aussi valide)
#   X >= 13 s -> la cadence s'effondre, REVENIR A 4 immediatement
# La cadence est MESUREE plate de 16 a 64 sequences (135-149 steps/s,
# Spearman -0,033) ; au-dela elle n'est qu'estimee (124 steps/s), d'ou le test.
# Repli : remettre 4.
# RETOUR A 4 le 24/08, apres un A/B mesure.
#
# Ce que le test a etabli (sprint desactive = 8 = BAKE_BATCH_SIZE) :
#  + la cadence TIENT a 128 sequences : groupe 8/8 pret a 7,7-8,2 s contre
#    14,2 s a sprint=4, et la tete sort meme plus tot (3,4 vs 4,1 s). La
#    premisse du code (« moins de sequences = plus rapide ») est FAUSSE, et
#    ops/AB_SPRINT.md visait dans la mauvaise direction.
#  + arrivee mediane 23,6 -> 14,0 s, et 5x plus d'entrees sous 12 s.
#  - MAIS le rang ne s'ameliore pas (37 -> 41) et 9 fenetres d'affilee sans
#    paiement.
#
# POURQUOI : la cle de tri du validateur est (-valeur, bucket, round, tiebreak)
# — la VALEUR passe AVANT le bucket. Mesure : le bucket monte bien (16 -> 22)
# mais le score d'enchere BAISSE de 11 % (0,162 -> 0,144) et k passe de 9,1 a
# 9,8. Sans sprint, les 8 groupes concourent ensemble et les premiers finis
# partent : or un groupe FACILE produit moins de tokens et finit plus tot. On
# selectionne donc involontairement les faciles, qui valent moins.
#
# Le sprint ne protege ni la vitesse ni le volume : il protege la VALEUR des
# tetes choisies par le predicteur.
#
# /!\ Echantillon modeste (33 vs 39 entrees). A re-mesurer sur ~30 fenetres
# par bras si on veut y revenir.
# 28/08 bras 2+3 : sprint=2 (la paire de tête décode à 32 séquences) + voie
# FIFO de tête. Le sprint seul a montré que le slot passé SEUL paie (envoi
# 9,2 s, rangs 4-15) mais que son jumeau faisait la queue (envoi 18,5 s,
# rangs 21+) : HEAD_FIFO=2 sérialise grade→tir des 2 premiers groupes livrés,
# dans l'ordre. Repli : HEAD_FIFO=0 (comportement d'avant), SPRINT_SIZE=4.
# 30/08 : SPRINT OFF (=BAKE_BATCH_SIZE) — ses 3 piliers sont morts sous #217
# (plus de course d'admission, plus de départage arrivée, OFF livre tout plus
# tôt). A/B entrelacé de confirmation dès 30 fenêtres mûres.
export RELIQUARY_SPRINT_SIZE=${RELIQUARY_SPRINT_SIZE:-8}
# HEAD_FIFO : optimisation de l'ancienne économie (priorité d'arrivée des
# têtes), jamais mesurée en prod — repli 0 pour la relance, A/B plus tard.
export RELIQUARY_HEAD_FIFO=${RELIQUARY_HEAD_FIFO:-0}
export RELIQUARY_HEAD_FIFO_WAIT_S=${RELIQUARY_HEAD_FIFO_WAIT_S:-12}
export RELIQUARY_GRADE_CONCURRENCY=${RELIQUARY_GRADE_CONCURRENCY:-3}

# --- LATENCE D'AMORÇAGE (24/08) -------------------------------------------
# Mesuré : le classement de tranche coûte 2,80 s p50 EN TÊTE de chaque
# fenêtre, sur le thread asyncio (aucun POST ne part pendant ce temps).
# Décomposition : parquet HF 0,95 s + get_problem 0,30 s + notation 0,91 s.
# Enjeu : 1 s d'arrivée = +1,46 place = 390 tokens ; -3 s double les payées.
#
# 1) MIROIR PARQUET LOCAL — supprime les 0,95 s de réseau et les 5,4 % de
#    fenêtres où un timeout HF fait exploser le budget (+9,5 s sur le 1er
#    groupe). Le fichier fait 1,39 Go ; ~85 Go libres sur /workspace.
#    ⚠️ RELIQUARY_PARQUET_EXPECTED_LEN est une GARDE : len() est le consensus
#    prompt-range, un miroir incomplet donnerait 100 % de prompt_out_of_range.
#    Vide => chemin distant historique, inchangé.
export RELIQUARY_PARQUET_LOCAL_ROOT=${RELIQUARY_PARQUET_LOCAL_ROOT:-/workspace/parquet_mirror}
export RELIQUARY_PARQUET_EXPECTED_LEN=${RELIQUARY_PARQUET_EXPECTED_LEN:-2481806}
#
# 2) TABLE DE SCORES PRÉ-CALCULÉE — supprime les 0,91 s de notation ET les
#    lectures de prompts. Générer avec :
#      python3 scripts/precompute_prompt_scores.py --out /workspace/prompt_scores.npz
#    La table porte une empreinte des 3 modèles : un prior ré-entraîné la
#    périme et le mineur retombe SEUL sur la notation en direct.
#    Vide => notation en direct, inchangée.
export RELIQUARY_PROMPT_SCORES=${RELIQUARY_PROMPT_SCORES:-/workspace/prompt_scores_v58_vol2.npz}
# Mode course 2026-08-19 : garde pré-flip (GPU libre au flip) + rafale 8
# 30/08 : fenêtres médianes 102 s (p10 87), seals anticipés 72-82 s.
# lf<gf OBLIGATOIRE (l'inverse rend la zone capped inatteignable).
# ⚠️ PREMIER A/B de la relance : guard 50 (bake continu) vs 10 (mode course
# qui faisait 16,5 payées/h sous l'ANCIENNE clé — inconnu sous la nouvelle).
export RELIQUARY_LATE_BAKE_FROM=${RELIQUARY_LATE_BAKE_FROM:-35}
export RELIQUARY_PREFLIP_GUARD_S=${RELIQUARY_PREFLIP_GUARD_S:-50}
export RELIQUARY_LATE_BAKE_CAP=${RELIQUARY_LATE_BAKE_CAP:-1200}
# Streaming C 19/08 : preuve spéculative des têtes de rafale (parallèle au grading)
# 27/08 : SPEC_PROOF remis à 0 — mesuré sans effet (grading 0,06 s, rien à
# paralléliser) ; le flip à 1 du 27/08 21:20 était une variable non contrôlée.
export RELIQUARY_SPEC_PROOF=${RELIQUARY_SPEC_PROOF:-0}
export RELIQUARY_SPEC_PROOF_SLOTS=${RELIQUARY_SPEC_PROOF_SLOTS:-4}
# AUTO-FILTRAGE 19/08 (rapport agents) : miroir local des checks validateur,
# posé APRÈS le bloc unset des seuils v3 plus haut — marges sûres v4.
export RELIQUARY_MIN_LOCAL_Q10=${RELIQUARY_MIN_LOCAL_Q10:-0.0005}
export RELIQUARY_MIN_LOCAL_MEDIAN=${RELIQUARY_MIN_LOCAL_MEDIAN:-0.08}
export RELIQUARY_LOCAL_TOKEN_AUTH=${RELIQUARY_LOCAL_TOKEN_AUTH:-1}
export RELIQUARY_LTA_CHOSEN_MAX=${RELIQUARY_LTA_CHOSEN_MAX:-1e-5}
export RELIQUARY_LTA_ARGMAX_MIN=${RELIQUARY_LTA_ARGMAX_MIN:-0.985}
# 25/08 22h — regime ckpt 660 (modele de base : ecrit 2,4x plus long, k=16 disparu).
# Le filtre dur passait de 3,8 % a 33 % de la production. Leur seuil REEL est
# TOKEN_AUTH_THRESHOLD=1e-8 (constants.py:1309), applique SANS condition
# d'argmax. On garde une marge x3 au lieu de x10. Repli : 1e-7.
export RELIQUARY_LTA_HARD_MIN=${RELIQUARY_LTA_HARD_MIN:-3e-8}
# Malus anti-rollout-court (20/08) : dé-priorise à la SÉLECTION les prompts
# qui produisent des rollouts <32 tok (inéligibles CHALLENGE_K, 0 payé/333).
export RELIQUARY_SHORT_RISK_MODEL=${RELIQUARY_SHORT_RISK_MODEL:-/workspace/risk_short_v1.json}
export RELIQUARY_SHORT_RISK_LAMBDA=${RELIQUARY_SHORT_RISK_LAMBDA:-0.08}
# GATE ROLLOUT COURT retire le 21/08 (upstream PR #188) : le validateur verifie
# desormais les completions <32 tokens a couverture complete au lieu de les
# rejeter d'office. Securite VERIFIEE en vol : 9 entrees courtes envoyees,
# 7 verdicts decides, ZERO logprob_mismatch, ZERO fenetre a dette.
# /!\ Gain NON demontre : l'A/B convergeait vers zero (+0,73 a 5 fenetres,
# +0,03 a 10) — on bake 100-140 groupes/fenetre et on n'en place que 3-5, donc
# les entrees courtes se SUBSTITUENT au lieu de s'ajouter. Repli : remettre 32.
export RELIQUARY_MIN_ROLLOUT_LEN=${RELIQUARY_MIN_ROLLOUT_LEN:-0}
# Bonus de VOLUME (20/08) : le rang du validateur est tokens // (rounds x 50),
# donc à arrivée égale le volume EST le rang. Mesuré : 7 % de payées sous 3 000
# tokens, 54 % au-dessus de 6 000. mu=0,05 calibré à la vraie pression de
# sélection (8 retenus sur 300, prompts jamais vus) : part des groupes >=6000
# tok 31 % -> 65 %, SANS perdre un groupe payable (in_zone reste 100 %).
export RELIQUARY_VOLUME_MODEL=${RELIQUARY_VOLUME_MODEL:-/workspace/volume_v2.json}
# 27/08 (archives R2, 1 245 couples vérifiés, 0 exception) : le paiement est
# PLAT — émission = groupes retenus / 32, le rang ne module RIEN. Le volume ne
# rapporte donc rien et coûte 0,86 s / 1000 tok d'arrivée (1 617 paires
# intra-fenêtre×mineur). Nos groupes : 11 048 tok vs 8 772 marché = +2,0 s.
# ⚠️ à surveiller : plus de sigma=0 possibles (prompts plus faciles). Repli : 0.05.
export RELIQUARY_VOLUME_MU=${RELIQUARY_VOLUME_MU:-0}
# File d'envoi (20/08) : jusqu'ici UN SEUL envoi en vol — quand le POST de la
# 1re entrée traînait (validateur lent), TOUTE la fenêtre attendait derrière,
# puis partait d'un bloc. Mesuré sur les fenêtres 29888/29889 : des entrées
# prêtes à +7,4 s ne partaient qu'à +21 s (13,8 s bloquées), passant de la
# bande qui paie 44 % à celle qui paie 0 %. Le plafond de 32 soumissions par
# fenêtre reste étanche (budget re-clampé sous _pool_lock).
# 3 -> 6 le 24/08. La file d'envoi est le goulot AVAL : mesure sur 86 envois,
# attente livraison->POST 4,19 s mediane (p75 7,22 s). A 3 voies et un POST de
# ~5 s (le validateur declare total_ms 4919), on ne draine qu'un envoi toutes
# les 1,7 s ; quand 3 sont prets l'attente monte a 9,7 s. Gain simule : +6 %
# d'entrees payees a sprint inchange, et surtout ca ouvre la porte au reste
# (a 6 voies, desactiver le sprint rendrait 3,0 s d'arrivee).
# Garde-fou deja en place (20/08) : le budget est RE-CALCULE sous _pool_lock
# dans _fire_for_window, sinon plusieurs tirs concurrents depassent le quota
# de 32 (mesure : 96 envois pour un plafond de 32).
# Repli : remettre 3.
export RELIQUARY_MAX_INFLIGHT_FIRES=${RELIQUARY_MAX_INFLIGHT_FIRES:-6}
# Poll du cooldown per-env espacé : il doublait le temps d'itération (2 GET
# séquentiels) donc retardait la détection du flip. Il grossit lentement.
export RELIQUARY_COOLDOWN_POLL_S=${RELIQUARY_COOLDOWN_POLL_S:-20}
# Guérison divergence : kernel cascade OFF (16 rollouts même prompt = forme
# de batch que le validateur ne vérifie jamais — cf. audit parité 19/08)
export RELIQUARY_VLLM_DISABLE_CASCADE=${RELIQUARY_VLLM_DISABLE_CASCADE:-0}
export RELIQUARY_BAKE_CHUNK=${RELIQUARY_BAKE_CHUNK:-64}
# TÉLÉCHARGEMENT DU CHECKPOINT (21/08) — poste de perte n°1, mesuré sur une
# nuit : 48,8 min de transfert contre 8,8 min de chargement, soit 85 % du temps
# perdu à chaque avancée de checkpoint (7 par nuit, ~7 min chacune).
# Le dépôt EST stocké en Xet (en-tête x-xet-hash, fichier UNIQUE de 8,04 Go) et
# hf-xet 1.6.0 est déjà installé et déjà utilisé (cache /workspace/hf/xet
# alimenté) — il ne manquait que le mode haute performance, qui parallélise le
# téléchargement par plages sur ce fichier unique.
# Mesuré sur la box : un curl atteint 61,7 Mo/s alors que le transfert du
# checkpoint plafonne à 18 Mo/s. Attendu : 7-8 min -> 2-3 min.
# ⚠️ NE PAS utiliser HF_HUB_ENABLE_HF_TRANSFER : ancienne génération, la lib
# répond « Please use HF_XET_HIGH_PERFORMANCE instead » (constants.py:295).
# RETIRE le 21/08 : pose sur une premisse FAUSSE (telechargement estime a 7 min,
# mesure a 6-30 s sur 8 rechargements). Inerte, ni gain ni nuisance.
# export HF_XET_HIGH_PERFORMANCE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_DEEP_GEMM=0
export VLLM_DEEP_GEMM_WARMUP=skip         # 0.24 enum: skip|full|relax (NOT 0/1)
export VLLM_USE_FLASHINFER_SAMPLER=0      # ptxas PTX 9.2 vs 9.0
export CUDA_HOME=/workspace/venv/lib/python3.12/site-packages/nvidia/cu13
export PATH=/workspace/venv/bin:$CUDA_HOME/bin:$PATH
# ── FIXES COURSE 27/08 (branche fix/course-2026-08-27) ──────────────────────
# HOT_SWAP : ÉTAPE 2, PAS ENCORE ARMÉ (28/08). Le gain est établi (présence
# 0/22 dans les 2 fenêtres post-avancée, 11/11, concurrents à 91-100 %) et le
# gel du 15/08 est couvert (sonde bornée 30 s, verrou à timeout 15 s). MAIS le
# self-gate à plancher 0,80 date de v4 : deux checkpoints consécutifs (1 pas
# d'entraînement) se ressemblent — un échange qui échoue EN SILENCE peut
# passer le gate et produire du SEED_MISMATCH en masse sous les vieux poids.
# Préalable avant de passer à 1 : repasser la gate forced-seed en exigeant des
# token ids IDENTIQUES (pas les planchers), sur la box. D'ici là : préchargement
# seul (le rebuild passe déjà de ~112 à ~55 s), 30 fenêtres mûres, puis étape 2.
export RELIQUARY_HOT_SWAP=${RELIQUARY_HOT_SWAP:-0}
# PREFETCH=1 : le téléchargement HF (57 s médian) domine l'arrêt de 67 s à
# l'avancée ; HF publie 100-350 s avant la bascule (6 avancées mesurées).
# Tâche de fond idempotente, ne touche pas le GPU. Repli : 0.
export RELIQUARY_CHECKPOINT_PREFETCH=${RELIQUARY_CHECKPOINT_PREFETCH:-1}
# HEADROOM=1.0 : 37 % de stale_round depuis le restart (70/188), retry +3,8 s
# médian, 2 essais puis drop. Tolérance arrière du round = ZÉRO ; notre
# lecture→arrivée ≈ 0,5-1,0 s sur cette box. S'il reste <1 s dans le round,
# attendre la frontière et signer le round SUIVANT (arrivée quasi inchangée,
# le round attaché devient le bon). Défaut code 0 = inactif. Repli : 0.
export RELIQUARY_DRAND_MIN_HEADROOM_S=${RELIQUARY_DRAND_MIN_HEADROOM_S:-1.0}
# 30/08 : couvre-feu d'envoi. La deadline no-reveal est FIXE à ouverture+100 s
# (server.py:1416, lu en source) ; chaîne tir→corps p90 ~7 s → 85 laisse 8-13 s
# de marge. Un tir post-seal adaptatif = PRECOMMIT_EXPIRED gratuit (0 point,
# 0 quota). ⚠️ Cette variable avait SAUTÉ au rebuild du 27/08 (3e récidive).
export RELIQUARY_FIRE_CURFEW_S=${RELIQUARY_FIRE_CURFEW_S:-85}

CHECKPOINT="${CHECKPOINT:-Qwen/Qwen3-4B-Base}"

# Sanity : refuse de démarrer si nos constantes ne reflètent pas le contrat.
# Cette garde a DEJA evite un lancement perdu le 24/08 : apres le port v5 du
# prompt, GENERATION_PROFILE_ID etait reste fige sur la valeur v4 — soit 100 %
# de GENERATION_CONTRACT_MISMATCH. Elle compare desormais au contrat LIVE, pas
# a des valeurs ecrites en dur : au prochain cutover elle dira quoi corriger.
/workspace/venv/bin/python - <<'EOF' || exit 1
import json, urllib.request
from reliquary import constants as c

# Invariants qui ne dependent pas de la version du protocole.
assert c.M_ROLLOUTS == 16 and not c.BFT_ENABLED
assert c.MAX_NEW_TOKENS_PROTOCOL_CAP == 8192
assert (c.T_PROTO, c.TOP_P_PROTO, c.TOP_K_PROTO) == (1.0, 1.0, 0)
assert c.MATH_ANSWER_FORMAT == "boxed"
assert c.RAW_COMPLETION_PROMPTS and c.OMI_TRAIN_SHARDS_ONLY
# Coherence interne : le domaine forced-seed suit la version (upstream
# constants.py:1292 -> f"reliquary-forced-seed-v{PROTOCOL_VERSION}").
assert c.FORCED_SEED_DOMAIN == f"reliquary-forced-seed-v{c.PROTOCOL_VERSION}", \
    c.FORCED_SEED_DOMAIN

# Parite avec le validateur LIVE. Si /health est injoignable on NE bloque pas
# (le launcher a deja teste l'egress plus haut) mais on le dit fort.
try:
    h = json.loads(urllib.request.urlopen(
        "http://209.20.157.231:8080/health", timeout=15).read())
except Exception as e:                      # noqa: BLE001
    print(f"[garde] /health injoignable ({e}) — parite NON verifiee")
else:
    ecarts = []
    if h.get("protocol_version") != c.PROTOCOL_VERSION:
        ecarts.append(f"protocole: nous {c.PROTOCOL_VERSION} / eux "
                      f"{h.get('protocol_version')}")
    if h.get("generation_profile_id") != c.GENERATION_PROFILE_ID:
        ecarts.append(f"profil: nous {c.GENERATION_PROFILE_ID} / eux "
                      f"{h.get('generation_profile_id')}")
    if ecarts:
        raise SystemExit("[garde] ECART AVEC LE VALIDATEUR — "
                         + " | ".join(ecarts))
    print(f"[garde] parite OK avec le validateur : {c.GENERATION_PROFILE_ID}")
print("constantes OK:", c.PROTOCOL_VERSION, c.GENERATION_PROFILE_ID)
EOF

cd /workspace/reliquary-miner-priv
exec /workspace/venv/bin/python -m reliquary.cli.main mine \
  --wallet-name camille81-v2 --hotkey hotkey81 --network finney --netuid 81 \
  ${RELIQUARY_VALIDATOR_URL:+--validator-url $RELIQUARY_VALIDATOR_URL} \
  --checkpoint "$CHECKPOINT" \
  --log-level INFO
