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

# ── PROFIL CARTE (19/09, branche h100/v1) ───────────────────────────────────
# Mesuré sur H200 le 14/09 (10 OOM, message torch) : hors vLLM, le processus
# mineur (modèle de preuve + passes de preuve) tient 21-25 Go et la réplique
# 17-20 Go, soit ~44 Go. Sur H200 (139,8 Go) la fraction 0,70 laisse cette
# place ; sur une carte < 100 Go elle la mangerait (0,70 × 79 = 55 Go + 44 =
# OOM à la 1re preuve). Sous 100 Go on calcule donc la fraction vLLM à partir
# de cette réserve : (total − réserve) / total, plancher 0,35.
#   H100 80 Go : (79,2 − 44) / 79,2 ≈ 0,44  →  ~24 Go de KV ≈ 170 k tokens
#   (pic d'un bake 10 ≈ 145 k : juste — surveiller les préemptions vLLM).
# Carte ≥ 100 Go : rien n'est posé, comportement H200 identique à l'octet.
# Une valeur RELIQUARY_VLLM_GPU_FRACTION explicite gagne toujours.
# Repli : RELIQUARY_GPU_PROFILE=off.
_GPU_TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc 0-9)
_GPU_PROFILE_FRACTION=""
if [ "${RELIQUARY_GPU_PROFILE:-auto}" != "off" ] && [ -n "$_GPU_TOTAL_MIB" ] \
   && [ "$_GPU_TOTAL_MIB" -lt 100000 ]; then
  _GPU_PROFILE_FRACTION=$(awk -v t="$_GPU_TOTAL_MIB" -v r="${RELIQUARY_NON_VLLM_RESERVE_GIB:-44}" \
    'BEGIN{g=t/1024; f=(g-r)/g; if(f<0.35)f=0.35; printf "%.2f", f}')
  echo "launch_v4: PROFIL CARTE <100 Go (${_GPU_TOTAL_MIB} MiB) — fraction vLLM par défaut ${_GPU_PROFILE_FRACTION} (réserve ${RELIQUARY_NON_VLLM_RESERVE_GIB:-44} Go hors vLLM)"
else
  echo "launch_v4: profil carte standard (${_GPU_TOTAL_MIB:-?} MiB) — réglages H200 inchangés"
fi

# ── TCP slow-start après inactivité COUPÉ (18/09 06:59, restart A) ──────────
# Posé À LA MAIN sur la box H200 (verdict_tcp.txt), jamais scripté : une box
# neuve repartait sans. Entre deux fenêtres la connexion au validateur est
# inactive ; avec le slow-start le 1er corps (~600 ko) repart à cwnd 10.
# Échec possible dans un conteneur (/proc/sys en lecture seule) : on le DIT,
# on ne bloque pas. Repli : RELIQUARY_TCP_SSAI_OFF=0.
if [ "${RELIQUARY_TCP_SSAI_OFF:-1}" = "1" ]; then
  if sysctl -qw net.ipv4.tcp_slow_start_after_idle=0 2>/dev/null; then
    echo "launch_v4: tcp_slow_start_after_idle=0 appliqué"
  else
    echo "launch_v4: AVERTISSEMENT tcp_slow_start_after_idle non modifiable (valeur $(sysctl -n net.ipv4.tcp_slow_start_after_idle 2>/dev/null || echo ?))" >&2
  fi
fi

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
# 10/09 : le validateur a DÉMÉNAGÉ (209.20.157.231:8080 → 62.238.81.36:8000,
# bascule V1/protocole 6). L'ancienne adresse était EN DUR ici — le mineur a
# tourné à vide toute la nuit. La liste est désormais ordonnée et surchargeable
# (`RELIQUARY_VALIDATOR_CANDIDATES`, séparés par des espaces) : un déménagement
# suivant se règle sans toucher au code.
_VAL_CANDIDATES=${RELIQUARY_VALIDATOR_CANDIDATES:-"http://62.238.81.36:8000 http://209.20.157.231:8080 http://127.0.0.1:8080"}
if [ -z "${RELIQUARY_VALIDATOR_URL:-}" ]; then
  _try=0
  _max=${RELIQUARY_EGRESS_WAIT_TRIES:-240}   # 240 x 15 s = 1 h
  while [ "$_try" -lt "$_max" ]; do
    for _cand in $_VAL_CANDIDATES; do
      _code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 4 \
              "${_cand}/health" 2>/dev/null)
      if [ "$_code" = "200" ]; then
        export RELIQUARY_VALIDATOR_URL="$_cand"
        echo "launch_v4: validateur joignable — $_cand"
        break
      fi
    done
    [ -n "${RELIQUARY_VALIDATOR_URL:-}" ] && break
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
# 08/09 (port v6, revue item 4) : repli sur /workspace/.protocol_version — posé
# par restart_miner.sh quand la version est passée dans l'env — AVANT le défaut
# 5, parce que le watchdog relance SANS l'env du shell. L'env gagne toujours.
_PV_FILE=${RELIQUARY_PROTOCOL_VERSION_FILE:-/workspace/.protocol_version}
export RELIQUARY_PROTOCOL_VERSION=${RELIQUARY_PROTOCOL_VERSION:-$([ -r "$_PV_FILE" ] && tr -dc 0-9 < "$_PV_FILE")}
export RELIQUARY_PROTOCOL_VERSION=${RELIQUARY_PROTOCOL_VERSION:-5}
# v6 (08/09) : mémorise les surcharges utilisateur des réglages que le bloc
# v6 (plus bas) recale — les défauts v5 ci-dessous les consommeraient sinon.
# Variables NON exportées : aucun effet sous v5.
_V6_USER_FIRE_CURFEW_S=${RELIQUARY_FIRE_CURFEW_S:-}
_V6_USER_LATE_BAKE_FROM=${RELIQUARY_LATE_BAKE_FROM:-}
_V6_USER_PREFLIP_GUARD_S=${RELIQUARY_PREFLIP_GUARD_S:-}
_V6_USER_LOCAL_TOKEN_AUTH=${RELIQUARY_LOCAL_TOKEN_AUTH:-}
_V6_USER_GRADE_TIMEOUT_S=${RELIQUARY_GRADE_TIMEOUT_S:-}
_V6_USER_SPRINT_SIZE=${RELIQUARY_SPRINT_SIZE:-}
_V6_USER_HEAD_FIFO=${RELIQUARY_HEAD_FIFO:-}
_V6_USER_MAX_INFLIGHT_FIRES=${RELIQUARY_MAX_INFLIGHT_FIRES:-}
_V6_USER_PREFETCH_POLL_S=${RELIQUARY_CHECKPOINT_PREFETCH_POLL_S:-}
_V6_USER_VLLM_GPU_FRACTION=${RELIQUARY_VLLM_GPU_FRACTION:-}
_V6_USER_GRADE_CONCURRENCY=${RELIQUARY_GRADE_CONCURRENCY:-}
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
# 04/09 : MIN_SCORE 0.23 filtrait sur sigma*(1-mean) alors que l'enchère est
# PLATE (value=1.0 pour tout k en zone) — le chargeur, lui, ne filtrait pas.
# 0 = in_zone seul, cohérent entre amorçage et mise à jour live.
export RELIQUARY_MEMO_MIN_SCORE=${RELIQUARY_MEMO_MIN_SCORE:-0}
# MÉMO DE TÊTE (04/09) : les 2 slots de sprint vont aux ex-payables mesurés de
# la tranche (zone→zone 90 % contre 67-69 % pour un pick classé ; 31 % de nos
# têtes étaient hors zone, 40 % des fenêtres perdues). Tri : run courant
# (fenêtre ≥ RUN_START = début du run basereset-20260825) > confirmations >
# fraîcheur. Repli : HEAD_SLOTS=0 (mémo historique en slot 3) ou MEMO_SLOT=0.
# 08/09 : teste a 3 avec SPRINT_SIZE=3 (13:24-13:55) puis REPLIE a 2 —
# cf. le bloc SPRINT_SIZE ci-dessous. Le memo a bien servi 3 vedettes, ce
# n'est pas lui qui a echoue : c'est la contention GPU du 3e groupe.
# 16/09 : 2 → 5. Diagnostic de la rafale (92 fen) : le 1er bake cuit 10 groupes
# mais n'en amène que 7 au tir (3 jetés en local), et il fait 62 % du revenu
# (4,33 payés/fen, taux 64 % contre 23 % au 2e bake). Sur le journal frais,
# out_of_zone = 30 % pour un pick MÉMO contre 63 % pour un pick classé ⇒ 3 slots
# échangés valent ~+1 groupe valide par bake, qui tire vers 21-24 s (51-62 %
# d'acceptation) ≈ +0,6 payé/fen. Le 08/09 le 3 avait échoué À CAUSE du sprint 3
# (contention GPU à 48 séquences), pas du mémo — ici le bake ne change pas.
# JUGER : out_of_zone du 1er bake (réf 3,4/bake), groupes du 1er bake qui
# atteignent le tir (réf 7, p25 5), puis payés/fen (réf 7,02) sur 30 fenêtres.
# VIGIE : réserve du mémo (137 ex-payables/tranche, min 60) et
# same_prompt_superseded (course forced-seed sur les prompts mémo, réf 4,4 %).
# REPLI : 2.
# ⛔ 16/09 : 5 TESTÉ (fen 46076-46086) et REPLIÉ à 2. Le mémo à 5 slots a
# retardé toute la rafale (+0,9 s sur le groupe 1, +2,5 s sur le 10e : les
# ex-payables génèrent +19 % de traînard) ; le tri par volume (MEMO_SORT=short)
# a bien ramené le 1er tir à 12,4-12,8 s mais les payés sont restés à 4,4/fen
# (méd 4) contre 5,7 sous mémo 2 et 8,50 dans l'ère validée du 15/09, marché
# INCHANGÉ (336 payés, 40-43 hotkeys, 1er mineur 16-23). Repli complet.
# 20/09 (H100, bake 8) : 2 -> 3. Mesuré sur 65 bakes de tête (fen 46433+) :
# hors zone d'un pick MÉMO 12,3 % (16/130) contre 50,5 % (197/390) pour un pick
# CLASSÉ (prior v5.9) ; acceptation 39,3 % contre 32,6 %. Un slot échangé vaut
# donc ~+0,38 groupe valide au bake de tête, payé à 94-100 % → ~+0,35 payé/fen.
# Les deux risques qui avaient fait replier le 5 du 16/09 ont FONDU : traînard
# des picks mémo +6 % (830 tok contre 785) au lieu de +19 %, et le groupe 1 sort
# à 7,1 s quand le 1er livré est un pick mémo contre 7,3 s sinon (aucun retard).
# Doublons : 0 content_in_cooldown, 0 hash_duplicate, 0 same_prompt_superseded
# sur 1 269 tirs mémo. Réserve : 96 007 payables connus.
# 3 et pas 4 : le test replié du 16/09 mettait 5 slots sur un bake de 10, soit
# la MOITIÉ du bake — 4 sur 8 y reviendrait.
# JUGER (8-10 fen) : hors zone du bake 1 (réf 3,3 sur 8), groupe 1 prêt
# (réf 7,2 s), tirs < 18 s/fen (réf 4,0-4,2), traînard des picks mémo (réf 830).
# REPLI à 2 si : groupe 1 > 7,5 s, OU tirs < 18 s sous 4,0, OU hors zone du
# bake 1 qui ne descend pas sous 3,0, OU hash_duplicate/content_in_cooldown
# au-delà de 0,5 % des tirs mémo.
# ⚠️ ANGLE MORT : /workspace/burned_idx.npy (veto anti-cooldown de contenu) est
# FIGÉ depuis le 10/09 (archive de l'ère v5) — à re-générer si les rejets de
# contenu réapparaissent.
# 20/09 : 3 -> 5 slots, mais RÉSERVÉS AU 1er BAKE (LATE=0).
# La réserve mémo FRAÎCHE (âge <=250 fen) ne vaut que ~19 payables par tranche
# alors qu'on consommait 39-45 picks mémo par fenêtre (3 slots x 13-15 bakes) :
# le bon mémo partait sur des bakes tardifs qui ne paient rien, puis on tapait
# dans les profondeurs 25-40 où les jetés remontent à 52,7 %. Concentré sur le
# 1er bake, on consomme 5 picks/fenêtre au lieu de 39-45 — enfin soutenable.
# Jetés cumulés par profondeur servie : 3 slots 13,8 % · 5 slots 13,2 % (donc
# GRATUIT) · 8 slots 17,7 %. Un slot CLASSÉ est jeté à 40,1 %.
# Mesuré aussi : les picks mémo sont 20 % plus COURTS que les classés (995
# contre 1 245 tok) et livrés plus tôt (p90 15 s contre 36 s) => la rafale ne
# s'allonge pas.
# JUGER sur la comparaison APPARIÉE INTRA-BAKE (jetés des slots mémo contre
# ceux des slots classés de la MÊME fenêtre) : ~14 fenêtres suffisent, et ça
# élimine le marché. ⛔ NE PAS juger sur les payés/fenêtre : leur écart-type
# est 3,16, il faudrait 627 fenêtres par bras (120 h) pour voir +0,5.
# REPLI : RELIQUARY_MEMO_HEAD_SLOTS=3 et vider MEMO_HEAD_SLOTS_LATE.
export RELIQUARY_MEMO_HEAD_SLOTS=${RELIQUARY_MEMO_HEAD_SLOTS:-5}
export RELIQUARY_MEMO_HEAD_SLOTS_LATE=${RELIQUARY_MEMO_HEAD_SLOTS_LATE:-0}
export RELIQUARY_MEMO_RUN_START=${RELIQUARY_MEMO_RUN_START:-32791}
# SOULÈVEMENT DE LA BANDE DU 1er BAKE (20/09, commit 45f6a84).
# Mesuré en vol sur 16 fenêtres (comparaison appariée intra-bake) : les slots
# CLASSÉS du 1er bake sont jetés à 46,2 %, contre 12,5 % pour les slots mémo.
# Les 46,2 % dépassent les 40,1 % de la moyenne des classés parce qu'en
# réservant 5 slots au mémo on a concentré les classés restants sur le SOMMET
# du classement — la zone pourrie par l'ÂGE (rang 1-10 : 51,4 % de jetés,
# 14,7 % de prompts frais, âge médian 7 160 fenêtres). La bande 51-250 vaut
# ~30,5 %. On sert donc les slots classés du 1er bake depuis le rang 51, et la
# tête revient JUSTE DERRIÈRE pour les bakes tardifs (qui ne paient quasi rien).
# La consommation totale de la fenêtre est INCHANGÉE.
# JUGER sur le taux de jetés des slots CLASSÉS du 1er bake (46,2 % -> ~30,5 %),
# apparié intra-bake, ~16 fenêtres. Vigie : groupe 1 prêt (7,7 s, replier >9 s).
# ⛔ NE PAS mettre un décalage GLOBAL : mesuré à +2,2 groupes jetés/fenêtre.
# REPLI : RELIQUARY_RANK_LIFT=0.
# ⛔ 21/09 : REPLIÉ À 0 après 47 fenêtres (46555-46601, vérité R2).
# Mécanisme VALIDÉ (slots classés du 1er bake 46,2 % -> 29,8 % de jetés, hors
# zone du 1er bake 2,94 -> 2,24) MAIS payés NON améliorés : 12,05 +/- 1,50 ->
# 11,06 +/- 0,91. Avant 22 s : 6,10 -> 6,20 payés (inchangé) ; 22-60 s :
# 5,15 -> 4,09 (-1,06), parce que les bakes 2-4 héritent du sommet vieilli du
# classement et tirent 3 groupes de moins (13,8 -> 10,8). La prémisse « les
# bakes tardifs ne paient quasi rien » était FAUSSE : la tranche 22-60 s fait
# ~40 % de notre revenu. Le soulèvement est donc à somme ~nulle.
# Le code reste (inerte à 0). La vraie cause — le sommet vieilli servi à
# quelqu'un quoi qu'on fasse — est à traiter autrement (cf. passation).
export RELIQUARY_RANK_LIFT=${RELIQUARY_RANK_LIFT:-0}
# TRI DU MÉMO PAR VOLUME (16/09, commit cf542d0) : sous V1 le paiement est
# `fill_closed_fixed_group` — un groupe payé rapporte pareil quelle que soit sa
# longueur, alors que le tri historique (fraîcheur) date de v5 où le bucket
# valait volume/round. Mesuré : un pick mémo génère 858 tokens de traînard
# contre 723 pour un pick classé (+19 %), et passer de 2 à 5 slots a retardé
# TOUTE la rafale de +0,9 s (groupe 1) à +2,5 s (groupe 10) — or l'acceptation
# vaut 100 % avant 18 s, 75 % à 18-20 et 62 % à 20-22. À validité égale (run
# courant, confirmations), on départage donc par le volume CROISSANT au-dessus
# du plancher. Sur 149 689 réapparitions : bande 4 000-6 000 = re-zone
# 83,7-85,4 % avec un traînard 587-672 (contre 1 019 au-delà de 10 000) ; sous
# 3 000 la re-zone tombe à 78,5 % et le rollout le plus court à 30-46 tokens
# (zone à risque CHALLENGE_K), d'où MIN_VOL=4000.
# JUGER : groupe 1 prêt (réf mémo-2 5,5 s ; mémo-5/fraîcheur 6,4 s) et traînard
# des picks mémo (réf 858). GARDE-FOU : si le groupe 1 n'est pas repassé sous
# ~5,8 s en 6 fenêtres, replier MEMO_HEAD_SLOTS à 2.
# REPLI : RELIQUARY_MEMO_SORT=fresh (chemin byte-identique).
export RELIQUARY_MEMO_SORT=${RELIQUARY_MEMO_SORT:-fresh}
export RELIQUARY_MEMO_MIN_VOL=${RELIQUARY_MEMO_MIN_VOL:-4000}
# VETO DES INDEX BRÛLÉS PAR CONTENU (04/09) : jumeaux de texte de prompts déjà
# sélectionnés (cooldown validateur à vie, invisible à /state) — 8 % de nos
# candidats mouraient content_in_cooldown. Fichier régénéré toutes les 15 min
# depuis la dev box (instantané R2 + digests), relu au changement de mtime.
export RELIQUARY_BURNED_IDX=${RELIQUARY_BURNED_IDX:-/workspace/burned_idx.npy}
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
# 17/09 : mesures pures (fichiers séparés, lus par aucun moniteur ni par le mémo)
export RELIQUARY_FLIP_DUMP=${RELIQUARY_FLIP_DUMP:-/workspace/flip_v4.jsonl}
export RELIQUARY_OOZ_DUMP=${RELIQUARY_OOZ_DUMP:-/workspace/ooz_v4.jsonl}
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
# ── 5 → 10 le 12/09 (V1 / fill-closed) ───────────────────────────────────────
# POURQUOI : sous fill-closed le paiement est le NOMBRE de groupes retenus — le
# round de la tête ne paie plus rien. La prime à la tête qui justifiait un batch
# étroit (et le sprint, et HEAD_FIFO) est MORTE avec l'ancienne économie.
# MESURÉ le 12/09 sur 9 fenêtres (45778-45786) : rétention 100 % pour une
# arrivée entre 30 et 89 s, 0 % au-delà de 150 s ; on place 4,5 groupes/fenêtre
# quand le meneur du lane code en place 17,3.
# Le goulot n'est PAS : le KV (10 % utilisé, 0 préemption, ~830 séquences
# tiendraient), ni l'ordonnancement (0,1 s de temps mort entre bakes), ni la
# chaîne d'envoi (2,4 s), ni le forced-seed batché (déjà actif). On ne donnait
# au GPU que 80 séquences (5×16) — 100 % d'occupation mais 443 W sur 700.
# 10 → 160 séquences, entre les deux points de banc ci-dessus (128 : 12,5k tok/s,
# 256 : 16k). MAX_NUM_SEQS=256 couvre déjà 16 prompts.
# ⚠️ ÉCART NON RÉSOLU : à 80 séquences le banc prédit ~9-10k tok/s, on mesure
# 2 489. Le banc était à 1 024 tok/rollout, nous sommes à 464 (médiane) : moins
# d'amortissement du prefill. Ce test le tranche en partie.
# CRITÈRE — 10 fenêtres mûres, sur les groupes RETENUS dans R2, PAS les acceptés
# (35-47 acceptés pour 4-5 retenus : l'admission ne veut rien dire ici).
#   référence avant : 4,5 groupes retenus/fenêtre, bake p50 16,3 s.
# VIGIE : `quota=N/64` (34/64 avant) — s'il sature, monter
# MAX_SUBMISSIONS_PER_WINDOW à 128 (le protocole en autorise 448, prouvé en vol
# le 12/09 : 35 et 39 acceptés > 32, zéro rate_limited).
# REPLI : remettre 5 (une variable) + restart.
# ⛔ 15/09 soir : 14 TESTÉ (38 fen, 21h-05h) puis REPLIÉ à 10 le 16/09.
# 224 séquences saturent le GPU : preuve 2,83 → 4,93 s, vérif EOS 2,92 → 5,21 s,
# prêt → précommit 3,59 → 5,74 s, tir médian 18,9 → 22,2 s — or l'admission
# tombe à 30 % dans la tranche 21-25 s (83 % à 18-21 s, 100 % avant 18 s).
# Résultat R2 : payés code < 25 s 4,27 → 3,42/fen, ratio à la médiane top-8
# 0,68 → 0,50 alors que le marché était PLUS favorable (39 mineurs payés au
# lieu de 47). Ne pas remonter le bake sans un 2e GPU pour preuve/réplique.
# ── 19/09 (H100) : 10 → 8. Base H100 à 10 (fen 46397-46418, config H200
# inchangée) : ms/token 11,8 contre 9,9 sur H200, groupe 1 prêt 8,1 s contre
# 6,7, tir g1-3 14,8 s contre 12,8 ; 0,7 groupe/fen passe d'« avant 18 s » à
# 18-25 s. 160 séquences sur une carte plus lente retardent la TÊTE ; à 128
# (8 × 16) la tête doit sortir plus tôt, au prix de 2 groupes par bake.
# JUGER (mécanique, 8-10 fen, hors fenêtre froide) : groupe 1 prêt (réf 8,1 s),
# tir g1-3 (réf 14,8 s), tirs < 18 s/fen (réf 4,13 au 1er bake), ms/token.
# REPLI si groupe 1 prêt ne passe pas sous ~7,5 s OU tirs < 18 s en baisse.
# Payés : 30 fen, en tenant compte du marché (56e place R2, réf 15,6 s).
# ✅ VERDICT 19/09 16:52 (11 fen 46421-46431 contre 23 à 10) : GARDÉ. Groupe 1
# prêt 8,1 → 6,9 s, ms/token 11,8 → 10,5, tir g1-3 14,7 → 14,2 s, tirs < 18 s
# 4,0 → 4,0 (stable), tirs < 17 s (R2) 2,80 → 3,27. Payés 5,50 → 5,82 NON
# démontré (marché plus facile : 56e place 15,8 → 16,6 s, n=11).
# 20/09 : 8 -> 10. Le rejet du 10 (19/09) datait d'AVANT les deux fixes de
# timing : depuis, on a récupéré ~1,1 s au démarrage (MS_FLIP_BAKE) et le bake 2
# part à 15,6 s au lieu de 15,9 (BAKE_WAIT_GRADES=0). Mesuré sur 75 fen : le
# bake 1 ne tire que 4,6 groupes sur 8 (3,3 hors zone) et TOUT ce qu'il tire
# avant 16 s est payé à 98 % — la zone payante (8-18 s) est à moitié vide,
# le bake 2 n'y arrive qu'à 18,5 s (35 % payés).
# Attendu : 10 bakés -> ~5,9 valides tirant 10-18 s, soit ~+1 payé/fen.
# JUGER (8-10 fen) : tirs du bake 1 avant 18 s (réf 4,2), groupe 1 prêt
# (réf 7,2 s), tir g1-3 (réf 12,1-12,5 s), payés (réf 7,0-7,4).
# REPLI à 8 si les tirs avant 18 s baissent OU si le groupe 1 dépasse 8,5 s.
export RELIQUARY_BAKE_BATCH_SIZE=${RELIQUARY_BAKE_BATCH_SIZE:-10}
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
# ⛔ 08/09 : SPRINT 3 (+ MEMO_HEAD_SLOTS=3) TESTE 13:24 -> REPLIE 13:55.
# RESULTAT NET, 7 fenetres : g1 pret p50 3,30 -> 4,50-4,90 s (+1,2 a +1,6 s,
# 4,4 ecarts types), sprint livre 4,50 -> 6,60 s. Le seuil de repli fixe
# d'avance etait 4,3 s. Point mort de la courbe = 1,0 s de retard, donc on
# etait deja du cote perdant (-0,02 a -0,16 payee/fen).
# ⇒ CONFIRME le rejet du 02/09 (+1,1 s) sur une config entierement nouvelle.
# La cause est la CONTENTION GPU (48 sequences au lieu de 32), PAS la qualite
# du 3e pick (le memo a bien servi 3 vedettes) NI le CUDA graph (mesure :
# le graphe ne vaut que 2 % a n=48). ⛔ NE PLUS RE-TESTER sans une carte
# supplementaire : sur CETTE carte une 3e tete precoce coute plus qu'elle
# ne rapporte, c'est le 3e echec (sprint 3 le 02/09, scan_holdoff le 03/09).
export RELIQUARY_SPRINT_SIZE=${RELIQUARY_SPRINT_SIZE:-2}
export RELIQUARY_SCAN_HOLDOFF_S=${RELIQUARY_SCAN_HOLDOFF_S:-0}
# HEAD_FIFO : optimisation de l'ancienne économie (priorité d'arrivée des
# têtes), jamais mesurée en prod — repli 0 pour la relance, A/B plus tard.
export RELIQUARY_HEAD_FIFO=${RELIQUARY_HEAD_FIFO:-2}
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
# miroir parquet absent sur box neuve (01/09) — HF direct en attendant sa reconstruction
export RELIQUARY_PARQUET_LOCAL_ROOT=${RELIQUARY_PARQUET_LOCAL_ROOT:-/workspace/parquet_mirror}
# FS_GRAPH armé 01/09 après gates PASS (bit-exact + token ids identiques, /tmp/gates_fs*.log)
export RELIQUARY_FS_GRAPH=1
export RELIQUARY_PARQUET_EXPECTED_LEN=${RELIQUARY_PARQUET_EXPECTED_LEN:-2481806}
#
# 2) TABLE DE SCORES PRÉ-CALCULÉE — supprime les 0,91 s de notation ET les
#    lectures de prompts. Générer avec :
#      python3 scripts/precompute_prompt_scores.py --out /workspace/prompt_scores.npz
#    La table porte une empreinte des 3 modèles : un prior ré-entraîné la
#    périme et le mineur retombe SEUL sur la notation en direct.
#    Vide => notation en direct, inchangée.
# 01/09 18h : PRIOR UNIQUE — score=(1-P(sigma0))^8×s59 (risk/vol à zéro dans la
# table, λ/μ inertes). Empreinte corrigée (le bake avait tourné sans l'env
# risk → trio différent). Repli : prompt_scores_zone_v1.npz + restart.
# 14/09 soir : PRIOR V1 « EN ZONE » — score = P(en zone) d'un TF-IDF+logistique
# entraîné sur l'ère V1 (4 504 groupes code, fen 45650-45919 ; hors zone des
# logs après 45896, le dump ne les écrit plus). Ancienne table : AUC 0,508 sur
# l'ère récente ; nouvelle : 0,755 (temporel), top 30 % en zone 93,9 % contre
# 78,2 %. Même empreinte (risk/volume/fingerprint repris). ops/prior_v1/.
# Repli : RELIQUARY_PROMPT_SCORES=/workspace/prompt_scores_unique_v1.npz + restart.
# 16/09 soir : TABLE « v5.9 SEUL » (score = prompt_predictor.score_prompt(v5.9), même
# empreinte). Éval 9 257 groupes post-46012 (prompts jamais vus) : AUC en zone
# inzone_v1 0,441 (sous le hasard dans ses propres picks) · candidat 0,564 ·
# v5.9 0,608 ; en zone top10/22 59,9 / 68,9 / 71,6 %. Réserve : inzone_v1 est
# pénalisé par la restriction de plage (les groupes sont ses picks). Verdict en vol
# sur out_of_zone/bake. Repli : RELIQUARY_PROMPT_SCORES=/workspace/prompt_scores_inzone_v1.npz + restart.
# 18/09 (restart F) : TABLE « EN ZONE V1c » (TF-IDF+logistique, 24 042 groupes :
# ère inzone_v1 + V1b + ère v5.9 46183-46266, négatifs de ooz_v4.jsonl). Test hors
# sélection (2e moitié de l'ère v5.9, prompts jamais vus) : AUC en zone 0,655 contre
# 0,515 pour v5.9 ; top10/22 76,3 % contre 67,1 %. Simulation 1er bake (41 fen) :
# valides en tête 2,05 -> 2,59, bake 6,44 -> 8,68 ; fenêtres à tête mauvaise 11/41
# -> 1/11. VIGIE : hors zone parmi les 3 premiers prêts de chaque bake (réf ~32 %).
# REPLI : RELIQUARY_PROMPT_SCORES=/workspace/prompt_scores_v59seul_v1.npz + restart.
# 18/09 : + PÉNALITÉ DE COLLISION — score = P(en zone, V1c) × (1 − P(pris par un autre
# avant 16 s))^0,5, modèle appris sur R2 (67 020 prompts, AUC test 0,754). Test sur
# tranches réelles (1 500 prompts/fen, 40 fen, modèles coupés à 46225) : top-50 en
# zone 83,5 % (v5.9 65,3 %, V1c seul 84,8 %) ; pris par un autre avant 16 s 2,8 %
# (v5.9 4,6 %, V1c seul 17,8 %). Table : ops/prior_v1c/bake_combo.py.
# ⛔ 18/09 10h : REPLIÉ. En vol (6 fen, 800 groupes) : hors zone 34,3 % -> 47,4 %,
# hors zone parmi les 3 premiers prêts 1,21 -> 1,74. Le test hors ligne (84 % en zone)
# ne comptait que les prompts DÉJÀ étiquetés ; V1c choisit surtout des prompts jamais
# générés, loin de son entraînement, et s'y trompe. Tester un prior = générer des
# prompts inconnus (bakes fantômes), pas relire des étiquettes existantes.
export RELIQUARY_PROMPT_SCORES=${RELIQUARY_PROMPT_SCORES:-/workspace/prompt_scores_v59seul_v1.npz}
# Mode course 2026-08-19 : garde pré-flip (GPU libre au flip) + rafale 8
# 30/08 : fenêtres médianes 102 s (p10 87), seals anticipés 72-82 s.
# lf<gf OBLIGATOIRE (l'inverse rend la zone capped inatteignable).
# ⚠️ PREMIER A/B de la relance : guard 50 (bake continu) vs 10 (mode course
# qui faisait 16,5 payées/h sous l'ANCIENNE clé — inconnu sous la nouvelle).
export RELIQUARY_LATE_BAKE_FROM=${RELIQUARY_LATE_BAKE_FROM:-35}
export RELIQUARY_PREFLIP_GUARD_S=${RELIQUARY_PREFLIP_GUARD_S:-50}
export RELIQUARY_LATE_BAKE_CAP=${RELIQUARY_LATE_BAKE_CAP:-1200}
# Streaming C 19/08 : preuve spéculative des têtes de rafale (parallèle au grading)
# 31/08 : SPEC_PROOF=1 — la prémisse du rejet est morte : le grading coûte
# désormais 1,02 s (GRADE_TIMEOUT_S=1.0, ~22 % des rollouts le mangent plein).
# Recouvrement grade CPU ∥ preuve GPU mesuré −0,72 s/tête (agents 1+2, 31/08) :
# jumeau 1 ~8,0 s = round 2 (frontière réelle 8,4 s). 4 slots/fenêtre.
# (historique 27/08 : SPEC_PROOF remis à 0 — grading 0,06 s à l'époque, rien à
# paralléliser) ; le flip à 1 du 27/08 21:20 était une variable non contrôlée.
export RELIQUARY_SPEC_PROOF=${RELIQUARY_SPEC_PROOF:-1}
export RELIQUARY_SPEC_PROOF_SLOTS=${RELIQUARY_SPEC_PROOF_SLOTS:-4}
export RELIQUARY_STALE_FAST_REFIRE=${RELIQUARY_STALE_FAST_REFIRE:-1}

# 31/08 : le cache de compilation vLLM a sauté au rebuild pour la 4e fois —
# sans lui, CHAQUE avancée de checkpoint repaie ~22 s de torch.compile
# (69 répertoires accumulés observés). Clé indépendante du checkpoint.
export RELIQUARY_VLLM_COMPILE_CACHE_DIR=${RELIQUARY_VLLM_COMPILE_CACHE_DIR:-/workspace/vllm_compile}
export RELIQUARY_CHECKPOINT_PREFETCH_POLL_S=${RELIQUARY_CHECKPOINT_PREFETCH_POLL_S:-15}
# AUTO-FILTRAGE 19/08 (rapport agents) : miroir local des checks validateur,
# posé APRÈS le bloc unset des seuils v3 plus haut — marges sûres v4.
export RELIQUARY_MIN_LOCAL_Q10=${RELIQUARY_MIN_LOCAL_Q10:-0.0005}
export RELIQUARY_MIN_LOCAL_MEDIAN=${RELIQUARY_MIN_LOCAL_MEDIAN:-0.08}
# 07/09 : gate douce OFF. Contre-épreuve R2 (forced-seed = mêmes tokens pour
# tous) : 47/47 groupes écartés par ce miroir ont été ADMIS par le validateur
# chez d'autres mineurs (12 payés), 0 token_tampered ; le n°1 du marché n'a
# aucun filtre local (0,13 token_tampered/fen). Coût du miroir : 14-17 % des
# groupes, 5-7 % des têtes. Ombre journalisée : pre_bake[shadow_token_auth].
# Vigie : token_tampered ≤ 0,15/fen. Repli : remettre 1 + restart.
# 08/09 : VERDICT sur 838 fen (gate OFF) vs 537 (gate ON) : NEUTRE, non prouvé —
# entrées ≤9,6 s hors ombres 1,15 = 1,15, ombres admises 282 / payées 2, Δ ratio
# top-8 +0,022 IC95 [−0,036 ; +0,079]. Remis à 1 (règle : non concluant = retour).
export RELIQUARY_LOCAL_TOKEN_AUTH=${RELIQUARY_LOCAL_TOKEN_AUTH:-1}
export RELIQUARY_LTA_CHOSEN_MAX=${RELIQUARY_LTA_CHOSEN_MAX:-1e-5}
export RELIQUARY_LTA_ARGMAX_MIN=${RELIQUARY_LTA_ARGMAX_MIN:-0.99}
# 25/08 22h — regime ckpt 660 (modele de base : ecrit 2,4x plus long, k=16 disparu).
# Le filtre dur passait de 3,8 % a 33 % de la production. Leur seuil REEL est
# TOKEN_AUTH_THRESHOLD=1e-8 (constants.py:1309), applique SANS condition
# d'argmax. On garde une marge x3 au lieu de x10. Repli : 1e-7.
export RELIQUARY_LTA_HARD_MIN=${RELIQUARY_LTA_HARD_MIN:-1e-8}

# 30/08 — plafond de grading (PERDU au rebuild, 4e recidive). Distribution
# BIMODALE mesuree sur 3 718 groupes : 76 % < 0,2 s, 22 % a 5,0 s pile (un
# rollout qui boucle), 1,26 % entre les deux. Sur la 1re entree : 15,2 % en
# souffrent, arrivee 9,1 -> 16,7 s. SANS risque de conformite (le validateur
# ECRASE notre reward). Vigie : taux out_of_zone des verdicts. Repli : 5.
export RELIQUARY_GRADE_TIMEOUT_S=${RELIQUARY_GRADE_TIMEOUT_S:-1.0}
# Malus anti-rollout-court (20/08) : dé-priorise à la SÉLECTION les prompts
# qui produisent des rollouts <32 tok (inéligibles CHALLENGE_K, 0 payé/333).
export RELIQUARY_SHORT_RISK_MODEL=${RELIQUARY_SHORT_RISK_MODEL:-/workspace/risk_zone_v1.json}
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
export RELIQUARY_SZ_BLACKLIST=${RELIQUARY_SZ_BLACKLIST:-1}
# 21/09 : un groupe HORS ZONE (autre que « tout réussi ») n'était écarté que
# 300 fenêtres (~2 jours, défaut du code), sur la prémisse qu'un tout-raté peut
# redevenir en zone. FAUX : sur 8 134 picks classés (67 fenêtres), un prompt dont
# la dernière mesure était hors zone est re-jeté à 60 % (90 % s'il était tout
# raté), contre 20,9 % s'il était en zone et 38,6 % s'il n'a jamais été mesuré.
# Ces revenants font 9,7 picks/fenêtre, dont 1,9 des 5 slots classés du 1er bake
# (ses 47 % de jetés). Gain estimé ~+0,9 payé/fen (+0,6 à +1,3). Fichier
# /workspace/sz_blacklist.json ré-amorcé le même jour (sauvegarde
# sz_blacklist.json.avant-dur20000-*).
# JUGER : jetés des slots classés du 1er bake (réf 47 %, attendu ~26 %).
# REPLI : RELIQUARY_SZ_BLACKLIST_DUR_FEN=300 + restaurer la sauvegarde.
export RELIQUARY_SZ_BLACKLIST_DUR_FEN=${RELIQUARY_SZ_BLACKLIST_DUR_FEN:-20000}
# 21/09 : BAKES FANTÔMES (spec docs/superpowers/specs/2026-09-21-bakes-fantomes-design.md).
# Le trou 503 (~416 s/cycle où le GPU dort) sert à étiqueter des prompts de la
# bande p95-p99 du prior avec une randomness SYNTHÉTIQUE (jamais soumettable).
# Sécurité : lot seulement 330-537 s après l'ouverture (cycle jamais < 627 s
# sur 349 fenêtres) ET interrompu au flip (should_abort à chaque pas moteur).
# FEED=1 : en zone -> mémo ; hors zone -> liste noire. Gain estimé +0,8 à +1,1
# payé/fen, qui monte à mesure que les étiquettes s'accumulent (heures/jours).
# VIGIE IMMÉDIATE (non-perturbation) : le 1er bake de chaque fenêtre doit
# toujours démarrer à +1,2-1,5 s et le groupe 1 rester vers 7,7 s.
# REPLI : RELIQUARY_GHOST_BAKE=0 (arrêt total) ou RELIQUARY_GHOST_FEED=0.
export RELIQUARY_GHOST_BAKE=${RELIQUARY_GHOST_BAKE:-1}
export RELIQUARY_GHOST_FEED=${RELIQUARY_GHOST_FEED:-1}
export RELIQUARY_GHOST_DUMP=${RELIQUARY_GHOST_DUMP:-/workspace/ghost_v4.jsonl}
# 21/09 soir : débit ×3-4. 1er déploiement (lot 8, T_MIN 330) = 1 173 prompts
# étiquetés en 4 h 45 pour ~100 000 dans la bande -> 12 picks réels sur 3 199
# portaient une étiquette : effet nul par dilution, 0 perturbation mesurée
# (1er bake 1,25 s, groupe 1 7,40 s). Lot 16 x 16 rollouts = 256 séquences =
# MAX_NUM_SEQS exact ; T_MIN 280 (plus rien n'est payé après 125 s).
# REPLI : GHOST_LOT=8, GHOST_T_MIN=330.
export RELIQUARY_GHOST_LOT=${RELIQUARY_GHOST_LOT:-16}
export RELIQUARY_GHOST_T_MIN=${RELIQUARY_GHOST_T_MIN:-280}
export RELIQUARY_TIMEOUT_IMPUTE=${RELIQUARY_TIMEOUT_IMPUTE:-1}
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
# 06/09 : 6 -> 3. Audit (4 agents) : stale_round 25 % quand un de NOS corps
# (610 ko, 0,96 s) est en cours d'upload a la signature, 13 % sinon ; les
# entrees 4-6 paient ~2 %. Limiter les tirs en vol protege le precommit des
# tetes de la bande passante des corps. Repli : 6 + restart.
export RELIQUARY_MAX_INFLIGHT_FIRES=${RELIQUARY_MAX_INFLIGHT_FIRES:-3}
# Poll du cooldown per-env espacé : il doublait le temps d'itération (2 GET
# séquentiels) donc retardait la détection du flip. Il grossit lentement.
export RELIQUARY_COOLDOWN_POLL_S=${RELIQUARY_COOLDOWN_POLL_S:-20}
# Guérison divergence : kernel cascade OFF (16 rollouts même prompt = forme
# de batch que le validateur ne vérifie jamais — cf. audit parité 19/08)
export RELIQUARY_VLLM_DISABLE_CASCADE=${RELIQUARY_VLLM_DISABLE_CASCADE:-0}
# BAKE_CHUNK purgé 01/09 : ne vit que dans le pipeline de repli _bake_streaming (mort, engine.py:4089 return avant)
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
# 08/09 : le PRÉCHARGEMENT du checkpoint (7,5 Go en 57 s ≈ 1,1 Gbit/s, hf-xet
# concurrence adaptative) sature le lien : le poll /state (1,5 Mo, timeout 3 s)
# expire en boucle (DEBUG, invisible) jusqu'à la fin du téléchargement → flip
# détecté 22,5 s p50 en retard (max 56) sur les fenêtres ouvertes pendant un
# téléchargement, 37/jour, payées 0,27 contre 1,11 = 3,4 % du revenu (log nuit
# 07→08/09 + R2 44005-44844). Fix : UNE connexion de téléchargement → part
# équitable TCP pour /state. Repli : retirer cette ligne + restart. Vigie :
# retard de flip pendant téléchargement (réf 22,5 s), payées de ces fenêtres
# (réf 0,27), durée de téléchargement (doit rester < 137 s = avance mini).
# ⛔ 08/09 : `HF_XET_FIXED_DOWNLOAD_CONCURRENCY=2` DÉPLOYÉ 08:45 puis REPLIÉ
# 11:25 — RÉGRESSION mesurée. Le banc (97 s) était FAUX : hf-xet déduplique les
# chunks contre le cache, donc un banc sur une révision proche sous-estime
# massivement. En production : 310-364 s (commit HF → « préchargement OK »)
# contre 57-62 s au défaut. Le téléchargement finissait alors 26-32 s APRÈS
# l'ouverture de la fenêtre de bascule → DEUX fenêtres perdues par
# rechargement au lieu d'une (fenêtre +0 : 1re admise 72-84 s et 0 payée
# contre 9,2 s / 1,35 payée ; global 0,95 payée/fen contre 1,05, ratio 0,57
# contre 0,67 sur 56 fen). NE PAS re-brider sans mesurer commit HF → OK en prod.
# Timeout du poll /state (défaut code 3 s) : à 3 s le poll expirait en boucle
# pendant le téléchargement (échecs en DEBUG, désormais WARNING 1/5 s).
export RELIQUARY_STATE_POLL_TIMEOUT_S=${RELIQUARY_STATE_POLL_TIMEOUT_S:-10}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_DEEP_GEMM=0
export VLLM_DEEP_GEMM_WARMUP=skip         # 0.24 enum: skip|full|relax (NOT 0/1)
export VLLM_USE_FLASHINFER_SAMPLER=0      # ptxas PTX 9.2 vs 9.0
export CUDA_HOME=/workspace/venv/lib/python3.12/site-packages/nvidia/cu13
export PATH=/workspace/venv/bin:$CUDA_HOME/bin:$PATH
# ── FIXES COURSE 27/08 (branche fix/course-2026-08-27) ──────────────────────
# ── QUOTA DE SOUMISSIONS (10/09) ────────────────────────────────────────────
# ⚡ On se bridait à 32/fenêtre alors que le validateur en autorise 512 sous
# fill-closed. Mesuré en vol le 10/09 : 50-70 groupes générés par fenêtre,
# 46-59 PAYABLES, 32 envoyés ⇒ 14-27 groupes payables JETÉS par fenêtre,
# pendant que /state affichait admitted[opencodeinstruct]=91 sur un budget
# d'admission de 512 (math 436/512 : le code est l'env le moins disputé).
# Source de vérité : image live `6b17632` == main `7468825`, et docs/mining.md
# (#241, mergé le 10/09) — « 512 attempts with V6 fill-closed enabled ».
# ⚠️ PALIER, PAS LE PLAFOND : chaque envoi porte une preuve GRAIL sur la même
# carte que la génération, et la preuve est une FILE (4 concurrentes → 5,76 s
# p50). Passer de 32 à 512 d'un coup retarderait les têtes, qui font
# l'essentiel du revenu. On monte par paliers, 30-40 fenêtres mûres chacun.
# Vigie du palier : `groupe 1/n prêt à Xs` (réf g1 ~3,8 s) et la durée de
# preuve ; replier si la tête recule. Repli = remettre 32 (UNE variable).
export RELIQUARY_MAX_SUBMISSIONS_PER_WINDOW=${RELIQUARY_MAX_SUBMISSIONS_PER_WINDOW:-64}
# HOT_SWAP — 3 valeurs : 0 (défaut) | shadow | 1. Toute autre valeur = 0.
# ⚡ V1/fill-closed a CHANGÉ L'ENJEU (mesuré 10/09, 11 avancées) : le
# rechargement tombe sur ~85 % des fenêtres (une toutes les 30 min, cadence
# des fenêtres) et gèle la boucle 41-43 s, dont 5-7 s de modèle de preuve HF
# et **36-37 s de reconstruction vLLM** (max 61 s). Il atterrit dans les
# ~100 premières secondes, exactement quand la porte du batch code se remplit
# (retenus du marché p50 100 s). En v5 ça valait 6 % du revenu sur 2,4
# avancées/h ; ici c'est structurel. Le hot-swap vise les 36-37 s.
# ⛔ Le préalable « exiger des token ids IDENTIQUES » est IMPOSSIBLE tel quel :
# la gate compare vLLM au teacher-forcing HF, or les deux diffèrent
# numériquement même sur un moteur sain (0,9793 groupe / 0,9231 pire rollout,
# gate de conformité du 06/08). Exiger 1,000 = FAIL systématique.
# ⇒ MARCHE INTERMÉDIAIRE : `shadow`. On échange, on passe la gate, on
# JOURNALISE, puis on reconstruit quand même — le moteur qui sert la fenêtre
# est celui d'aujourd'hui, zéro risque de conformité. Le mode ombre sonde
# DEUX fois : `temoin-avant-echange` (vLLM encore sur les ANCIENS poids,
# hf_model déjà sur les NOUVEAUX = la signature exacte d'un `reload_weights`
# silencieusement raté) puis `apres-echange`. Le plancher n'est défendable
# que si les deux distributions se SÉPARENT ; si elles se recouvrent, le
# hot-swap reste NO-GO à tout plancher et on le referme pour de bon.
# Coût du mode ombre : l'échange + 2 sondes s'ajoutent au gel, ~10-20 s sur
# quelques fenêtres. Repli : remettre 0.
# Lecture : grep 'hot-swap self-gate\[' miner.log
export RELIQUARY_HOT_SWAP=${RELIQUARY_HOT_SWAP:-0}
# Gate réglable SANS redéploiement (un restart coûte une fenêtre). Le
# plancher se fixe DEPUIS la mesure du mode ombre, jamais d'intuition.
# 256 tokens plutôt que 48 : à taux vrai ~0,93 l'écart-type tombe de 3,7 pts
# à 1,6 pt, sinon un moteur sain échouerait par pur bruit d'échantillonnage.
export RELIQUARY_HOT_SWAP_GATE_TOKENS=${RELIQUARY_HOT_SWAP_GATE_TOKENS:-256}
export RELIQUARY_HOT_SWAP_GATE_FLOOR=${RELIQUARY_HOT_SWAP_GATE_FLOOR:-0.80}
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
export RELIQUARY_FIRE_CURFEW_S=${RELIQUARY_FIRE_CURFEW_S:-27}

# ── PORT v6 fill-closed (08/09, rapports B §4.1 / C §5) ─────────────────────
# Fenêtre remplie au quota (~1 800 s, cutoff precommit lu sur /state.fill_closed)
# : plus de flip à l'horloge, plus de deadline 100 s. Les réglages calés sur
# 100 s tueraient le mineur à 27-50 s. Bloc NON exécuté sous v5 (byte-identique).
if [ "${RELIQUARY_PROTOCOL_VERSION}" = "6" ]; then
  export RELIQUARY_FIRE_CURFEW_S=${_V6_USER_FIRE_CURFEW_S:-0}          # garde fill_closed (cutoff − marge) dans le code
  export RELIQUARY_LATE_BAKE_FROM=${_V6_USER_LATE_BAKE_FROM:-999999}   # bake_guard_decision → toujours "full"
  export RELIQUARY_PREFLIP_GUARD_S=${_V6_USER_PREFLIP_GUARD_S:-999999}
  # LATE_BAKE_CAP=1200 inchangé (inactif : la zone capped est inatteignable)
  export RELIQUARY_V6_FILL_CUTOFF_MARGIN_S=${RELIQUARY_V6_FILL_CUTOFF_MARGIN_S:-40}  # 33 s de grâce + p90 chaîne corps ~7 s
  export RELIQUARY_STATE_RETRY_MAX_S=${RELIQUARY_STATE_RETRY_MAX_S:-0.25}           # backoff 503 (50 ms → 250 ms)
  # Dette de preuve v6 : 2 échecs token_tampered/grail = fenêtre morte (1 800 s).
  # Le miroir local revient à ON sous v6 SEULEMENT (v5 reste :-0, cf. 07/09).
  export RELIQUARY_LOCAL_TOKEN_AUTH=${_V6_USER_LOCAL_TOKEN_AUTH:-1}
  # 14/09 : miroir des SEULS contrôles d'authenticité enforcés en fill-closed
  # (seuil dur 1e-8, tokens numériques 1e-6 & argmax ≥ 0,99 — marges ×10).
  # La porte « tous tokens » (non enforcée en V1) jetait 9 groupes sur 20.
  # Repli : RELIQUARY_LTA_MODE=soft.
  export RELIQUARY_LTA_MODE=${RELIQUARY_LTA_MODE:-validator}
  # 14/09 : avec la réparation de l'EOS, chaque groupe attend surtout la
  # réplique et vLLM (pas le CPU) ; à 3, jusqu'à 26 s d'attente de sémaphore
  # mesurées (fen 45896).
  # 16/09 : 8 -> 3 (défaut du code). Le sémaphore enveloppe `_pre_bake_entry`
  # (engine.py:5590) QUI CONTIENT LA PREUVE (t_proof_start) : à 8, les 10 groupes
  # du bake 0 entrent en preuve ensemble et la tête paie le plein tarif — preuve
  # p50 0,78 s à 0 chevauchement contre 4,90-5,45 s à 6, et nos têtes tournent à 6.
  # Attendu : tir de tête 19,8 -> ~17 s, où l'admission passe de 58 % à >90 %.
  # JUGER (mécanique, 8-10 fen) : heure p50 de nos 3 premiers corps (réf 19,8 s)
  # et durée p50 du bloc preuve∥EOS des groupes 1-3 du bake 0 (réf 3,47-4,02 s).
  # REPLI à 8 si, à 10 fenêtres : le bloc ne descend pas sous 3,0 s, OU les corps
  # acceptés dans la bande 25-80 s tombent sous 3,5/fen (réf ~4,3 — les bakes 1-3
  # font 36 % du revenu et c'est EUX que ce réglage sérialise), OU les corps
  # acceptés < 60 s passent sous 5,8/fen.
  # ⛔ 16/09 13h40 : 3 REPLIÉ à 8. Mécanisme confirmé (preuve p50 3,67 -> 2,14 s sur
  # 125 groupes) MAIS paiement R2 46090-46094 = 4,5,3,2,0 -> 2,80/fen contre 6,33
  # (réf 46050-46079) ; admis <25 s 3,67 -> 2,00 ; critère posé d'avance (payés
  # < 5,0) franchi. Mécanisme de la baisse NON démontré : attente sémaphore p90
  # 1,02 s seulement (646 groupes). Confondant : marché plus rapide (56e place code
  # 17,5 -> 14,8 s) mais leader inchangé à 14/fen. CE REPLI EST L'EXPÉRIENCE QUI
  # TRANCHE : remontée vers ~6/fen = c'était le fix 1 ; ~3/fen = le marché.
  # Ne PAS viser 6,33 : cette référence date d'un marché 2,7 s plus lent.
  # Prochain fix : priorité de tête (2-3 premiers groupes) SANS brider les suivants.
  export RELIQUARY_GRADE_CONCURRENCY=${_V6_USER_GRADE_CONCURRENCY:-8}
  # 17/09 : 5 s → 1 s. Le motif du 09/09 (« la fenêtre de 1 800 s ne se joue plus
  # à la seconde ») est faux sous V1 FIFO : 15-24 % de nos groupes de tête
  # attendaient 5 s leur notation avant la preuve, et le bake suivant attend
  # la fin de TOUTES les notations (GPU à l'arrêt 2,9-4,3 s entre bakes).
  # Accord validateur à 1 s : 99,87 % (05/09) ; TIMEOUT_IMPUTE=1 actif.
  # JUGER (8-10 fen) : part des 1ers tirs < 25 s avec pregrade ≥ 4,5 s (réf 15 %)
  # et `bake_diag: attente_notations` ; garde : hors zone validateur ≤ 0,2/fen.
  # REPLI : 5.0 + restart.
  export RELIQUARY_GRADE_TIMEOUT_S=${_V6_USER_GRADE_TIMEOUT_S:-1.0}
  # 17/09 : A/B des réessais batch_filled par parité de fenêtre (impaire =
  # rapide : pas 0,5 s, plafond 2 s, 40 essais, arrêt 90 s après l'ouverture ;
  # paire = règle historique). JUGER (~40 fen) : payés R2 impaires − paires.
  # REPLI : RELIQUARY_RETRY_AB=0.
  # 18/09 : A/B CLOS — 46 fen (46212-46257), impaires rapides 6,83 payés/fen contre
  # paires témoin 6,96 : aucun gain. Retour à la règle historique partout.
  export RELIQUARY_RETRY_AB=${RELIQUARY_RETRY_AB:-0}
  # 18/09 (mesure pure) : marge CDF de l'EOS final calculée sur la passe HF de la
  # preuve, à comparer au journal de la réplique (prérequis vérif EOS sélective).
  export RELIQUARY_PROOF_MARGIN_DUMP=${RELIQUARY_PROOF_MARGIN_DUMP:-/workspace/proof_margin_v4.jsonl}
  # 18/09 (restart F) : PRIORITÉ DES PREUVES DE TÊTE. Preuve seule 1,18 s p50, puis
  # +1,2 s par preuve déjà en vol (46 fen) ; les groupes 1-3 en ont jusqu'à 2 devant
  # eux. JUGER (8-10 fen) : preuve seule des groupes 1-3 (réf 2,00 s p50, 4,93 p90),
  # tir des groupes 1-3 (réf 14,6 s), preuve des rangs 4-8 (réf 2,82 — ne doit pas
  # exploser). REPLI : RELIQUARY_PROOF_PRIORITY=0.
  export RELIQUARY_PROOF_PRIORITY=${RELIQUARY_PROOF_PRIORITY:-1}
  # 18/09 (restart H) : VÉRIFICATION EOS SÉLECTIVE. Bloc preuve∥EOS des groupes 1-3 =
  # 2,73 s, tenu par la réplique (preuve 1,23 s). Marge côté preuve > 0,15 ⇒ pas de
  # réplique (42 237 rollouts appariés : 0 des 96 refus au-dessus de 0,0496 ; 5,3
  # rollouts vérifiés par groupe au lieu de 16). JUGER : bloc des groupes 1-3 (réf
  # 2,73 s), tir des groupes 1-3 (réf 14,1 s), VIGIE bad_termination = 0.
  # REPLI : RELIQUARY_EOS_SELECTIVE=0.
  export RELIQUARY_EOS_SELECTIVE=${RELIQUARY_EOS_SELECTIVE:-1}
  export RELIQUARY_EOS_SELECTIVE_MARGIN=${RELIQUARY_EOS_SELECTIVE_MARGIN:-0.15}
  # 17/09 restart B — 2 changements à traces DISTINCTES :
  # (1) flip par GET /miner-state (5 ko) pendant le trou 503. flip_diag restart A
  #     (11 fen) : 4/11 fenêtres signalent le flip à ~5 s (GET /state 2,8-3,5 s +
  #     cooldown 1,5 s) au lieu de ~1,5 s. JUGER : signal_off p50/p90 et part > 3 s
  #     (réf 4/11). REPLI : RELIQUARY_MINER_STATE_FLIP=0.
  export RELIQUARY_MINER_STATE_FLIP=${RELIQUARY_MINER_STATE_FLIP:-1}
  # 19/09 (H100) : le bake n'attend plus le GET /state après le flip /miner-state.
  # Mesuré signal du flip → bake_start : 1,21 s médiane H200 (31 fen), 1,31 s H100
  # (24 fen), p90 2,2-2,3 s — le générateur relisait l'ancien _last_state (d'avant
  # le trou 503) et se rendormait jusqu'à la confirmation par /state (669 ko).
  # JUGER (mécanique, 8 fen) : signal → bake_start (réf 1,31 s, attendu ~0,1-0,2),
  # lignes « ms_flip_bake » (1/fen) et « ms_flip_confirm » (délai /state évité),
  # VIGIE : 0 rejet window_mismatch / prompt_out_of_range / content_in_cooldown
  # de plus qu'avant. REPLI : RELIQUARY_MS_FLIP_BAKE=0.
  export RELIQUARY_MS_FLIP_BAKE=${RELIQUARY_MS_FLIP_BAKE:-1}
  # 20/09 : le bake suivant n'attend plus les notations/preuves du lot.
  # Mesuré (bake_diag, 872 bakes) : 1,35 s médiane (3,37 s p90) de GPU à l'arrêt
  # en fin de bake. Le code attendait pour vider _phase1_cache d'un bloc ; on
  # purge désormais SÉLECTIVEMENT (on garde la fenêtre courante) et la randomness
  # du bake voyage par contexte, donc une notation tardive lit le bon cache au
  # lieu de régénérer (~40 s) — et son entrée est refusée au pool si la fenêtre
  # a changé.
  # JUGER (mécanique, 8-10 fen) : bake_diag attente_notations (réf 1,35 s → ~0),
  # taches_en_vol > 0, ms/token des bakes 2+ (réf 10,3), tir g1-3 (réf 12,46 s),
  # tirs < 18 s/fen (réf 4,0).
  # ⚠️ RISQUE : le bake suivant démarre pendant les preuves du lot précédent —
  # contention GPU (bake 14 et sprint 3 ont échoué là-dessus). REPLIER si
  # ms/token monte, si le tir g1-3 recule, ou si « entrée PÉRIMÉE … randomness »
  # apparaît plus d'une fois par fenêtre.
  # REPLI : RELIQUARY_BAKE_WAIT_GRADES=1.
  export RELIQUARY_BAKE_WAIT_GRADES=${RELIQUARY_BAKE_WAIT_GRADES:-0}
  export RELIQUARY_MS_FLIP_BAKE_MAX_S=${RELIQUARY_MS_FLIP_BAKE_MAX_S:-15}
  # (2) T1 : vérification EOS précoce (réplique au fil du décodage) COUPÉE — la
  #     vérification reste faite, après « prêt », par la réparation. Décodage bake 10
  #     8,7 ms/token (14/09, avant f66be53) contre 12,6 (17/09) ; preuve seule des
  #     groupes 1-5 2,79 s p50 contre ~0,55 à vide. JUGER : ms/token (bake_speed.py),
  #     preuve seule, chaîne prêt→précommit (réf 3,75 s). REPLI si ms/token ≥ 11 ou
  #     chaîne p50 > 4,5 s : RELIQUARY_TERMINAL_EARLY_VERIFY=1.
  export RELIQUARY_TERMINAL_EARLY_VERIFY=${RELIQUARY_TERMINAL_EARLY_VERIFY:-0}
  # Watchdog : 1 800 s de fenêtre + marge ; le heartbeat du moteur = signe de vie.
  export WATCHDOG_WEDGE_S=${WATCHDOG_WEDGE_S:-2700}
  # ── V1 FIFO (validateur 1f1cc16/#253, live 12/09 23:23) ─────────────────
  # Sélection = ordre d'arrivée du CORPS par env, paiement FIXE par groupe
  # retenu (1/336 du pool). Les 2 « têtes » précoces visaient la course aux
  # rounds drand, disparue : une rafale dense vaut mieux. Les gagnants posent
  # 8-12 groupes entre 13 et 35 s.
  export RELIQUARY_SPRINT_SIZE=${_V6_USER_SPRINT_SIZE:-0}
  export RELIQUARY_HEAD_FIFO=${_V6_USER_HEAD_FIFO:-0}
  # Chaque seconde d'attente d'un corps prêt recule sa place FIFO. Plafond
  # validateur : 16 reçus non révélés par hotkey.
  export RELIQUARY_MAX_INFLIGHT_FIRES=${_V6_USER_MAX_INFLIGHT_FIRES:-8}
  # Checkpoint publié ~138 s avant l'ouverture : détecter vite, précharger les
  # poids pendant le trou 503 (engine.preload_decision). Repli : PRELOAD=0.
  export RELIQUARY_CHECKPOINT_PREFETCH_POLL_S=${_V6_USER_PREFETCH_POLL_S:-5}
  export RELIQUARY_CHECKPOINT_PRELOAD=${RELIQUARY_CHECKPOINT_PRELOAD:-1}
  # Réplique du validateur (ops/replica_service.py, venv_val : torch 2.7,
  # transformers 5.10.4, flash-attn 2.8.3) : verdict EXACT de l'EOS final
  # (#253). Service lancé par restart_miner.sh. Vide = garde locale seule.
  export RELIQUARY_REPLICA_SOCKET=${RELIQUARY_REPLICA_SOCKET-/workspace/replica.sock}
  # Budget de tokens par continuation de réparation (14/09) : au-delà le groupe
  # est abandonné. 512 jetait ~34 % des groupes en zone ; le bake n'attend plus
  # les continuations, donc 2048.
  export RELIQUARY_TERMINAL_REPAIR_MAX_NEW=${RELIQUARY_TERMINAL_REPAIR_MAX_NEW:-2048}
  # 15/09 : fils du vérificateur précoce de l'EOS final. 16 TESTÉ (fen 46020-46025)
  # puis REPLIÉ : prêt → précommit −0,8 s p50 (p90 +0,2 s), tirs < 18 s 2,2 → 3,2/fen,
  # MAIS payés code < 25 s inchangés (4,27 → 4,40) et payés totaux 8,64 → 6,60
  # (5 fen, coupures plus courtes). Le délai de vérification n'est pas le verrou
  # de la rafale de tête. Garder 4 tant que la rafale n'a pas grossi.
  # 16/09 : 4 → 8. Le verdict « 16 fils = pas de gain » ci-dessus est
  # SOUS-DIMENSIONNÉ : 6 fenêtres pour un effet attendu de +0,3 payé/fen.
  # La courbe d'acceptation mesurée sur 91 fen (100 % avant 18 s, 75 % à
  # 18-20, 62 % à 20-22, 51 % à 22-24) donne ~+0,5 payé par seconde gagnée
  # sur la chaîne ; à 16 fils la chaîne perdait 0,8 s. On prend la valeur
  # intermédiaire (8) pour ne pas re-charger la réplique (2 workers).
  # JUGER : t_pick → t_precommit_sent p50 (réf 4,1 s) ET p90 (réf 4,94 s à
  # 4 fils, 5,18 à 16) sur 30 fenêtres. REPLI : 4.
  # ⛔ 16/09 : 8 replié à 4 avec le mémo — retour à l'ère validée, un seul
  # changement à la fois ensuite. Sa signature propre était BONNE (chaîne p50
  # 4,1 -> 2,9-3,2 s) : à re-tester SEUL sur 30 fenêtres.
  export RELIQUARY_TERMINAL_EARLY_WORKERS=${RELIQUARY_TERMINAL_EARLY_WORKERS:-4}
  # 16/09 14h05 : ACTIVE le patch 2 (02feed0). _on_rollout envoie chaque rollout
  # à la réplique dès sa sortie du décodeur, AVANT le grading ; 46-54 % des groupes
  # sont ensuite jetés out_of_zone et la réplique est FIFO à 2 workers. Au rejet
  # hors zone, les vérifications encore EN FILE de ce groupe sont annulées.
  # JUGER (mécanique, ~8 fen) : durée t_pregrade_end -> t_repair_end des groupes
  # GARDÉS (réf ère GRADE=8 mémo 2). REPLI immédiat (=0) si les payés tombent
  # sous 2/fen sur 4 fenêtres consécutives, ou si la réparation s'ALLONGE.
  export RELIQUARY_EARLY_CANCEL_OOZ=${RELIQUARY_EARLY_CANCEL_OOZ:-1}
  # 16/09 14h25 : ACTIVE le réveil au flip (0ebac59). Pendant le trou 503 le
  # générateur dormait par tranches d'1 s et rien ne le réveillait à l'ouverture :
  # le bake partait 0-1 s après la détection (0,5 s en moyenne). Le flip lève
  # désormais un Event ; la pause l'attend au plus 1,0 s.
  # JUGER (mécanique) : délai ouverture -> début du bake et 1er groupe prêt,
  # hors 1re fenêtre après restart. REPLI (=0) si ce délai ne baisse pas ou si
  # le GPU tourne à vide (boucle chaude).
  export RELIQUARY_WAKE_ON_FLIP=${RELIQUARY_WAKE_ON_FLIP:-1}
  # 15/09 : tokens RÉELLEMENT soumis (1 ligne/groupe grâce au cache du
  # finalize), pour rejouer chaque seed_mismatch token par token. ~300 Mo/jour.
  # Vide = coupé.
  export RELIQUARY_DUMP_SUBMISSION=${RELIQUARY_DUMP_SUBMISSION-/workspace/submissions_dump.jsonl}
  # Dépôt des checkpoints connu dès le démarrage : un redémarrage pendant le
  # trou 503 précharge quand même (engine._active_ckpt_repo).
  export RELIQUARY_CHECKPOINT_REPO_DEFAULT=${RELIQUARY_CHECKPOINT_REPO_DEFAULT:-ReliquaryForge/qwen3-4b-base-dapo-v4}
  # Place VRAM pour le modèle de la réplique (~8 Go) : cache KV vLLM utilisé
  # à ~10 % (mesuré 12/09), 0,76 → 0,70.
  # 19/09 : sur carte < 100 Go, défaut = fraction du profil carte (cf. en tête).
  export RELIQUARY_VLLM_GPU_FRACTION=${_V6_USER_VLLM_GPU_FRACTION:-${_GPU_PROFILE_FRACTION:-0.70}}
  # INCHANGÉS et voulus : VOLUME_MU=0, DRAND_MIN_HEADROOM_S=1.0,
  # CHECKPOINT_PREFETCH=1, COOLDOWN_POLL_S=20, MEMO_HEAD_SLOTS.
fi

CHECKPOINT="${CHECKPOINT:-Qwen/Qwen3-4B-Base}"

# Sanity : refuse de démarrer si nos constantes ne reflètent pas le contrat.
# Cette garde a DEJA evite un lancement perdu le 24/08 : apres le port v5 du
# prompt, GENERATION_PROFILE_ID etait reste fige sur la valeur v4 — soit 100 %
# de GENERATION_CONTRACT_MISMATCH. Elle compare desormais au contrat LIVE, pas
# a des valeurs ecrites en dur : au prochain cutover elle dira quoi corriger.
/workspace/venv/bin/python - <<'EOF' || exit 1
import json, os as _os, urllib.request
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
    # 10/09 : cette URL etait EN DUR et pointait sur l'ancien validateur —
    # la garde criait « parite NON verifiee » alors que le mineur tournait bien
    # sur la nouvelle adresse. On lit celle que le launcher vient de resoudre.
    h = json.loads(urllib.request.urlopen(
        _os.environ.get("RELIQUARY_VALIDATOR_URL",
                        "http://62.238.81.36:8000").rstrip("/") + "/health",
        timeout=15).read())
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
    # 08/09 (port v6) : sha256 CANONIQUE du generation_contract publie —
    # json.dumps(sort_keys, separators compacts), methode du validateur
    # (shared/training_payload.py). Attendus recalcules depuis profiles.py
    # de la branche v6 et VERIFIES contre le live v5 (19e98f5a...).
    # Revue item 3 : le validateur n'impose que protocole + profil
    # (server.py:4112-4118) -> un contrat retouche sans changement de
    # generation (redeploiement d'image) = AVERTISSEMENT, jamais un abort
    # (sinon boucle watchdog -> launcher -> abort toutes les 2 min). Seuls
    # ABORTENT : protocole, profil, sha256 des TEMPLATES (ce qui casse la
    # generation). Inerte si /health ne publie pas de contrat.
    import hashlib
    EXPECTED = {5: "19e98f5a3ddac1980efe66fd80db1ec0f8db87a5e60934efd5d0e8985435eadd",
                6: "1696eef2a8ff52284842f2253d6f699b50bc657dc93b20fc61a257db7d449385"}
    gc = h.get("generation_contract") or {}
    if gc:
        live_sha = hashlib.sha256(json.dumps(
            gc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        attendu = EXPECTED.get(c.PROTOCOL_VERSION)
        if attendu and live_sha != attendu:
            # cles connues de nos constantes : on nomme celles qui different
            nous = {
                "profile_id": c.GENERATION_PROFILE_ID,
                "protocol_version": c.PROTOCOL_VERSION,
                "model_id": c.DEFAULT_BASE_MODEL,
                "model_revision": c.DEFAULT_BASE_MODEL_REVISION,
                "sampling.rollouts": c.M_ROLLOUTS,
                "sampling.temperature": c.T_PROTO,
                "sampling.top_p": c.TOP_P_PROTO,
                "sampling.top_k": c.TOP_K_PROTO,
            }
            for env_name in (gc.get("environments") or {}):
                nous[f"environments.{env_name}.max_new_tokens"] = c.MAX_NEW_TOKENS_PROTOCOL_CAP
            diffs = []
            for k, v in nous.items():
                cur = gc
                for part in k.split("."):
                    cur = cur.get(part) if isinstance(cur, dict) else None
                if cur != v:
                    diffs.append(f"{k}: nous {v!r} / eux {cur!r}")
            print(f"[garde] AVERTISSEMENT contrat v{c.PROTOCOL_VERSION} : sha live "
                  f"{live_sha[:12]} != attendu {attendu[:12]} ; cles connues "
                  f"differentes : {diffs or 'aucune (cle inconnue ajoutee/retouchee)'} "
                  f"; cles live : {sorted(gc)}")
        from reliquary.protocol.profiles import prompt_template_for
        for env_name, env_c in (gc.get("environments") or {}).items():
            tpl = prompt_template_for(env_name)
            leur = (env_c.get("prompt_template") or {}).get("sha256")
            if tpl is not None and tpl.sha256() != leur:
                ecarts.append(f"template {env_name}: nous {tpl.sha256()[:12]} "
                              f"/ eux {str(leur)[:12]}")
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
