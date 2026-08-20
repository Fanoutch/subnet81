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
export RELIQUARY_DRAND_URLS=${RELIQUARY_DRAND_URLS:-"https://api.drand.sh,https://api2.drand.sh,https://api3.drand.sh,https://drand.cloudflare.com"}

# ── Bascule v4 : LE flag + les ceintures ────────────────────────────────────
export RELIQUARY_PROTOCOL_VERSION=${RELIQUARY_PROTOCOL_VERSION:-4}
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
export RELIQUARY_SCAN_HOLDOFF_S=${RELIQUARY_SCAN_HOLDOFF_S:-10}
# Vedette memo rapide : bande de longueur mesuree (meta gagnant 2900-7200 tok)
export RELIQUARY_MEMO_FAST_BAND=${RELIQUARY_MEMO_FAST_BAND:-1500,5500}
# Couvre-feu de bake (etude drain 18/08) : pas de nouveau bake apres T+95s
export RELIQUARY_BAKE_CURFEW_S=${RELIQUARY_BAKE_CURFEW_S:-95}
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
export RELIQUARY_PROMPT_PREDICTOR=${RELIQUARY_PROMPT_PREDICTOR:-/workspace/predictor_v50.json}
# 2 slots explore = labels non biaisés pour les ré-entraînements (obligatoire
# dès qu'un prior influence les picks — leçon v4.3/mémorisation).
export RELIQUARY_EXPLORE_SLOTS=${RELIQUARY_EXPLORE_SLOTS:-2}

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
export RELIQUARY_SPRINT_SIZE=${RELIQUARY_SPRINT_SIZE:-8}
export RELIQUARY_GRADE_CONCURRENCY=${RELIQUARY_GRADE_CONCURRENCY:-8}
# Mode course 2026-08-19 : garde pré-flip (GPU libre au flip) + rafale 8
export RELIQUARY_LATE_BAKE_FROM=${RELIQUARY_LATE_BAKE_FROM:-110}
export RELIQUARY_PREFLIP_GUARD_S=${RELIQUARY_PREFLIP_GUARD_S:-170}
export RELIQUARY_LATE_BAKE_CAP=${RELIQUARY_LATE_BAKE_CAP:-1200}
# Streaming C 19/08 : preuve spéculative des têtes de rafale (parallèle au grading)
export RELIQUARY_SPEC_PROOF=${RELIQUARY_SPEC_PROOF:-0}
export RELIQUARY_SPEC_PROOF_SLOTS=${RELIQUARY_SPEC_PROOF_SLOTS:-4}
# AUTO-FILTRAGE 19/08 (rapport agents) : miroir local des checks validateur,
# posé APRÈS le bloc unset des seuils v3 plus haut — marges sûres v4.
export RELIQUARY_MIN_LOCAL_Q10=${RELIQUARY_MIN_LOCAL_Q10:-0.0005}
export RELIQUARY_MIN_LOCAL_MEDIAN=${RELIQUARY_MIN_LOCAL_MEDIAN:-0.08}
export RELIQUARY_LOCAL_TOKEN_AUTH=${RELIQUARY_LOCAL_TOKEN_AUTH:-1}
export RELIQUARY_LTA_CHOSEN_MAX=${RELIQUARY_LTA_CHOSEN_MAX:-1e-5}
export RELIQUARY_LTA_ARGMAX_MIN=${RELIQUARY_LTA_ARGMAX_MIN:-0.985}
# Malus anti-rollout-court (20/08) : dé-priorise à la SÉLECTION les prompts
# qui produisent des rollouts <32 tok (inéligibles CHALLENGE_K, 0 payé/333).
export RELIQUARY_SHORT_RISK_MODEL=${RELIQUARY_SHORT_RISK_MODEL:-/workspace/risk_short_v1.json}
export RELIQUARY_SHORT_RISK_LAMBDA=${RELIQUARY_SHORT_RISK_LAMBDA:-0.08}
# Guérison divergence : kernel cascade OFF (16 rollouts même prompt = forme
# de batch que le validateur ne vérifie jamais — cf. audit parité 19/08)
export RELIQUARY_VLLM_DISABLE_CASCADE=${RELIQUARY_VLLM_DISABLE_CASCADE:-0}
export RELIQUARY_BAKE_CHUNK=${RELIQUARY_BAKE_CHUNK:-64}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_DEEP_GEMM=0
export VLLM_DEEP_GEMM_WARMUP=skip         # 0.24 enum: skip|full|relax (NOT 0/1)
export VLLM_USE_FLASHINFER_SAMPLER=0      # ptxas PTX 9.2 vs 9.0
export CUDA_HOME=/workspace/venv/lib/python3.12/site-packages/nvidia/cu13
export PATH=/workspace/venv/bin:$CUDA_HOME/bin:$PATH
# HOT_SWAP et FS_GRAPH restent OFF en v4 (items 14/15 : recalibrer sous T=1.0
# et re-mesurer la VRAM à M=16 avant activation) — défauts code = 0.

CHECKPOINT="${CHECKPOINT:-Qwen/Qwen3-4B-Base}"

# Sanity : refuse de démarrer si les constantes ne reflètent pas le contrat v4.
/workspace/venv/bin/python - <<'EOF' || exit 1
from reliquary import constants as c
assert c.PROTOCOL_VERSION == 4, c.PROTOCOL_VERSION
assert c.M_ROLLOUTS == 16 and not c.BFT_ENABLED
assert c.MAX_NEW_TOKENS_PROTOCOL_CAP == 8192
assert (c.T_PROTO, c.TOP_P_PROTO, c.TOP_K_PROTO) == (1.0, 1.0, 0)
assert c.FORCED_SEED_DOMAIN == "reliquary-forced-seed-v4"
assert c.GENERATION_PROFILE_ID == "qwen3-4b-base-dapo-v4", c.GENERATION_PROFILE_ID
assert c.MATH_ANSWER_FORMAT == "boxed"
assert c.RAW_COMPLETION_PROMPTS and c.OMI_TRAIN_SHARDS_ONLY
print("v4 constants OK:", c.GENERATION_PROFILE_ID)
EOF

cd /workspace/reliquary-miner-priv
exec /workspace/venv/bin/python -m reliquary.cli.main mine \
  --wallet-name camille81-v2 --hotkey hotkey81 --network finney --netuid 81 \
  ${RELIQUARY_VALIDATOR_URL:+--validator-url $RELIQUARY_VALIDATOR_URL} \
  --checkpoint "$CHECKPOINT" \
  --log-level INFO
