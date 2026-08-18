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

# ── Réseau (calibré box 31.22.104.180) ─────────────────────────────────────
export RELIQUARY_VALIDATOR_URL=${RELIQUARY_VALIDATOR_URL:-http://209.20.157.231:8080}  # egress direct testé
# secureweb3 exclu : injoignable depuis cette box (gel 15-20 s/tirage, flips
# ratés 28968-70) ; les 4 miroirs re-testés OK le 2026-08-18.
export RELIQUARY_DRAND_URLS=${RELIQUARY_DRAND_URLS:-"https://api.drand.sh,https://api2.drand.sh,https://api3.drand.sh,https://drand.cloudflare.com"}

# ── Bascule v4 : LE flag + les ceintures ────────────────────────────────────
export RELIQUARY_PROTOCOL_VERSION=${RELIQUARY_PROTOCOL_VERSION:-4}
export RELIQUARY_AUCTION_MIN_SCORE=${RELIQUARY_AUCTION_MIN_SCORE:-0}
export RELIQUARY_RANKING_BUDGET_S=${RELIQUARY_RANKING_BUDGET_S:-12}
export RELIQUARY_SAMPLE_DUMP=${RELIQUARY_SAMPLE_DUMP:-/workspace/samples_v4.jsonl}
# Slot mémo : ON avec store FRAIS (le mémo s'amorce depuis SAMPLE_DUMP — le
# chemin v4 neuf garantit zéro contamination v3 ; il se remplit tout seul).
# MEMO_MIN_SCORE : 0 au départ (= comportement mesuré +6 pts d'armement) ;
# à recalibrer une fois la distribution des scores v4 connue (la zone 0.24
# rend « in_zone » quasi universel — cf. runbook §Prior v5).
export RELIQUARY_MEMO_SLOT=${RELIQUARY_MEMO_SLOT:-1}
# Dual-env dès H+0 : le split émissions est ~50/50 par env (part non minée =
# burn) et la métrique n°1 du runbook (boxing spontané math) exige des groupes
# math dans le dump. Si boxing ~0 après diagnostic → repasser code-only ici.
export RELIQUARY_ACTIVE_ENVS=${RELIQUARY_ACTIVE_ENVS:-openmathinstruct,opencodeinstruct}
# PURGE des réglages v3 hérités (K_MIN/K_MAX 2/6, MAX_NEW_TOKENS 2600/16384,
# prédicteurs v3, sprint 90 s, seuils q10 v3) : ne PAS les poser ici.
unset RELIQUARY_K_MIN RELIQUARY_K_MAX RELIQUARY_MAX_NEW_TOKENS \
      RELIQUARY_PROMPT_PREDICTOR RELIQUARY_PROMPT_PREDICTOR_2 \
      RELIQUARY_MIN_LOCAL_Q10 RELIQUARY_MIN_LOCAL_MEDIAN \
      RELIQUARY_SPRINT_MAX_WAIT_S RELIQUARY_MAX_TRUNCATED_CODE 2>/dev/null || true

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
