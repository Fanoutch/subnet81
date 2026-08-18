#!/bin/bash
# Launcher v4 compatible restart/watchdog (balayage 18/08 G1) — à déployer sur
# la box en /workspace/launch_miner_v4.sh, puis activer la chaîne de relance :
#   echo /workspace/launch_miner_v4.sh > /workspace/.miner_launcher
# (restart_miner.sh lit ce marqueur ; sans lui il relance launch_miner.sh = v3
# = 100 % GENERATION_CONTRACT_MISMATCH après le premier restart du watchdog.)
# Contrairement à start_miner_v4_DRAFT.sh (lancement manuel, tee foreground),
# ce script écrit sur stdout et laisse tmux/restart_miner.sh gérer le log.
set -uo pipefail

export PYTHONPATH=/workspace/reliquary-miner-priv
export HF_HOME=/workspace/hf
export GRAIL_ATTN_IMPL=sdpa

# ── Bascule v4 : LE flag + les ceintures ────────────────────────────────────
export RELIQUARY_PROTOCOL_VERSION=${RELIQUARY_PROTOCOL_VERSION:-4}
export RELIQUARY_AUCTION_MIN_SCORE=${RELIQUARY_AUCTION_MIN_SCORE:-0}
export RELIQUARY_RANKING_BUDGET_S=${RELIQUARY_RANKING_BUDGET_S:-12}
export RELIQUARY_VLLM_GPU_FRACTION=${RELIQUARY_VLLM_GPU_FRACTION:-0.76}
export RELIQUARY_SAMPLE_DUMP=${RELIQUARY_SAMPLE_DUMP:-/workspace/samples_v4.jsonl}
# PURGE des réglages v3 hérités (K_MIN/K_MAX 2/6, MAX_NEW_TOKENS 2600/16384,
# prédicteurs v3, sprint 90 s, seuils q10 v3) : ne PAS les poser ici.
unset RELIQUARY_K_MIN RELIQUARY_K_MAX RELIQUARY_MAX_NEW_TOKENS \
      RELIQUARY_PROMPT_PREDICTOR RELIQUARY_PROMPT_PREDICTOR_2 \
      RELIQUARY_MIN_LOCAL_Q10 RELIQUARY_MIN_LOCAL_MEDIAN \
      RELIQUARY_SPRINT_MAX_WAIT_S RELIQUARY_MAX_TRUNCATED_CODE 2>/dev/null || true

CHECKPOINT="${CHECKPOINT:-Qwen/Qwen3-4B-Base}"

# Sanity : refuse de démarrer si les constantes ne reflètent pas le contrat v4.
python3 - <<'EOF' || exit 1
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
exec python3 -m reliquary.cli.main mine \
  --wallet-name camille81-v2 --hotkey hotkey81 --network finney --netuid 81 \
  --checkpoint "$CHECKPOINT" \
  --log-level INFO
