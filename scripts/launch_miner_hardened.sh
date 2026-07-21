#!/usr/bin/env bash
# Hardened miner launcher — auto-restart supervisor.
#
# WHY: the boot path has unguarded external calls (Finney RPC, HF dataset/model
# load, validator discovery). A transient failure at boot raises and, with a
# bare `nohup … &` launch, leaves the miner DEAD until a manual restart → zero
# submissions. This wrapper restarts on ANY exit (backoff, reset after a healthy
# run), so a transient hiccup self-heals.
#
# Also fixes two deploy footguns found in scripts/launch_miner.sh:
#   * checkpoint defaulted to `gpt2` (wrong model) → here defaults to Qwen3.5
#     (correct arch); maybe_pull_checkpoint then follows the validator's
#     PUBLISHED trained checkpoint. NEVER hardcode a stale base model.
#   * no OOM guard → sets RELIQUARY_MAX_NEW_TOKENS=3500 (–40% compute on runaway
#     generations, safer under a shared H100).
#
# Override anything via env vars; sensible prod defaults below.
set -u

MINER_DIR="${RELIQUARY_MINER_DIR:-/root/reliquary-miner-priv}"
PYTHON="${RELIQUARY_PYTHON:-/root/venv/bin/python}"
LOG="${RELIQUARY_MINER_LOG:-$MINER_DIR/miner.log}"
WALLET="${BT_WALLET_NAME:-camille81}"
HOTKEY="${BT_HOTKEY:-miner1}"
NETWORK="${BT_NETWORK:-finney}"
NETUID="${NETUID:-81}"
# Boot model = correct ARCH only; the validator's published checkpoint is
# pulled at runtime by maybe_pull_checkpoint. Do NOT set this to gpt2/Qwen3-4B.
CHECKPOINT="${RELIQUARY_CHECKPOINT:-Qwen/Qwen3.5-4B}"

cd "$MINER_DIR" || { echo "cannot cd $MINER_DIR"; exit 1; }
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export RELIQUARY_MAX_NEW_TOKENS="${RELIQUARY_MAX_NEW_TOKENS:-3500}"
# Forced-seed: checkpoint_hash is a seed input (u_at). Entries baked under the
# OLD hash are guaranteed SEED_MISMATCH after a checkpoint advance — drop them
# (the pre-forced-seed "optimistic keep" GRAIL bet is always-losing now).
export RELIQUARY_DROP_POOL_ON_CKPT="${RELIQUARY_DROP_POOL_ON_CKPT:-1}"

backoff=5
while true; do
  start=$SECONDS
  echo "$(date -u +%FT%TZ) [supervisor] starting miner (checkpoint=$CHECKPOINT, max_new_tokens=$RELIQUARY_MAX_NEW_TOKENS)" | tee -a "$LOG"
  "$PYTHON" -m reliquary.cli.main mine \
      --wallet-name "$WALLET" --hotkey "$HOTKEY" \
      --network "$NETWORK" --netuid "$NETUID" \
      --checkpoint "$CHECKPOINT" --log-level INFO \
      >> "$LOG" 2>&1
  code=$?
  ran=$(( SECONDS - start ))
  # A run that survived >120s was healthy → reset backoff so a later crash
  # restarts fast. A crash-loop at boot keeps backing off (up to 60s) so a
  # hard-down endpoint isn't hammered.
  if [ "$ran" -gt 120 ]; then backoff=5; fi
  echo "$(date -u +%FT%TZ) [supervisor] miner exited code=$code after ${ran}s — restarting in ${backoff}s" | tee -a "$LOG"
  sleep "$backoff"
  if [ "$backoff" -lt 60 ]; then backoff=$(( backoff * 2 )); fi
done
