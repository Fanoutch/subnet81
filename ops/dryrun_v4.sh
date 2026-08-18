#!/bin/bash
# DRY-RUN de conformité v4 sur box GPU (sans toucher au réseau réel) :
# wallet JETABLE + validateur factice local + le VRAI mineur en boucle
# complète (poll /state → bake forced-seed → grade → gates → merkle/signature/
# precommit → submit → /verdicts). Le mock juge avec nos ports des checks
# validateur et écrit /workspace/mock_checks.jsonl.
# Usage (box) : bash ops/dryrun_v4.sh   puis lire mock_checks après ~5-8 min.
set -uo pipefail
cd "$(dirname "$0")/.."
source /workspace/venv/bin/activate
source ops/bench_env.sh
export PYTHONPATH=/workspace/reliquary-miner-priv

# 1) wallet jetable (signature auto-cohérente : le mock vérifie avec la même clé)
python - <<'EOF'
import bittensor as bt
w = bt.wallet(name="dryrun", hotkey="h1")  # chemin défaut (~/.bittensor) : la CLI n'a pas --wallet-path
try:
    w.coldkey_file.exists_on_device() or w.create_new_coldkey(use_password=False, overwrite=True)
    w.hotkey_file.exists_on_device() or w.create_new_hotkey(use_password=False, overwrite=True)
except Exception:
    w.create_new_coldkey(use_password=False, overwrite=True)
    w.create_new_hotkey(use_password=False, overwrite=True)
print("wallet dryrun prêt:", w.hotkey.ss58_address)
EOF

# 2) mock validateur (tmux mockv)
rm -f /workspace/mock_checks.jsonl
tmux kill-session -t mockv 2>/dev/null || true
tmux new-session -d -s mockv "cd /workspace/reliquary-miner-priv && \
  RELIQUARY_PROTOCOL_VERSION=4 PYTHONPATH=. \
  /workspace/venv/bin/python ops/mock_validator_v4.py 2>&1 | tee /workspace/mockv.log"
sleep 2
curl -s http://127.0.0.1:9999/state | head -c 200; echo

# 3) le VRAI mineur, config v4 de prod (GPU 0.60 : place pour le modèle HF des
#    preuves), pointé sur le mock. Dumps dédiés dry-run.
tmux kill-session -t dryrun 2>/dev/null || true
tmux new-session -d -s dryrun "cd /workspace/reliquary-miner-priv && \
  export PYTHONPATH=. HF_HOME=/workspace/hf GRAIL_ATTN_IMPL=sdpa && \
  export RELIQUARY_PROTOCOL_VERSION=4 RELIQUARY_VALIDATOR_URL=http://127.0.0.1:9999 && \
  export RELIQUARY_ACTIVE_ENVS=opencodeinstruct RELIQUARY_AUCTION_MIN_SCORE=0 && \
  export RELIQUARY_VLLM_FORCED_SEED=1 RELIQUARY_VLLM_CUDA_GRAPHS=1 && \
  export RELIQUARY_VLLM_GPU_FRACTION=0.60 RELIQUARY_VLLM_MAX_NUM_SEQS=256 && \
  export RELIQUARY_BAKE_BATCH_SIZE=4 RELIQUARY_MEMO_SLOT=1 RELIQUARY_MEMO_MIN_SCORE=0.23 && \
  export RELIQUARY_SAMPLE_DUMP=/workspace/dryrun_samples.jsonl && \
  export RELIQUARY_VERDICTS_DUMP=/workspace/dryrun_verdicts.jsonl && \
  export RELIQUARY_SUBMIT_DUMP=/workspace/dryrun_submits.jsonl && \
  export RELIQUARY_WINDOW_DUMP=/workspace/dryrun_windows.jsonl && \
  export VLLM_USE_DEEP_GEMM=0 VLLM_DEEP_GEMM_WARMUP=skip VLLM_USE_FLASHINFER_SAMPLER=0 && \
  export CUDA_HOME=/workspace/venv/lib/python3.12/site-packages/nvidia/cu13 && \
  export PATH=/workspace/venv/bin:\$CUDA_HOME/bin:\$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
  /workspace/venv/bin/python -m reliquary.cli.main mine \
    --wallet-name dryrun --hotkey h1 \
    --network finney --netuid 81 \
    --validator-url http://127.0.0.1:9999 \
    --checkpoint Qwen/Qwen3-4B-Base --log-level INFO \
    2>&1 | tee /workspace/dryrun_miner.log"
echo "dry-run lancé : tmux mockv + dryrun. Bilan : bash ops/dryrun_v4_bilan.sh"
