#!/bin/bash
# reliquary-miner-priv launch — post 2026-05-26 validator update
#   K=[2,6], MAX_NON_BTOK=5 (matches validator MAX_TRUNCATED=5 + no frontier check)
#   DROP_POOL_ON_CKPT=1 (avoid stale bakes across checkpoint advances)
cd /root/reliquary-miner-priv
PYTHONPATH=. \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
RELIQUARY_BAKE_BATCH_SIZE=6 \
RELIQUARY_DROP_POOL_ON_CKPT=1 \
RELIQUARY_K_MIN=2 \
RELIQUARY_K_MAX=6 \
RELIQUARY_MAX_NON_BTOK_IN_SUBMISSION=5 \
/root/venv/bin/python -m reliquary.cli.main mine \
    --wallet-name camille81 --hotkey miner1 --network finney --netuid 81 \
    --checkpoint Qwen/Qwen3-4B-Instruct-2507 --log-level INFO \
    2>&1 | tee /root/reliquary-miner-priv/miner.log
