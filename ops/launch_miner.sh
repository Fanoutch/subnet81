#!/bin/bash
# Canonical miner launch script — rsync to the GPU box at /workspace/launch_miner.sh.
# All env gaps we hit on 2026-07-17/18 are baked in. Wallet camille81-v2/hotkey81
# is ALREADY registered — no re-registration on a fresh box, just copy the wallet
# (hotkey + coldkeypub ONLY, never the coldkey secret).
# Si la box ne joint pas le validateur en direct (egress bloque : observe sur
# la H100 93.120.231.186, SYN-SENT vers 209.20.157.231:8080), monter un tunnel
# inverse DEPUIS la dev box qui, elle, y a acces :
#   ssh -f -N -R 8080:209.20.157.231:8080 -p <port> root@<box>
# puis lancer avec RELIQUARY_VALIDATOR_URL=http://127.0.0.1:8080
export PYTHONPATH=/workspace/reliquary-miner-priv
export HF_HOME=/workspace/hf
export GRAIL_ATTN_IMPL=sdpa               # no flash_attn wheel for torch 2.11+cu130
export RELIQUARY_MAX_NEW_TOKENS=8192
export RELIQUARY_PROMPT_RANGE_FROM_WINDOW=0   # arm per-window prompt-range (validator enforces it)
# Dual-env. The validator opens a slice on BOTH envs every window and splits
# emissions ~50/50; the unmined half is BURNED, so math-only caps us at ~50%.
# Code is also far less contested (window 24308: math 44 distinct prompts taken
# vs code 2) and its reward is CONTINUOUS in [0,1] (passed/total test cases)
# rather than binary, so its sigma distribution differs from math's -- where we
# measured 0/238 payable. Dataset pin verified identical to the validator
# (R0mAI/opencodeinstruct-curated@d3caaefc). Safety: single-env behaviour is
# unchanged, so this cannot degrade the math path.
# Box H100 : egress bloque vers le validateur (SYN-SENT vers 209.20.157.231:8080).
# Tunnel inverse monte DEPUIS la dev box (qui, elle, y accede) :
#   ssh -f -N -R 8080:209.20.157.231:8080 -p <port> root@<box>
export RELIQUARY_VALIDATOR_URL=${RELIQUARY_VALIDATOR_URL:-}
export RELIQUARY_ACTIVE_ENVS=${RELIQUARY_ACTIVE_ENVS:-opencodeinstruct}
export RELIQUARY_VLLM_FORCED_SEED=1       # vLLM phase-1: gate valide 2026-07-21 sur v3 (groupe 0.9897, pire rollout 0.9375, planchers 0.80/0.75)
# Candidate throughput. Only groups with sigma>=0.43 (k in [2,6] of 8 rollouts)
# are payable, and observed hit rate is low — so we need MANY candidates per
# window, not one. Default 2 left the H200 at ~18% memory / 34% util.
# HARD CONSTRAINT: generation is bound to the window randomness (pool is flushed
# on flip), so a bake batch that overruns the 100s collection window is fully
# wasted. Tune upward only while batch wall-time stays under ~90s.
# ⚠️ H100 80 GiB : batch à re-mesurer (le KV cache est ~2x plus petit qu'en
# H200 143 GiB, où 25 tenait). Démarrer bas et monter en mesurant.
# CUDA graphs : +23% de debit a batch egal (banc H100 2026-07-22 : 2116 vs
# 1725 tok/s a 128 sequences), et mieux qu'un batch de 256 en eager.
# Conformite forced-seed VERIFIEE : seed_consistency 0.9880 groupe / 0.9531
# pire rollout (planchers validateur 0.80 / 0.75).
# ⚠️ Re-passer scripts/validate_vllm_forced_seed_group.py (GATE_EAGER=0) sur
# toute NOUVELLE carte avant de s'y fier — le defaut du code reste eager.
export RELIQUARY_VLLM_CUDA_GRAPHS=${RELIQUARY_VLLM_CUDA_GRAPHS:-1}
# OPTIMUM MESURE (banc H100 2026-07-22, CUDA graphs actifs) :
#   batch 16 = 2125 tok/s | 32 = 2181 (+2.6%) | 48 = 2202 (+0.9%)
# Tripler le batch ne rapporte que 3.6% : les graphes ont deja supprime le
# surcout par pas de decodage que le batch servait a amortir. On prend donc le
# PLUS PETIT batch au plateau — moins de sequences en vol = les prompts
# sortent plus tot dans la fenetre de 100 s (l'effet de position reste entier,
# la phase-2 BFT n'etant pas batchee).
export RELIQUARY_BAKE_BATCH_SIZE=${RELIQUARY_BAKE_BATCH_SIZE:-20}
# vLLM env (needed even with flag=0 if the CLI builds the backend; harmless otherwise):
# Proof forwards allocate ~8 GiB spikes next to vLLM's KV cache; expandable
# segments stop allocator fragmentation from turning that into an OOM
# (45 lost bakes measured 2026-07-21 at gpu_fraction=0.75).
# ⚠️ H100 80 GiB : 0.55 est la valeur ÉPROUVÉE sur cette carte (0.60 faisait
# échouer ~1 reload sur 5). Le 0.78 calibré H200 = OOM garanti ici.
export RELIQUARY_VLLM_GPU_FRACTION=${RELIQUARY_VLLM_GPU_FRACTION:-0.55}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_DEEP_GEMM=0
export VLLM_DEEP_GEMM_WARMUP=skip         # 0.24 enum: skip|full|relax (NOT 0/1)
export VLLM_USE_FLASHINFER_SAMPLER=0      # native pytorch sampler (flashinfer sampling ptxas fails: PTX 9.2 vs 9.0)
export CUDA_HOME=/workspace/venv/lib/python3.12/site-packages/nvidia/cu13
export PATH=/workspace/venv/bin:$CUDA_HOME/bin:$PATH
cd /workspace/reliquary-miner-priv
exec /workspace/venv/bin/python -m reliquary.cli.main mine \
  --wallet-name camille81-v2 --hotkey hotkey81 --network finney --netuid 81 \
  ${RELIQUARY_VALIDATOR_URL:+--validator-url $RELIQUARY_VALIDATOR_URL} \
  --checkpoint ReliquaryForge/qwen3.5-2b-reliquary-v3 --log-level INFO
