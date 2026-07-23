#!/bin/bash
# Env commun au banc (identique au lancement prod, hors validateur).
export PYTHONPATH=/workspace/reliquary-miner-priv
export HF_HOME=/workspace/hf
export GRAIL_ATTN_IMPL=sdpa
export CUDA_HOME=/workspace/venv/lib/python3.12/site-packages/nvidia/cu13
export PATH=/workspace/venv/bin:$CUDA_HOME/bin:$PATH
export VLLM_USE_DEEP_GEMM=0
export VLLM_DEEP_GEMM_WARMUP=skip
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SMOKE_CKPT=ReliquaryForge/qwen3.5-2b-reliquary-v3
export SMOKE_REV=5db7a1f55218c68fd3dff1d927b02c7508684c9e
