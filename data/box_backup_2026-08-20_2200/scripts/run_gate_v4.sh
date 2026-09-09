#!/bin/bash
# Gate forced-seed v4 sur CETTE carte — contrôle OBLIGATOIRE avant de miner.
# Elle vérifie que nos tokens sont ceux que le validateur reconstruira par
# teacher-forcing. Planchers protocole : 0.80 groupe / 0.75 pire rollout.
# ⚠️ On ne source PAS bench_env.sh : il force SMOKE_CKPT sur le 2B périmé.
export PYTHONPATH=/workspace/reliquary-miner-priv
export HF_HOME=/workspace/hf
export GRAIL_ATTN_IMPL=sdpa
export CUDA_HOME=/workspace/venv/lib/python3.12/site-packages/nvidia/cu13
export PATH=/workspace/venv/bin:$CUDA_HOME/bin:$PATH
export VLLM_USE_DEEP_GEMM=0
export VLLM_DEEP_GEMM_WARMUP=skip
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RELIQUARY_PROTOCOL_VERSION=4
unset SMOKE_CKPT SMOKE_REV          # -> modèle par défaut du protocole v4
export GATE_MAX_NUM_SEQS=64         # >64 dépasse les blocs Mamba du 4B
P=/workspace/venv/bin/python

echo "##### GATE EAGER (référence de conformité) #####"
GATE_EAGER=1 $P /workspace/validate_vllm_forced_seed_group.py 2>&1 \
  | grep -aE "\[gate\]|model|Error|Traceback|Assertion"
echo
echo "##### GATE CUDA GRAPHS (le mode de production) #####"
GATE_EAGER=0 $P /workspace/validate_vllm_forced_seed_group.py 2>&1 \
  | grep -aE "\[gate\]|model|Error|Traceback|Assertion"
echo "GATES_TERMINEES"
