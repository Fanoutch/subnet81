#!/bin/bash
# Campagne banc H200 — ÉTAGE 2 (2026-08-14). À lancer APRÈS bench_sprint_matrix.sh
# (étage 1 = prompts × forced-seed). Ici : les paramètres secondaires, mesurés
# au voisinage de la config gagnante de l'étage 1.
#   BEST_PROMPTS=2 bash bench_sprint_matrix2.sh   (défaut 2 = sprint exclusif)
# Trois sweeps :
#   A. CUDA graphs ON/OFF au batch gagnant (juillet H100 : +48% — à revérifier
#      sur H200 à PETIT batch, là où le surcoût par step domine)
#   B. max_num_seqs 64 vs 512 (tailles de capture des graphs + scheduler)
#   C. courbe longueur : per-seq à 512/2048/8192 tokens — Qwen3.5 est hybride
#      Mamba, le coût par token devrait peu croître avec la longueur ; si ça
#      décroche, la stratégie "groupes longs" perd de sa valeur et il faut le
#      savoir AVANT de viser 40-75k tokens par groupe.
set -e
export PYTHONPATH=/workspace/reliquary-miner-priv
export HF_HOME=/workspace/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_DEEP_GEMM=0
export VLLM_DEEP_GEMM_WARMUP=skip
export VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_HOME=/workspace/venv/lib/python3.12/site-packages/nvidia/cu13
export PATH=/workspace/venv/bin:$CUDA_HOME/bin:$PATH
export GRAIL_ATTN_IMPL=sdpa
export SMOKE_CKPT=Qwen/Qwen3.5-4B
export SMOKE_REV=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
export BENCH_ENV=opencodeinstruct
export BENCH_GPU_FRAC=0.78
export BENCH_REPEATS=3
BP=${BEST_PROMPTS:-2}
P=/workspace/venv/bin/python
OUT=/workspace/bench_matrix2_$(date -u +%m%d_%H%M).log
run() { echo "##### $1 #####" | tee -a "$OUT"; shift; env "$@" BENCH_PROMPTS=${BENCH_PROMPTS_OVERRIDE:-$BP} $P /workspace/reliquary-miner-priv/ops/bench_tokens.py 2>&1 | grep -E '\[bench\]|toks_per_s|\{' | tee -a "$OUT"; }

# A. graphs
run "A graphs=ON  prompts=$BP" BENCH_EAGER=0 BENCH_MAX_TOKENS=1024 BENCH_MAX_NUM_SEQS=512
run "A graphs=OFF prompts=$BP" BENCH_EAGER=1 BENCH_MAX_TOKENS=1024 BENCH_MAX_NUM_SEQS=512
# B. max_num_seqs
run "B mns=64  prompts=$BP" BENCH_EAGER=0 BENCH_MAX_TOKENS=1024 BENCH_MAX_NUM_SEQS=64
run "B mns=512 prompts=$BP" BENCH_EAGER=0 BENCH_MAX_TOKENS=1024 BENCH_MAX_NUM_SEQS=512
# C. courbe longueur (per-seq vs longueur de séquence)
run "C len=512  prompts=$BP" BENCH_EAGER=0 BENCH_MAX_TOKENS=512  BENCH_MAX_NUM_SEQS=512 BENCH_MAX_MODEL_LEN=4096
run "C len=2048 prompts=$BP" BENCH_EAGER=0 BENCH_MAX_TOKENS=2048 BENCH_MAX_NUM_SEQS=512 BENCH_MAX_MODEL_LEN=8192
run "C len=8192 prompts=$BP" BENCH_EAGER=0 BENCH_MAX_TOKENS=8192 BENCH_MAX_NUM_SEQS=512 BENCH_MAX_MODEL_LEN=16384

echo "##### SYNTHÈSE (voir les blocs JSON toks_per_s_median ci-dessus) #####" | tee -a "$OUT"
echo "FIN — log: $OUT"
