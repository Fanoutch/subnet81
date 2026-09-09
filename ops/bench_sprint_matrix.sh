#!/bin/bash
# Campagne banc H200 (2026-08-14) — AVANT relance du mining.
# Objectif : chiffrer la vitesse PAR SÉQUENCE selon (nb de prompts en vol ×
# forced-seed on/off), car le rang #173 = tokens_du_groupe ÷ rounds, et son
# plafond = vitesse_par_seq × 8 × 3/50. Mesures de référence :
#   102 tok/s/seq à ~32 seqs (sprint 4, cap 8192) ; ~70 sur H100.
# La colonne NO_FORCED donne le plafond matériel : l'écart FS→noFS à petit
# batch = le surcoût par step du processeur forced-seed (candidat kernel).
# Mode CONTRÔLÉ (longueur fixe, ignore_eos) : compare des configs, pas des
# cartes. Durée totale ~45-60 min (10 configs × 3 répétitions + rebuilds).
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
export BENCH_MAX_NUM_SEQS=512
export BENCH_EAGER=0            # CUDA graphs ON = config prod
export BENCH_MAX_TOKENS=1024    # assez long pour le régime stationnaire de décodage
export BENCH_REPEATS=3
P=/workspace/venv/bin/python
OUT=/workspace/bench_matrix_$(date -u +%m%d_%H%M).log
echo "matrice: prompts ∈ {1,2,4,8,16} × forced ∈ {on,off} — sortie $OUT"
for NP in 1 2 4 8 16; do
  for NF in 0 1; do
    echo "##### prompts=$NP (seqs=$((NP*8))) forced=$([ $NF = 0 ] && echo ON || echo OFF) #####" | tee -a "$OUT"
    BENCH_PROMPTS=$NP BENCH_NO_FORCED=$NF $P /workspace/reliquary-miner-priv/ops/bench_tokens.py 2>&1 | grep -E '\[bench\]|toks_per_s|\{' | tee -a "$OUT"
  done
done
echo "##### SYNTHÈSE #####" | tee -a "$OUT"
$P - "$OUT" << 'PYEOF' | tee -a "$OUT"
import json, re, sys
rows=[]
cfg=None
for line in open(sys.argv[1]):
    m=re.match(r'##### prompts=(\d+) \(seqs=(\d+)\) forced=(\w+)', line)
    if m: cfg=(int(m.group(1)), int(m.group(2)), m.group(3)); continue
    if line.strip().startswith('{'):
        try: d=json.loads(line)
        except: continue
        if cfg and 'toks_per_s_median' in d:
            rows.append((*cfg, d['toks_per_s_median']))
print(f"{'prompts':>7s} {'seqs':>5s} {'forced':>6s} {'agrégé tok/s':>13s} {'par-seq':>8s}")
for np_,seqs,forced,agg in rows:
    print(f"{np_:7d} {seqs:5d} {forced:>6s} {agg:13.0f} {agg/seqs:8.1f}")
by={}
for np_,seqs,forced,agg in rows: by[(np_,forced)]=agg
print("\nsurcoût forced-seed par config:")
for np_ in (1,2,4,8,16):
    on=by.get((np_,'ON')); off=by.get((np_,'OFF'))
    if on and off: print(f"  prompts={np_}: FS coûte {100*(1-on/off):.0f}% (ON {on:.0f} / OFF {off:.0f})")
PYEOF
echo "FIN — log: $OUT"
