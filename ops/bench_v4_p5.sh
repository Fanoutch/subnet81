#!/bin/bash
# P5 — leviers NOUVEAUX ouverts par le modèle DENSE (fermés sur l'hybride
# Mamba v3). À lancer APRÈS bench_v4_matrix.sh (fichier séparé exprès : ne
# jamais écraser un script bash en cours d'exécution).
# P5a : scaling max_num_seqs — le Mamba plafonnait à 64 seqs (blocs d'état) ;
#       le dense n'a que le KV-cache comme limite → 8 prompts × 16 = 128 seqs,
#       puis 256 en vol. (v3 : 128 seqs = le régime des 10 961 tok/s.)
# P5b : backend d'attention vLLM — sur le Mamba on était verrouillés
#       gdn_prefill_backend=triton ; le dense laisse choisir
#       (défaut vs FLASH_ATTN vs FLASHINFER).
# ⚠️ TOUJOURS INTERDITS (protocole, ne pas re-tester) : décodage spéculatif
# (tokens forcés par u_at), quantization ET kv_cache fp8 (casseraient
# seed_consistency — tout changement de logits doit repasser le gate parité).
set -uo pipefail
cd "$(dirname "$0")/.."
source /workspace/venv/bin/activate 2>/dev/null || true
source ops/bench_env.sh
export PYTHONPATH=.

export RELIQUARY_PROTOCOL_VERSION=4
export SMOKE_CKPT="Qwen/Qwen3-4B-Base"
export SMOKE_REV="906bfd4b4dc7f14ee4320094d8b41684abff8539"
export BENCH_ROLLOUTS=16
export BENCH_MAX_MODEL_LEN=9216
export BENCH_GPU_FRAC=${BENCH_GPU_FRAC:-0.76}
export BENCH_REPEATS=${BENCH_REPEATS:-2}

run() { echo; echo "───── $* ─────"; "$@" || echo "ÉCHEC (on continue)"; }

echo "===== P5a : scaling seqs en vol — 8 prompts (128 seqs) puis max_num_seqs ====="
run env BENCH_PROMPTS=8 BENCH_MAX_TOKENS=1024 BENCH_EAGER=1 python ops/bench_tokens.py
run env BENCH_PROMPTS=8 BENCH_MAX_TOKENS=1024 BENCH_EAGER=1 BENCH_MAX_NUM_SEQS=256 python ops/bench_tokens.py
run env BENCH_PROMPTS=16 BENCH_MAX_TOKENS=1024 BENCH_EAGER=1 BENCH_MAX_NUM_SEQS=256 python ops/bench_tokens.py

echo "===== P5b : backend attention (défaut vs FLASH_ATTN vs FLASHINFER), 4 prompts ====="
run env BENCH_PROMPTS=4 BENCH_MAX_TOKENS=1024 BENCH_EAGER=1 python ops/bench_tokens.py
run env BENCH_PROMPTS=4 BENCH_MAX_TOKENS=1024 BENCH_EAGER=1 VLLM_ATTENTION_BACKEND=FLASH_ATTN python ops/bench_tokens.py
run env BENCH_PROMPTS=4 BENCH_MAX_TOKENS=1024 BENCH_EAGER=1 VLLM_ATTENTION_BACKEND=FLASHINFER python ops/bench_tokens.py

echo "===== P5c : le combo gagnant P3 (graphs/prefix) au meilleur NP de P5a ====="
run env BENCH_PROMPTS=${P5C_NP:-8} BENCH_MAX_TOKENS=1024 BENCH_EAGER=${P5C_EAGER:-0} BENCH_PREFIX_CACHE=${P5C_PFX:-1} BENCH_MAX_NUM_SEQS=${P5C_MNS:-256} python ops/bench_tokens.py

echo "===== FIN P5 ====="
