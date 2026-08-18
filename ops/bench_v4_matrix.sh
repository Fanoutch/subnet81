#!/bin/bash
# Matrice débit v4 (2026-08-18) — ordonnée par priorité : si le temps de
# location manque, les réponses les plus importantes sont déjà tombées.
# Socle = ops/bench_tokens.py (mesure sur appels TERMINÉS — jamais la barre
# vLLM). Tout tourne sous le contrat v4 : Qwen3-4B-Base@906bfd4b, M=16,
# T1.0/top_p1.0/top_k0, contexte 9216.
# Usage : bash ops/bench_v4_matrix.sh [P1|P2|P3|P4]   (défaut : tout)
set -uo pipefail
cd "$(dirname "$0")/.."
source /workspace/venv/bin/activate 2>/dev/null || true
export HF_HOME=${HF_HOME:-/workspace/hf}
export VLLM_USE_DEEP_GEMM=0
export CUDA_HOME=${CUDA_HOME:-/workspace/venv/lib/python3.12/site-packages/nvidia/cu13}
export PYTHONPATH=.

export RELIQUARY_PROTOCOL_VERSION=4
export SMOKE_CKPT="Qwen/Qwen3-4B-Base"
export SMOKE_REV="906bfd4b4dc7f14ee4320094d8b41684abff8539"
export BENCH_ROLLOUTS=16
export BENCH_MAX_MODEL_LEN=9216
export BENCH_GPU_FRAC=${BENCH_GPU_FRAC:-0.76}
export BENCH_REPEATS=${BENCH_REPEATS:-3}

ONLY="${1:-ALL}"
run() { echo; echo "───── $* ─────"; "$@" || echo "ÉCHEC (on continue)"; }
stage() { [ "$ONLY" = "ALL" ] || [ "$ONLY" = "$1" ]; }

# ── P1 — LA grande inconnue : coût du forced-seed en FULL-SUPPORT ──────────
# v3 : le processeur FS coûtait 9,7× (tri top-20 + CDF). v4 : top_k=0/top_p=1
# → PLUS DE TRI top_p (garde court-circuitée) mais CDF sur le vocab PLEIN
# (151k). Personne ne sait lequel gagne. Sprint réaliste = 2 prompts × 16 = 32 seqs.
if stage P1; then
  echo "===== P1 : forced-seed ON vs OFF (2 prompts × M=16, 1024 tok) ====="
  run env BENCH_PROMPTS=2 BENCH_MAX_TOKENS=1024 BENCH_EAGER=1 BENCH_NO_FORCED=0 python ops/bench_tokens.py
  run env BENCH_PROMPTS=2 BENCH_MAX_TOKENS=1024 BENCH_EAGER=1 BENCH_NO_FORCED=1 python ops/bench_tokens.py
fi

# ── P2 — per-seq × prompts en vol → fixe SPRINT_SIZE (fenêtre 150 s) ───────
if stage P2; then
  echo "===== P2 : per-seq à {1,2,4} prompts en vol (16/32/64 seqs), FS ON ====="
  for NP in 1 2 4; do
    run env BENCH_PROMPTS=$NP BENCH_MAX_TOKENS=1024 BENCH_EAGER=1 python ops/bench_tokens.py
  done
fi

# ── P3 — spécifique modèle DENSE (ex-impossible sur l'hybride Mamba) ───────
# CUDA graphs : +48 % mesuré v3 — à re-confirmer sur le 4B-Base dense.
# Prefix caching : 16 rollouts partagent le MÊME préfixe prompt — sur un
# dense c'est du KV réutilisable (état Mamba ne l'était pas).
if stage P3; then
  echo "===== P3 : CUDA graphs ON/OFF × prefix caching ON/OFF (2 prompts) ====="
  for EAGER in 1 0; do
    for PFX in 0 1; do
      run env BENCH_PROMPTS=2 BENCH_MAX_TOKENS=1024 BENCH_EAGER=$EAGER BENCH_PREFIX_CACHE=$PFX python ops/bench_tokens.py
    done
  done
fi

# ── P4 — courbe longueur (attention dense = coût croît avec la longueur ; ──
# les groupes longs paient sous le rang tokens/rounds si régime plat)
if stage P4; then
  echo "===== P4 : courbe longueur 512 / 2048 / 8192 (config gagnante P3) ====="
  for MT in 512 2048 8192; do
    run env BENCH_PROMPTS=2 BENCH_MAX_TOKENS=$MT BENCH_REPEATS=2 BENCH_EAGER=${P4_EAGER:-1} BENCH_PREFIX_CACHE=${P4_PFX:-0} python ops/bench_tokens.py
  done
fi

echo; echo "===== FIN — reporter les gagnants dans ops/launch_miner_v4.sh ====="
echo "  (SPRINT_SIZE via P2 ; BENCH_EAGER/PREFIX gagnants via P3 → flags vLLM ;"
echo "   si FS OFF ≫ FS ON en P1 → chantier kernel CDF plein-vocab, cf. runbook)"
