#!/bin/bash
# H200 — 2026-08-12. Base = config VALIDÉE de la H100 du 08-11 (cap 8192 +
# prédicteur in-zone, rangs 1-8 payés sous le classement live : #173 tokens/
# round par rollout + #171 flat auction). Deltas H200 uniquement :
#   • RELIQUARY_VALIDATOR_URL via tunnel (egress direct filtré sur cette box) :
#       ssh -f -N -R 8080:209.20.157.231:8080 -p 40500 root@93.120.231.186
#     (monté en tmux `tunnel81` sur la dev box, reconnexion auto)
#   • GPU_FRACTION 0.78 (calibré H200 143 GiB ; 0.55 = valeur H100 du launcher)
#   • prédicteur v3 (A_8192, entraîné 2026-08-12 sur les labels natifs cap-8192
#     de la H100 : held-out 46.5% payable&propre vs 31.8% v2_A, tronqués 2.9%)
# Garde anti-course d'init (incident 11:10-11:19 le 2026-08-13) : au restart,
# un EngineCore zombie peut tenir la VRAM plusieurs minutes → l'init vLLM voit
# 10 Go libres au lieu de 109 et brûle ses 5 tentatives. On attend le GPU vide.
for i in $(seq 1 24); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ -z "$used" ] || [ "$used" -lt 2000 ] && break
  echo "start_miner: GPU encore occupé (${used} MiB), attente ($i/24)..."
  sleep 5
done
export RELIQUARY_VALIDATOR_URL=http://127.0.0.1:8080
export RELIQUARY_SAMPLE_DUMP=/workspace/samples_code.jsonl
export RELIQUARY_SPRINT_SIZE=2         # 2 prompts seuls sur le GPU (16 seqs) : per-seq ~102 → ~180-250 tok/s attendus, plafond bucket ~45
export RELIQUARY_SPRINT_MAX_WAIT_S=90   # sprint EXCLUSIF (2026-08-13) : laisser les vedettes finir seules avant le balayage
export RELIQUARY_TRUNC_DIAG=/workspace/trunc_diag.jsonl
# v4 (2026-08-13) : cible = Σ tokens réalisés du groupe (numérateur du rang
# #173), 0 si non-soumissible. Held-out 8192 : numérateur moyen du pick 2933
# vs 1900 (v3), payable&propre 36.1% vs 24.8%. Labels cap-8192 → v4.1 à
# recalibrer sur les dumps cap-16384 dès qu'il y a du volume.
export RELIQUARY_PROMPT_PREDICTOR=/workspace/predictor_v41_bucket.json
export RELIQUARY_BAKE_BATCH_SIZE=16      # choix utilisateur — la H200 montait encore linéairement à 200 séq (banc 07-22)
export RELIQUARY_BAKE_CHUNK=64           # >= batch : un seul appel vLLM
# CAP AU MAX CONTRAT (2026-08-13) : le classement (#173) crédite jusqu'à 15616
# tokens PAR ROLLOUT (plafond groupe 124 928). Nuit du 12→13 à cap 8192 : nos
# groupes soumis ~7 200 tokens médians, arrivée excellente (27 s) mais rangs
# 20-60 → le numérateur avait pris le pas sur l'arrivée. 16384 (max contrat)
# plutôt que 15616 : la marge convertit des tronqués en terminés soumissibles.
export RELIQUARY_MAX_NEW_TOKENS=16384
export RELIQUARY_AUCTION_MIN_SCORE=0     # flat auction live : envoyer tout k in [2,6]
export RELIQUARY_VLLM_GPU_FRACTION=0.78
export VENV=/workspace/venv
exec bash /workspace/reliquary-miner-priv/ops/launch_miner.sh
