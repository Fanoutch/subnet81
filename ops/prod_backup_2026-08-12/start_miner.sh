#!/bin/bash
# PRODUCTION : filtres sigma RÉTABLIS (on ne soumet que du payable).
# Le mode diagnostic (start_miner_diag.sh) désactivait les DEUX seuils sigma
# — celui de _skip_for_out_of_zone ET celui du sélecteur code (0.43+marge) —
# ce qui brûlait les 8 places de la fenêtre avec des groupes non payables.
export RELIQUARY_VALIDATOR_URL=http://209.20.157.231:8080
export RELIQUARY_SAMPLE_DUMP=/workspace/samples_code.jsonl
export RELIQUARY_SPRINT_MAX_WAIT_S=12   # un groupe court livre en 5-14s ; au-dela de 12s le traînard ne finira pas vite — le balayage entre, le traînard continue en partage
export RELIQUARY_TRUNC_DIAG=/workspace/trunc_diag.jsonl
# RÉACTIVÉ 2026-08-06 avec le modèle k2-RÉALISÉ (remplace le v0 dureté-pure,
# qui choisissait des prompts durs qui TRONQUENT : intacts 12.7% -> 2.2%).
# Cible du nouveau modèle = valeur d enchère réalisée au cap 2600 (0 si un
# rollout déborde) : il vise dur ET terminé. Si le cap monte a 16384 ->
# repasser sur prompt_predictor_v1_full.json (dureté).
export RELIQUARY_PROMPT_PREDICTOR=/workspace/predictor_v2_A_inzone_clean.json  # candidate A (labels cap-invariants), config validée 2026-08-11
export RELIQUARY_BAKE_BATCH_SIZE=16      # cycle 65 s, ~4,6 cycles/fenêtre
export RELIQUARY_BAKE_CHUNK=64           # >= batch : un seul appel vLLM (concurrence max)
export VENV=/workspace/venv
export RELIQUARY_MAX_NEW_TOKENS=8192  # config validée 2026-08-11 (rangs 1-8 payés)
# export RELIQUARY_PROMPT_PARITY=0  # DÉSACTIVÉ (H200 morte dans la nuit) — réactiver SEULEMENT quand la H200 re-mine réellement
export RELIQUARY_AUCTION_MIN_SCORE=0   # flat auction live (0a5aeaf): envoyer k in [2,6], pas que k=2
exec bash /workspace/reliquary-miner-priv/ops/launch_miner.sh
