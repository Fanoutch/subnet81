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
export RELIQUARY_PROMPT_PREDICTOR=/workspace/prompt_predictor_v11.json  # v1.1 : meme cible, 10721 prompts (3x), x4.3 vrais k=2 top5% held-out (v1 en rollback)
export RELIQUARY_BAKE_BATCH_SIZE=16      # cycle 65 s, ~4,6 cycles/fenêtre
export RELIQUARY_BAKE_CHUNK=64           # >= batch : un seul appel vLLM (concurrence max)
export VENV=/workspace/venv
exec bash /workspace/reliquary-miner-priv/ops/launch_miner.sh
