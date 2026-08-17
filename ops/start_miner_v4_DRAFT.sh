#!/usr/bin/env bash
# DRAFT v4 — NE PAS LANCER tant que /health du validateur n'annonce pas
# protocol_version 4 / generation_profile_id qwen3-4b-base-dapo-v4.
#   curl -s http://209.20.157.231:8080/health
# Bascule : RELIQUARY_PROTOCOL_VERSION=4 fait TOUT côté code (constantes,
# domaine forced-seed v4, BFT off, cap 8192, 16 rollouts, sampling 1.0/1.0/0,
# manifest OMI train-*, grader Answer:). Ce script n'ajoute que l'env + le
# checkpoint de départ.
#
# Jour J, AVANT lancement (cf. plan docs/superpowers/plans/2026-08-17-port-v4-dapo.md) :
#   1. Re-passer la gate forced-seed GPU sur Qwen3-4B-Base (full-support) :
#      RELIQUARY_PROTOCOL_VERSION=4 ops/validate_vllm_forced_seed_group.py
#   2. Vérifier la révision modèle épinglée par /health (attendu :
#      Qwen/Qwen3-4B-Base @ 906bfd4b4dc7f14ee4320094d8b41684abff8539) et le
#      checkpoint suivi via /state.
#   3. Réconcilier les fixes OOM non commités de l'arbre prod
#      (_kill_stale_engine_cores + watchdog v2.1) dans cette branche.
# CHECKLIST jour J (audit 2026-08-17, items ops) :
#   - NE PAS reprendre de start_miner_h200.sh : RELIQUARY_K_MIN=2/K_MAX=6
#     (écraseraient les défauts v4 1/15), RELIQUARY_PROMPT_PREDICTOR(_2)
#     (priors du monde v3 — laisser le fallback uniforme, garder SAMPLE_DUMP
#     pour collecter des labels v4 → prédicteur v5), SPRINT_SIZE=2/
#     SPRINT_MAX_WAIT_S=90 (90 s = 60 % d'une fenêtre de 150 s ; re-bencher :
#     WAIT ≲ 10-15 s, SIZE 1-2 — à M=16, SIZE=4 = 64 séquences en vol).
#   - NE PAS poser RELIQUARY_MIN_LOCAL_Q10=0.05 ni MIN_LOCAL_MEDIAN v3 :
#     seuils v4 = q10 0.0002 / médiane 0.05 (marges sûres ~0.0005 / 0.08).
#   - Probe : RELIQUARY_PROTOCOL_VERSION=4 … --expect-protocol 4 --temperature 1.0
set -euo pipefail

export RELIQUARY_PROTOCOL_VERSION=4
# Ceinture (défaut déjà 0.0 sous v4 — audit item 3) :
export RELIQUARY_AUCTION_MIN_SCORE="${RELIQUARY_AUCTION_MIN_SCORE:-0}"
# Le checkpoint réel est suivi dynamiquement via /state ; point de départ v4 :
CHECKPOINT="${CHECKPOINT:-Qwen/Qwen3-4B-Base}"

# Sanity pré-lancement : les constantes doivent refléter le contrat v4.
PYTHONPATH=. python3 - <<'EOF'
from reliquary import constants as c
assert c.PROTOCOL_VERSION == 4, c.PROTOCOL_VERSION
assert c.M_ROLLOUTS == 16 and not c.BFT_ENABLED
assert c.MAX_NEW_TOKENS_PROTOCOL_CAP == 8192
assert (c.T_PROTO, c.TOP_P_PROTO, c.TOP_K_PROTO) == (1.0, 1.0, 0)
assert c.FORCED_SEED_DOMAIN == "reliquary-forced-seed-v4"
assert c.GENERATION_PROFILE_ID == "qwen3-4b-base-dapo-v4", c.GENERATION_PROFILE_ID
assert c.RAW_COMPLETION_PROMPTS and c.OMI_TRAIN_SHARDS_ONLY
print("v4 constants OK:", c.FORCED_SEED_DOMAIN)
EOF

cd "$(dirname "$0")/.." && PYTHONPATH=. \
  python3 -m reliquary.cli.main mine \
    --wallet-name camille81-v2 --hotkey hotkey81 --network finney --netuid 81 \
    --checkpoint "$CHECKPOINT" \
    --log-level INFO 2>&1 | tee miner_v4.log
