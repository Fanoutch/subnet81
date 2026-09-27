#!/bin/bash
# Superviseur du mineur SFT (27/09). Relance après un plantage, JAMAIS après
# un refus définitif du validateur ni si /workspace/SFT_STOP existe (posé par
# la vigie de la dev box au premier échec d audit confirmé).
export HF_HOME=/workspace/hf VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_V2_MODEL_RUNNER=0
export RELIQUARY_TASK_CONTRACT=/workspace/corpus-code-v1.contract.json PYTHONPATH=/workspace/sft_src
cd /workspace/reliquary_upstream
while true; do
  [ -e /workspace/SFT_STOP ] && { echo "$(date -u +%FT%T) SFT_STOP présent, arrêt"; exit 0; }
  echo "$(date -u +%FT%T) === démarrage du mineur SFT"
  /workspace/venv_corpus/bin/python -m sft_miner.miner --wallet-name camille81-v2 --hotkey hotkey81.2 \
      --max-in-flight 32 --max-ahead 256 --admit-below 0.6 2>&1 | tee -a /workspace/sft_run.log
  rc=${PIPESTATUS[0]}
  echo "$(date -u +%FT%T) === sortie rc=$rc"
  if tail -n 50 /workspace/sft_run.log | grep -qE "arrêt : hotkey_not_registered"; then
    NR=$((${NR:-0}+1))
    # hotkey tout juste (ré)enregistrée : le validateur relit les
    # enregistrements toutes les 10 min. 3 essais, puis abandon.
    [ "$NR" -gt 3 ] && { echo "$(date -u +%FT%T) toujours non enregistrée après 3 essais, arrêt"; exit 0; }
    echo "$(date -u +%FT%T) hotkey pas encore vue par le validateur, nouvel essai dans 10 min ($NR/3)"
    sleep 600; continue
  fi
  if tail -n 50 /workspace/sft_run.log | grep -qE "arrêt : (job_complete|miner_banned)"; then
    echo "$(date -u +%FT%T) arrêt définitif, pas de relance"; exit 0
  fi
  sleep 30
done
