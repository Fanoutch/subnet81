#!/bin/bash
# Arrêt du mineur -> banc forced-seed (carte vide, durée PLAFONNÉE) -> relance.
# À lancer au trou 503, détaché :  setsid nohup bash /workspace/bench_then_restart.sh &
# Le mineur est TOUJOURS relancé, que le banc réussisse, échoue ou dépasse.
# Motif construit à l'exécution : la ligne de commande de ce script ne doit
# jamais correspondre au pkill (cf. restart_miner.sh).
LOG=/workspace/bench_fs_h100.log
BUDGET=${BENCH_BUDGET_S:-75}
PAT="cli.""main mine"
PID=$(pgrep -f "$PAT" | head -1)
{
echo "=== $(date -u +%FT%TZ) bench_then_restart (pid mineur ${PID:-absent}, budget ${BUDGET}s)"
ENVF=/tmp/miner_environ.$$
if [ -n "$PID" ]; then cp /proc/$PID/environ "$ENVF"; else : > "$ENVF"; fi
MODEL=$(grep -o "model='[^']*'" /workspace/miner.log | tail -1 | cut -d"'" -f2)
echo "modèle: $MODEL"
pkill -9 -f "$PAT" 2>/dev/null
pkill -9 -f EngineCore 2>/dev/null
sleep 4
nvidia-smi --query-gpu=memory.used --format=csv,noheader
if [ -n "$MODEL" ] && [ -s "$ENVF" ]; then
  mapfile -d '' ENVV < "$ENVF"
  # Passes : « libre+fs » (chemin actuel), puis « fs » avec RELIQUARY_FS_FAST=1.
  # Les empreintes tokens_sha des deux passes fs doivent être IDENTIQUES.
  for PASS in ${BENCH_PASSES:-"0:fs,libre" "1:fs"}; do
    FAST=${PASS%%:*}; MODES=${PASS#*:}
    cd /workspace/reliquary-miner-priv && env -i "${ENVV[@]}" PYTHONPATH=. \
      RELIQUARY_FS_FAST=$FAST BENCH_MODES=$MODES \
      BENCH_MODEL="$MODEL" BENCH_LEN=${BENCH_LEN:-512} BENCH_PROMPTS=10,16 \
      timeout -k 5 "$BUDGET" /workspace/venv/bin/python ops/bench_fs_h100.py
    echo "banc (FS_FAST=$FAST $MODES): code retour $?"
    pkill -9 -f EngineCore 2>/dev/null; sleep 3
  done
else
  echo "banc SAUTÉ (modèle ou environ introuvable)"
fi
pkill -9 -f EngineCore 2>/dev/null
rm -f "$ENVF"
echo "=== $(date -u +%FT%TZ) relance du mineur"
} >> "$LOG" 2>&1
bash /workspace/restart_miner.sh >> "$LOG" 2>&1
echo "=== $(date -u +%FT%TZ) relance faite" >> "$LOG"
