#!/bin/bash
# Watchdog mineur (incident 2026-08-06 : reload x bake a tué la géné, process VIVANT, 11h nul).
# Restart si process présent MAIS aucun 'stream_fire' depuis 15 min.
# GARDE (doc §6) : ignorer si le process a <15 min d'âge (sinon tue un mineur en warmup
# en lisant un vieux stream_fire du log tee -a).
LOG=/workspace/miner.log; WLOG=/workspace/watchdog.log
echo "$(date -u +%FT%TZ) watchdog démarré" >> "$WLOG"
while true; do
  sleep 60
  mpid=$(pgrep -f "reliquary.cli.main mine" | head -1) || continue
  [ -z "$mpid" ] && continue
  page=$(ps -o etimes= -p "$mpid" 2>/dev/null | tr -d " ")
  { [ -n "$page" ] && [ "$page" -lt 900 ]; } && continue   # process <15min : warmup, ignorer
  last=$(grep -oE "^2026-[0-9-]+ [0-9:]+" <(tail -c 400000 "$LOG" | grep "stream_fire: groupe") | tail -1)
  [ -z "$last" ] && continue
  last_s=$(date -d "$last" +%s 2>/dev/null) || continue
  age=$(( $(date +%s) - last_s ))
  if [ "$age" -gt 900 ]; then
    echo "$(date -u +%FT%TZ) WEDGE (dernier groupe ${age}s, proc ${page}s) — restart" >> "$WLOG"
    tmux kill-session -t miner 2>/dev/null
    pkill -9 -f "reliquary.cli.main mine"; pkill -9 -f EngineCore; sleep 8
    tmux new-session -d -s miner "bash /workspace/start_miner.sh 2>&1 | tee -a $LOG"
    sleep 600
  fi
done
