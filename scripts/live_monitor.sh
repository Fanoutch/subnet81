#!/bin/bash
# Moniteur live v4 : CHAQUE fenêtre — envois (offsets) puis seal (rangs, payées, échecs).
STATE=/tmp/claude-0/-root-subnet81/f9183f14-00bf-4224-9c73-17c52762fc26/scratchpad/monitor_state
mkdir -p "$STATE"
while true; do
  scp -o ConnectTimeout=15 -P 20098 root@38.255.28.21:/workspace/submits_v4.jsonl /root/subnet81/data/submits_v4.jsonl >/dev/null 2>&1
  scp -o ConnectTimeout=15 -P 20098 root@38.255.28.21:/workspace/verdicts_v4.jsonl /root/subnet81/data/verdicts_v4.jsonl >/dev/null 2>&1
  python3 /root/subnet81/scripts/live_monitor_tick.py "$STATE"
  NERR=$(ssh -o ConnectTimeout=15 -p 20098 root@38.255.28.21 'grep -cE "ERROR|Traceback" /workspace/miner.log' 2>/dev/null || echo "")
  if [ -n "$NERR" ]; then
    PREV=$(cat "$STATE/err_count" 2>/dev/null || echo 0)
    if [ "$NERR" -gt "$PREV" ] 2>/dev/null; then
      DELTA=$((NERR-PREV))
      LINE=$(ssh -o ConnectTimeout=15 -p 20098 root@38.255.28.21 'grep -E "ERROR|Traceback" /workspace/miner.log | grep -v "<" | tail -1 | cut -c1-120' 2>/dev/null)
      echo "erreurs mineur +$DELTA: $LINE"
    fi
    echo "$NERR" > "$STATE/err_count"
  fi
  sleep 60
done
