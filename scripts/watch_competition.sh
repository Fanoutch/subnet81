#!/bin/bash
# Vigie concurrence : rang de notre 1re entrée, heure de fermeture du batch,
# lag des gagnants du peloton. Alerte quand la course se durcit.
STATE=/tmp/claude-0/-root-subnet81/8859b6c2-555a-4b99-a451-f97bf060bb3b/scratchpad
mkdir -p "$STATE"
while true; do
  sleep 900
  scp -o ConnectTimeout=20 -P 20098 root@38.255.28.21:/workspace/submits_v4.jsonl /root/subnet81/data/submits_v4.jsonl >/dev/null 2>&1
  scp -o ConnectTimeout=20 -P 20098 root@38.255.28.21:/workspace/verdicts_v4.jsonl /root/subnet81/data/verdicts_v4.jsonl >/dev/null 2>&1
  M=$(curl -s --max-time 10 "https://www.reliqua.ai/api/miners" 2>/dev/null)
  echo "$M" > "$STATE/market_last.json"
  python3 /root/subnet81/scripts/competition_tick.py "$STATE"
done
