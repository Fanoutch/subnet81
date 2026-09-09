#!/bin/bash
# Surveillance continue du mineur v4 (tmux `monitor` sur la box) : un digest
# toutes les 10 min via rapport_v4.py + sonde du chemin POST validateur.
cd /workspace/reliquary-miner-priv
while true; do
  echo "════════════════ $(date -u +%H:%M) UTC ════════════════"
  POST=$(curl -s -o /dev/null -w "%{http_code}" --max-time 8 -X POST \
    -H "Content-Type: application/json" -d '{}' \
    http://209.20.157.231:8080/submit/precommit 2>/dev/null)
  echo "chemin POST validateur: HTTP ${POST:-mort} (422=ouvert, 503=fermé)"
  python3 ops/rapport_v4.py 2>/dev/null | grep -vE "^$"
  sleep 600
done
