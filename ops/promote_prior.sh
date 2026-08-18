#!/bin/bash
# Promoteur de prior (go utilisateur 18/08 : « auto au creux ») — tourne en
# tmux `promoter` sur la box. Si un candidat gagnant du duel nocturne est
# présent (/workspace/predictor_v50_candidate.json, déposé par le cron dev
# box), attend un MOMENT GRATUIT — validateur down/503 ou fenêtre non-open —
# puis promeut (swap + restart via la chaîne standard) sans sacrifier de slot.
set -u
CAND=/workspace/predictor_v50_candidate.json
LIVE=/workspace/predictor_v50.json
while true; do
  if [ -s "$CAND" ]; then
    POST=$(curl -s -o /dev/null -w "%{http_code}" --max-time 6 -X POST \
      -H "Content-Type: application/json" -d '{}' \
      http://209.20.157.231:8080/submit/precommit 2>/dev/null)
    STATE=$(curl -s --max-time 6 "http://209.20.157.231:8080/state?env=opencodeinstruct" 2>/dev/null \
      | python3 -c "import json,sys;print(json.load(sys.stdin).get('state',''))" 2>/dev/null)
    if [ "$POST" = "503" ] || [ "$POST" = "000" ] || { [ -n "$STATE" ] && [ "$STATE" != "open" ]; }; then
      echo "$(date -u +%F' '%H:%M) PROMOTION: creux détecté (POST=$POST state=$STATE) — swap + restart"
      cp "$LIVE" "${LIVE}.prev" 2>/dev/null
      mv "$CAND" "$LIVE"
      bash /workspace/restart_miner.sh
      sleep 5
      tmux new-session -d -s watchdog81 "bash /workspace/watchdog.sh 2>&1 | tee -a /workspace/watchdog.log" 2>/dev/null
      echo "$(date -u +%F' '%H:%M) PROMOTION FAITE (ancien → ${LIVE}.prev)"
    fi
  fi
  sleep 60
done
