#!/bin/bash
# Watchdog mineur — né de l'incident du 2026-08-06 22:42 : une course
# reload-checkpoint × bake en vol a tué la génération ; le process restait
# VIVANT donc rien ne l'a vu — 11 h de production nulle (~45 fenêtres).
# Règle : process présent MAIS aucun groupe généré depuis 15 min => restart.
# 15 min > pire cycle post-reload observé (408 s) + marge : zéro faux positif
# attendu en fonctionnement normal.
LOG=/workspace/miner.log
WLOG=/workspace/watchdog.log
while true; do
  sleep 60
  pgrep -f "reliquary.cli.main mine" >/dev/null || continue   # pas de process = pas notre affaire
  last=$(grep -oE "^2026-[0-9-]+ [0-9:]+" <(tail -c 400000 "$LOG" | grep "stream_fire: groupe") | tail -1)
  [ -z "$last" ] && continue                                   # démarrage : laisser charger
  last_s=$(date -d "$last" +%s 2>/dev/null) || continue
  now_s=$(date +%s)
  age=$(( now_s - last_s ))
  if [ "$age" -gt 900 ]; then
    echo "$(date -u +%FT%TZ) WEDGE détecté (dernier groupe il y a ${age}s) — restart" >> "$WLOG"
    tmux kill-session -t miner 2>/dev/null
    pkill -f "reliquary.cli.main mine"; sleep 8
    tmux new-session -d -s miner "bash /workspace/start_miner.sh 2>&1 | tee -a $LOG"
    sleep 600   # laisser le redémarrage aboutir avant de resurveiller
  fi
done
