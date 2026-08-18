#!/bin/bash
# Restart the miner. Kills via a pattern built at runtime so this script's OWN
# command line never matches it (an inline `pkill -f "cli.main mine"` over ssh
# matches the ssh argv itself and kills the restarting shell).
PAT="cli.""main mine"
pkill -9 -f "$PAT" 2>/dev/null
pkill -9 -f EngineCore 2>/dev/null
sleep 3
echo "survivants: $(ps -eo cmd | grep -c "[c]li.main mine")"
tmux kill-server 2>/dev/null
sleep 1
# Balayage v4 G1 : le launcher est résolu via le marqueur persistant
# /workspace/.miner_launcher (posé au jour J de la bascule v4) — sans lui,
# fallback = launch_miner.sh historique (v3, comportement inchangé). Un
# restart/watchdog ne doit JAMAIS relancer un protocole différent de celui
# qui tournait.
LAUNCHER="$(cat /workspace/.miner_launcher 2>/dev/null || echo /workspace/launch_miner.sh)"
tmux new-session -d -s miner "bash $LAUNCHER 2>&1 | tee /workspace/miner.log"
sleep 6
echo "tmux: $(tmux ls 2>&1)"
echo "proc: $(ps -eo pid,cmd | grep "[c]li.main mine" | head -1)"
