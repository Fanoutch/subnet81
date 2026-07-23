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
tmux new-session -d -s miner "bash /workspace/launch_miner.sh 2>&1 | tee /workspace/miner.log"
sleep 6
echo "tmux: $(tmux ls 2>&1)"
echo "proc: $(ps -eo pid,cmd | grep "[c]li.main mine" | head -1)"
