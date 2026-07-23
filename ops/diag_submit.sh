#!/bin/bash
# TEMPORARY DIAGNOSTIC — do not leave running.
# Lowers the local sigma gate to 0 so a real group reaches the production submit
# path (precommit -> reveal -> verdict), which has never run in prod. Expected
# verdict: OUT_OF_ZONE (rejected before the GRAIL proof path, so no
# expensive-proof budget is consumed). Restore with restart_miner.sh.
PAT="cli.""main mine"
pkill -9 -f "$PAT" 2>/dev/null; pkill -9 -f EngineCore 2>/dev/null
sleep 4
tmux kill-server 2>/dev/null; sleep 1
export RELIQUARY_ZONE_SIGMA_MIN=0.0
tmux new-session -d -s miner "RELIQUARY_ZONE_SIGMA_MIN=0.0 bash /workspace/launch_miner.sh 2>&1 | tee /workspace/miner_diag.log"
sleep 6
echo "tmux: $(tmux ls 2>&1)"
echo "override: $(tr '\0' '\n' < /proc/$(ps -eo pid,cmd | grep "[c]li.main mine" | awk '{print $1}' | head -1)/environ 2>/dev/null | grep ZONE_SIGMA)"
