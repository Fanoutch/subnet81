#!/bin/bash
# Déploiement des 3 protections du 2026-08-16 — NE LANCER QUE SUR GO UTILISATEUR.
#   1. vllm_backend.py : kill des EngineCore zombies au reload (7 tests verts)
#   2. watchdog v2     : restart auto si >=5 pre_bake failed / 5 min
#   3. GPU_FRACTION    : 0.78 -> 0.76 (~3 Go de marge pour les preuves GRAIL)
# Usage : bash ops/deploy_fixes_2026-08-16.sh   (depuis /root/subnet81/reliquary-miner-priv)
set -euo pipefail
BOX=root@31.22.104.180; PORT=40300
SSH="ssh -o BatchMode=yes -p $PORT $BOX"

echo "== 1/4 : push du code et du watchdog =="
scp -P $PORT reliquary/miner/vllm_backend.py "$BOX:/workspace/reliquary-miner-priv/reliquary/miner/vllm_backend.py"
scp -P $PORT ops/watchdog_h200_v2.sh "$BOX:/workspace/watchdog.sh"

echo "== 2/4 : GPU_FRACTION 0.78 -> 0.76 dans start_miner.sh =="
$SSH 'cp /workspace/start_miner.sh /workspace/start_miner.sh.bak-$(date -u +%F-%H%M);
      sed -i "s/RELIQUARY_VLLM_GPU_FRACTION=0.78/RELIQUARY_VLLM_GPU_FRACTION=0.76/" /workspace/start_miner.sh;
      grep GPU_FRACTION /workspace/start_miner.sh'

echo "== 3/4 : restart mineur + watchdog =="
$SSH 'cat > /tmp/deploy_restart.sh <<"EOF"
echo "$(date -u +%FT%TZ) RESTART DEPLOIEMENT (fixes zombie-kill + watchdog v2 + fraction 0.76)" >> /workspace/watchdog.log
tmux kill-session -t miner 2>/dev/null; tmux kill-session -t watchdog81 2>/dev/null
pkill -9 -f "cli.main mine"; pkill -9 -f EngineCore; sleep 8
tmux new-session -d -s miner "bash /workspace/start_miner.sh 2>&1 | tee -a /workspace/miner.log"
tmux new-session -d -s watchdog81 "bash /workspace/watchdog.sh 2>&1 | tee -a /workspace/watchdog.log"
sleep 4; tmux ls
EOF
bash /tmp/deploy_restart.sh'

echo "== 4/4 : vérification =="
$SSH 'sleep 10; pgrep -f "cli.main mine" | head -1; tail -3 /workspace/watchdog.log'
echo "Déploiement terminé — surveiller le warmup (~2-4 min) puis les premiers submits."
