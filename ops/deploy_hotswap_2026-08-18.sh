#!/bin/bash
# Déploiement fix hot-swap 2026-08-18 — NE LANCER QUE SUR GO UTILISATEUR.
# Réactive RELIQUARY_HOT_SWAP=1 avec le correctif des gels du 15/08 :
#   - drapeau d'interruption honoré par le driver (verrou libéré en ~1 step)
#   - reload_weights_inplace : verrou avec timeout 15 s, repli rebuild
#   - sonde self-gate sous verrou + bornée 30 s
# Gain attendu : reload checkpoint ~3 min -> ~15-30 s => récupère la majorité
# des 28 % de fenêtres perdues mesurées par le labo.
set -euo pipefail
BOX=root@31.22.104.180; PORT=40300
cd /root/subnet81/reliquary-miner-priv

echo "== 1/3 : push du code =="
scp -P $PORT reliquary/miner/vllm_backend.py "$BOX:/workspace/reliquary-miner-priv/reliquary/miner/vllm_backend.py"
scp -P $PORT reliquary/miner/engine.py "$BOX:/workspace/reliquary-miner-priv/reliquary/miner/engine.py"

echo "== 2/3 : RELIQUARY_HOT_SWAP=1 dans start_miner.sh =="
ssh -p $PORT $BOX 'cp /workspace/start_miner.sh /workspace/start_miner.sh.bak-$(date -u +%F-%H%M);
  sed -i "s/RELIQUARY_HOT_SWAP=0.*/RELIQUARY_HOT_SWAP=1  # réactivé 2026-08-18 : fix gels du 15\/08 (interrupt+timeouts, 7 tests)/" /workspace/start_miner.sh;
  grep HOT_SWAP /workspace/start_miner.sh'

echo "== 3/3 : restart =="
ssh -p $PORT $BOX 'cat > /tmp/deploy_hs.sh <<"EOF"
echo "$(date -u +%FT%TZ) RESTART DEPLOIEMENT hot-swap fixe (go utilisateur)" >> /workspace/watchdog.log
date -u +%s > /workspace/.watchdog_last_restart
tmux kill-session -t miner 2>/dev/null
pkill -9 -f "cli.main mine"; pkill -9 -f EngineCore; sleep 8
tmux new-session -d -s miner "bash /workspace/start_miner.sh 2>&1 | tee -a /workspace/miner.log"
EOF
nohup bash /tmp/deploy_hs.sh >/dev/null 2>&1 & exit 0'
echo "Déployé — surveiller le premier ckpt-advance : attendu ~15-30 s au lieu de ~3 min."
