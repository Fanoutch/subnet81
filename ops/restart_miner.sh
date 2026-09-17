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
# --- protocol v6 ---
# 08/09 (port v6, revue item 4) : la version de protocole passée dans l'env
# (`RELIQUARY_PROTOCOL_VERSION=6 bash restart_miner.sh`) est PERSISTÉE dans
# /workspace/.protocol_version ; sans env (nouvelle session ssh, relance par le
# watchdog qui n'a pas l'env du shell) on la relit, et le launcher la lit aussi
# en repli avant son défaut 5. L'env explicite gagne toujours.
_PV_FILE=${RELIQUARY_PROTOCOL_VERSION_FILE:-/workspace/.protocol_version}
if [ -n "${RELIQUARY_PROTOCOL_VERSION:-}" ]; then
  echo "$RELIQUARY_PROTOCOL_VERSION" > "$_PV_FILE"
elif [ -s "$_PV_FILE" ]; then
  export RELIQUARY_PROTOCOL_VERSION="$(tr -dc 0-9 < "$_PV_FILE")"
fi
# --- fin protocol ---
LAUNCHER="$(cat /workspace/.miner_launcher 2>/dev/null || echo /workspace/launch_miner.sh)"
# V1 (#253) : réplique du validateur (verdict exact de l'EOS final), dans son
# propre venv. Relancée ici car ``tmux kill-server`` ci-dessus l'a tuée.
# Forwards simultanés : /workspace/.replica_workers (défaut 1 = sérialisé) —
# >1 seulement après PASS de ops/replica_parallel_gate.py sur la carte.
_RW="${REPLICA_WORKERS:-$(tr -dc 0-9 < /workspace/.replica_workers 2>/dev/null)}"
_RW="${_RW:-1}"
if [ -x /workspace/venv_val/bin/python ] && [ -d /workspace/reliquary_upstream ]; then
  tmux new-session -d -s replica81 "cd /workspace && REPLICA_WORKERS=$_RW REPLICA_MARGIN_DUMP=${REPLICA_MARGIN_DUMP:-/workspace/eos_margin_v4.jsonl} PYTHONPATH=/workspace/reliquary_upstream RELIQUARY_PROTOCOL_VERSION=6 RELIQUARY_PROTOCOL_PROFILE=qwen3-4b-base-dapo-reliquary-v1 RELIQUARY_EXPERIMENTAL_FILL_CLOSED_ENABLED=1 HF_HOME=/workspace/hf /workspace/venv_val/bin/python /workspace/reliquary-miner-priv/ops/replica_service.py /workspace/replica.sock 2>&1 | tee -a /workspace/replica.log"
fi
# Journal conservé (14/09 : logs de la 45899 écrasés par un redémarrage) :
# l'ancien est renommé avec la date, les 5 derniers sont gardés.
if [ -s /workspace/miner.log ]; then
  mv /workspace/miner.log "/workspace/miner.log.$(date -u +%Y%m%d-%H%M%S)"
  ls -1t /workspace/miner.log.2* 2>/dev/null | tail -n +6 | xargs -r rm -f
fi
tmux new-session -d -s miner "bash $LAUNCHER 2>&1 | tee /workspace/miner.log"
sleep 6
echo "tmux: $(tmux ls 2>&1)"
echo "proc: $(ps -eo pid,cmd | grep "[c]li.main mine" | head -1)"

# relance watchdog+monitor (fix 19/08 : le reload ckpt nocturne relançait le
# mineur seul — 00:13 la nuit dernière, surveillance morte jusqu au matin)
sleep 2
# --- wedge v6 ---
# 08/09 : le watchdog est lancé SANS l'env du launcher. Sous v6 (fenêtre
# fill-closed ~1 800 s) son seuil de wedge doit passer 900 → 2700 s, sinon il
# tue le mineur en pleine fenêtre. Version effective = env, sinon le défaut
# écrit dans le launcher. WATCHDOG_WEDGE_S explicite gagne toujours.
_pv="${RELIQUARY_PROTOCOL_VERSION:-$(grep -oE 'RELIQUARY_PROTOCOL_VERSION:-[0-9]+' "$LAUNCHER" 2>/dev/null | head -1 | sed 's/.*:-//')}"
if [ -z "${WATCHDOG_WEDGE_S:-}" ] && [ "$_pv" = "6" ]; then WATCHDOG_WEDGE_S=2700; fi
WATCHDOG_WEDGE_S="${WATCHDOG_WEDGE_S:-900}"
# --- fin wedge ---
tmux new-session -d -s watchdog81 "WATCHDOG_WEDGE_S=$WATCHDOG_WEDGE_S bash /workspace/watchdog.sh 2>&1 | tee -a /workspace/watchdog.log" 2>/dev/null
tmux new-session -d -s monitor "bash /workspace/reliquary-miner-priv/ops/monitor_v4.sh" 2>/dev/null
