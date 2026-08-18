#!/bin/bash
# Rapatriement continu du dump v4 (box → dev box) — à lancer sur la DEV BOX
# dans un tmux `pull81` dès le lancement v4 :
#   tmux new-session -d -s pull81 "bash /root/subnet81/reliquary-miner-priv/ops/pull_samples_v4.sh"
# Le corpus du prior v5 se construit tout seul pendant que le mineur tourne.
set -u
BOX="${BOX:-root@31.22.104.180}"
PORT="${PORT:-40300}"
REMOTE="${REMOTE:-/workspace/samples_v4.jsonl}"
DEST="${DEST:-/root/subnet81/data/samples_v4.jsonl}"
INTERVAL="${INTERVAL:-1800}"

mkdir -p "$(dirname "$DEST")"
while true; do
  # rsync --append-verify : ne re-transfère que la queue du JSONL (append-only)
  rsync -e "ssh -p $PORT" --append-verify "$BOX:$REMOTE" "$DEST" 2>/dev/null \
    && echo "$(date -u +%H:%M) pull ok — $(wc -l < "$DEST") lignes" \
    || echo "$(date -u +%H:%M) pull FAIL (box injoignable ?)"
  sleep "$INTERVAL"
done
