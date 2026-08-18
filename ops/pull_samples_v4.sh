#!/bin/bash
# Rapatriement continu du dump v4 (box → dev box) — à lancer sur la DEV BOX
# dans un tmux `pull81` dès le lancement v4 :
#   tmux new-session -d -s pull81 "bash /root/subnet81/reliquary-miner-priv/ops/pull_samples_v4.sh"
# Le corpus du prior v5 se construit tout seul pendant que le mineur tourne.
set -u
BOX="${BOX:-root@31.22.104.180}"
PORT="${PORT:-40300}"
DEST_DIR="${DEST_DIR:-/root/subnet81/data}"
INTERVAL="${INTERVAL:-1800}"
# Les 4 JSONL de l'étude v4 (etudev4.md §B) : corpus prior + verdicts/rangs +
# soumissions (jointure merkle) + fenêtres (cooldown/mémo shadow).
FILES="samples_v4.jsonl verdicts_v4.jsonl submits_v4.jsonl windows_v4.jsonl"

mkdir -p "$DEST_DIR"
while true; do
  ok=0
  for f in $FILES; do
    # rsync --append-verify : ne re-transfère que la queue (append-only)
    rsync -e "ssh -p $PORT" --append-verify "$BOX:/workspace/$f" "$DEST_DIR/$f" 2>/dev/null && ok=$((ok+1))
  done
  echo "$(date -u +%H:%M) pull $ok/4 — $(wc -l "$DEST_DIR"/samples_v4.jsonl 2>/dev/null | cut -d' ' -f1) groupes"
  sleep "$INTERVAL"
done
