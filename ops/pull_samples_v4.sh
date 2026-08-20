#!/bin/bash
# Rapatriement continu du dump v4 (box → dev box) — à lancer sur la DEV BOX
# dans un tmux `pull81` dès le lancement v4 :
#   tmux new-session -d -s pull81 "bash /root/subnet81/reliquary-miner-priv/ops/pull_samples_v4.sh"
# Le corpus du prior v5 se construit tout seul pendant que le mineur tourne.
set -u
BOX="${BOX:-root@38.255.28.21}"
PORT="${PORT:-20098}"
DEST_DIR="${DEST_DIR:-/root/subnet81/data}"
INTERVAL="${INTERVAL:-1800}"
# Les 4 JSONL de l'étude v4 (etudev4.md §B) : corpus prior + verdicts/rangs +
# soumissions (jointure merkle) + fenêtres (cooldown/mémo shadow).
FILES="samples_v4.jsonl verdicts_v4.jsonl submits_v4.jsonl windows_v4.jsonl"

mkdir -p "$DEST_DIR"
while true; do
  ok=0
  for f in $FILES; do
    # GARDE ANTI-ÉCRASEMENT (20/08) : --append-verify suppose que le fichier
    # distant ne fait que GRANDIR. Quand la box est reconstruite (reboot Lium =
    # /workspace effacé), le distant repart à zéro et rsync ÉCRASE l'historique
    # local — c'est arrivé le 20/08 21h45, submits/verdicts réduits à 10 et 401
    # lignes. On refuse donc tout transfert qui RÉTRÉCIRAIT le fichier local, et
    # on bascule alors sur un fichier daté pour ne rien perdre des deux côtés.
    loc=$(stat -c%s "$DEST_DIR/$f" 2>/dev/null || echo 0)
    rem=$(ssh -o ConnectTimeout=10 -p "$PORT" "$BOX" "stat -c%s /workspace/$f 2>/dev/null || echo 0" 2>/dev/null)
    rem=${rem:-0}
    if [ "$rem" -lt "$loc" ]; then
      # la box a été réinitialisée : on archive le nouveau flux à part
      alt="$DEST_DIR/${f%.jsonl}_box$(date -u +%m%d).jsonl"
      rsync -e "ssh -p $PORT" "$BOX:/workspace/$f" "$alt" 2>/dev/null && ok=$((ok+1))
      echo "  ⚠️ $f : distant ($rem o) < local ($loc o) — historique local PRÉSERVÉ, nouveau flux dans $(basename "$alt")"
    else
      rsync -e "ssh -p $PORT" --append-verify "$BOX:/workspace/$f" "$DEST_DIR/$f" 2>/dev/null && ok=$((ok+1))
    fi
  done
  echo "$(date -u +%H:%M) pull $ok/4 — $(wc -l "$DEST_DIR"/samples_v4.jsonl 2>/dev/null | cut -d' ' -f1) groupes"
  sleep "$INTERVAL"
done
