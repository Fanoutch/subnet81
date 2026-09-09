#!/bin/bash
# Sauvegarde périodique des données du mineur, box GPU -> dev box.
#
# Historique des pertes qui ont motivé ce script :
#  - 2026-08-11 : coupure Lium, ~5 000 groupes cap-8192 perdus (aucune sauvegarde).
#  - 2026-08-20 21h20 : la H200 redémarre, /workspace ENTIÈREMENT vide. Le corpus
#    de la mémoire des vedettes (29 559 groupes, plusieurs jours de minage) n'a
#    survécu que parce qu'une copie traînait sur la dev box.
#  - 2026-08-20 21h45 : pire encore, le rapatriement a ÉCRASÉ l'historique local
#    par les fichiers neufs de la box reconstruite (rsync --append-verify suppose
#    que le distant ne fait que grandir). submits_v4.jsonl réduit à 10 lignes.
#
# D'où les deux principes de ce script :
#   1. il écrit dans des dossiers DATÉS — une sauvegarde n'écrase jamais la
#      précédente, donc une box réinitialisée ne peut pas détruire l'historique ;
#   2. il refuse de remplacer la dernière sauvegarde par une plus petite, et le
#      signale au lieu de le faire en silence.
#
# Cron (dev box) : */30 * * * *  — perte maximale 30 minutes.
# ⚠️ METTRE À JOUR PORT/HOST ICI quand la box change (le port change à chaque
# reboot chez ce fournisseur).

PORT=20098
HOST=root@38.255.28.21
FILES="samples_v4.jsonl verdicts_v4.jsonl submits_v4.jsonl windows_v4.jsonl predictor_v50.json risk_short_v1.json"

DEST_ROOT=/root/subnet81/data_backups
JOUR=$(date -u +%Y-%m-%d)
DEST="$DEST_ROOT/$JOUR"
LOG="$DEST_ROOT/backup.log"
mkdir -p "$DEST"

echo "$(date -u +%FT%TZ) — début ($HOST:$PORT)" >> "$LOG"

if ! ssh -p "$PORT" -o ConnectTimeout=10 -o BatchMode=yes \
        -o StrictHostKeyChecking=no "$HOST" true 2>/dev/null; then
  echo "$(date -u +%FT%TZ)   box injoignable — rien tenté" >> "$LOG"
  exit 0
fi

for f in $FILES; do
  tmp="$DEST/.$f.tmp"
  if ! rsync -q -e "ssh -p $PORT -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no" \
       "$HOST:/workspace/$f" "$tmp" 2>/dev/null; then
    echo "$(date -u +%FT%TZ)   $f : absent de la box" >> "$LOG"
    rm -f "$tmp"
    continue
  fi
  neuf=$(stat -c%s "$tmp" 2>/dev/null || echo 0)
  # dernière sauvegarde connue, tous jours confondus
  ancien_f=$(ls -t "$DEST_ROOT"/*/"$f" 2>/dev/null | head -1)
  ancien=$(stat -c%s "$ancien_f" 2>/dev/null || echo 0)
  if [ "$neuf" -lt "$ancien" ]; then
    # la box a probablement été réinitialisée : on garde les DEUX
    alt="$DEST/${f}.box-$(date -u +%H%M)"
    mv "$tmp" "$alt"
    echo "$(date -u +%FT%TZ)   ⚠️ $f : $neuf o < $ancien o (box réinitialisée ?)" \
         "— ancienne sauvegarde CONSERVÉE, nouvelle dans $(basename "$alt")" >> "$LOG"
  else
    mv "$tmp" "$DEST/$f"
    echo "$(date -u +%FT%TZ)   $f : $neuf o" >> "$LOG"
  fi
done

# rétention : garder 14 jours de sauvegardes quotidiennes
find "$DEST_ROOT" -maxdepth 1 -type d -name "20*-*-*" -mtime +14 \
     -exec rm -rf {} + 2>/dev/null

echo "$(date -u +%FT%TZ) — fin ($(du -sh "$DEST" 2>/dev/null | cut -f1))" >> "$LOG"
