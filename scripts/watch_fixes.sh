#!/bin/bash
# Santé du moteur : âge réel, gen1 + gen8 (famine CPU), fantômes de grading,
# taux de rollouts courts, et durée du dernier rechargement de checkpoint.
#
# Le corps Python vit dans scripts/watch_fixes_tick.py et est copié sur la box :
# le passer par un document en ligne à travers ssh exposait ses accents graves
# et ses apostrophes à l'interprétation du shell distant — trois pannes
# silencieuses le 20-21/08. Un fichier ne s'échappe pas.
BOX=root@38.255.28.21
PORT=20098
TICK=/root/subnet81/scripts/watch_fixes_tick.py
scp -q -o ConnectTimeout=15 -P "$PORT" "$TICK" "$BOX:/workspace/watch_fixes_tick.py" 2>/dev/null
while true; do
  sleep 600
  ssh -o ConnectTimeout=20 -p "$PORT" "$BOX" \
      '/workspace/venv/bin/python /workspace/watch_fixes_tick.py' 2>/dev/null
done
