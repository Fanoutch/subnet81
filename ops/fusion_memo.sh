#!/bin/bash
# Restauration de la mémoire des vedettes après reconstruction de la box.
# Le corpus /workspace/samples_v4.jsonl est relu au démarrage par payable_memo :
# effacé avec la box, il ne connaissait plus que 92 payables au lieu de 27 381.
set -e
A=$(wc -l < /workspace/samples_archive.jsonl)
C=$(wc -l < /workspace/samples_v4.jsonl)
cat /workspace/samples_archive.jsonl /workspace/samples_v4.jsonl > /workspace/samples_merged.jsonl
M=$(wc -l < /workspace/samples_merged.jsonl)
echo "  $A (archive) + $C (courant) = $M"
if [ "$M" -ne $((A+C)) ]; then echo "  ABANDON : fusion incoherente"; exit 1; fi
cp /workspace/samples_v4.jsonl /workspace/samples_v4.jsonl.avant_fusion
mv /workspace/samples_merged.jsonl /workspace/samples_v4.jsonl
echo "  en place : $(wc -l < /workspace/samples_v4.jsonl) lignes"
bash /workspace/restart_miner.sh 2>&1 | tail -3
