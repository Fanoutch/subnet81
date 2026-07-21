#!/bin/bash
# apply_pr54_alignment.sh
# À LANCER UNIQUEMENT quand le validateur a déployé PR #54
#   (band [3,5] retiré, SIGMA_MIN 0.43 -> 0.33).
# Avant ça = soumissions rejetées par le validateur encore en [3,5].
set -e
cd /root/reliquary-miner-priv
TS=$(date +%Y%m%d_%H%M%S)
cp reliquary/constants.py        reliquary/constants.py.bak_pre54_$TS
cp reliquary/miner/engine.py     reliquary/miner/engine.py.bak_pre54_$TS

# 1. SIGMA_MIN 0.43 -> 0.33
sed -i 's/^SIGMA_MIN = 0\.43/SIGMA_MIN = 0.33/' reliquary/constants.py

# 2. retirer le band [3,5] : accepter tout ce qui a passé le gate sigma
sed -i 's/^    return not (3 <= correct <= 5)/    return False  # PR#54: band [3,5] retire, sigma>=0.33 (=k[1,7]) seul gate/' reliquary/miner/engine.py

echo "=== verif valeurs ==="
grep -n "^SIGMA_MIN = 0" reliquary/constants.py
grep -n "PR#54: band" reliquary/miner/engine.py
echo "=== compile ==="
/root/venv/bin/python -m py_compile reliquary/constants.py reliquary/miner/engine.py && echo "OK compiles"
echo ">>> Patch applique. Redemarrer le mineur (tmux kill-session + relance) pour charger."
