#!/bin/bash
# Ré-entraînement nocturne du prior v5 (cron dev box, 05:30 UTC).
# 1) le corpus est déjà frais (pull81 toutes les 30 min) ;
# 2) entraîne un candidat sur TOUTES les données v4 (train_prior_v50.py :
#    split temporel + prompts jamais vus = duel sans fuite) ;
# 3) LE DUEL : compare le candidat AU MODÈLE EN PLACE sur le même holdout ;
# 4) s'il gagne → scp vers la box en predictor_v50_candidate.json + rapport.
#    Le RESTART (promotion effective) suit la politique décidée par l'user.
set -u
cd /root/subnet81
LOG=/root/subnet81/data/retrain_nightly.log
{
echo "══════ $(date -u +%F' '%H:%M) UTC ══════"
# Référence du duel = ce qui TOURNE réellement sur la box (pas une copie
# locale qui peut être périmée après une promotion) ; en cas d'échec scp,
# on garde la référence locale existante.
# Le chemin du prior DEPLOYE se lit dans le launcher de la box (il a change
# 3 fois : v50 -> v58 -> v59) — ne jamais le coder en dur.
DEPLOYED=$(ssh -p 20300 root@157.10.162.245 \
  "grep -oE 'PROMPT_PREDICTOR:-[^}]*' /workspace/launch_miner_v4.sh" 2>/dev/null | cut -d- -f2-)
DEPLOYED=${DEPLOYED:-/workspace/predictor_v59.json}
scp -q -P 20300 "root@157.10.162.245:$DEPLOYED" \
  /root/subnet81/data/predictor_deployed.json 2>/dev/null \
  && echo "référence du duel synchronisée depuis la box ($DEPLOYED)"
PYTHONPATH=/root/subnet81/.worktrees/miner-priv-port-v4-dapo \
  python3 scripts/train_prior_v50.py 2>&1 | tee /tmp/retrain_out.txt
# ⛔ 2 bugs historiques ici : le motif predictor_v50_* ratait les candidats
# predictor_v5.3+ (4 jours de candidats geles, 24/08) ET pointait du coup sur
# le VIEUX modele du 18/08 — le promouvoir aurait REMIS l'ancien. Motif =
# fichiers dates du traineur, REPLI exclus.
NEW=$(ls -t /root/subnet81/data/predictor_v*_20*.json | head -1)
# duel symétrique intégré à train_prior_v50.py (verdict CANDIDAT_GAGNANT/PERDANT)
if grep -q CANDIDAT_GAGNANT /tmp/retrain_out.txt || tail -5 "$LOG" | grep -q CANDIDAT_GAGNANT; then
  scp -q -P 20300 "$NEW" root@157.10.162.245:/workspace/.cand.tmp && ssh -p 20300 root@157.10.162.245 "mv /workspace/.cand.tmp /workspace/predictor_v50_candidate.json" \
    && cp "$NEW" /root/subnet81/data/predictor_candidate_latest.json \
    && echo "candidat déployé sur la box (predictor_v50_candidate.json) — restart requis pour activer"
fi
} >> "$LOG" 2>&1
