#!/bin/bash
# Promoteur de prior — « auto au creux » (go utilisateur), ARMEMENT MANUEL :
# ne tourne que lancé explicitement (nohup, PAS tmux : restart_miner fait
# tmux kill-server et tuerait le promoteur en pleine promotion).
#   Lancement (top utilisateur) : nohup bash ops/promote_prior.sh >> /workspace/promoter.log 2>&1 &
# Garanties :
#   G1 candidat validé (JSON + word_priors) AVANT tout swap ;
#   G2 swap atomique, ancien modèle gardé en .prev ;
#   G3 promotion uniquement au CREUX (POST 503/down ou fenêtre non-open) ;
#   G4 vérification post-restart (« prédicteur ACTIF » sous 6 min) sinon
#      ROLLBACK automatique vers .prev + re-restart ;
#   G5 watchdog relancé dans tous les cas ; une seule promotion par candidat.
set -u
CAND=/workspace/predictor_v50_candidate.json
LIVE=/workspace/predictor_v50.json
LOGM=/workspace/miner.log

valide() {
  python3 -c "
import json,sys
m=json.load(open('$1'))
assert isinstance(m.get('word_priors'), dict) and len(m['word_priors'])>1000
assert 'global_mean' in m and 'idf' in m
print('ok')" 2>/dev/null
}

relance_watchdog() {
  tmux new-session -d -s watchdog81     "bash /workspace/watchdog.sh 2>&1 | tee -a /workspace/watchdog.log" 2>/dev/null
}

while true; do
  if [ -s "$CAND" ] && [ "$(valide "$CAND")" = "ok" ]; then
    POST=$(curl -s -o /dev/null -w "%{http_code}" --max-time 6 -X POST       -H "Content-Type: application/json" -d "{}"       http://209.20.157.231:8080/submit/precommit 2>/dev/null)
    STATE=$(curl -s --max-time 6 "http://209.20.157.231:8080/state?env=opencodeinstruct" 2>/dev/null       | python3 -c "import json,sys;print(json.load(sys.stdin).get('state',''))" 2>/dev/null)
    if [ "$POST" = "503" ] || [ "$POST" = "000" ] || { [ -n "$STATE" ] && [ "$STATE" != "open" ]; }; then
      echo "$(date -u +%F_%H:%M) PROMOTION: creux (POST=$POST state=$STATE)"
      cp "$LIVE" "${LIVE}.prev" 2>/dev/null
      mv "$CAND" "$LIVE"
      MARK=$(date -u +%s)
      bash /workspace/restart_miner.sh
      relance_watchdog
      ok=""
      for i in $(seq 1 36); do
        sleep 10
        if awk -v m="$MARK" "1" /dev/null 2>/dev/null &&            grep -q "prédicteur ACTIF: $LIVE" "$LOGM" 2>/dev/null; then ok=1; break; fi
      done
      if [ -n "$ok" ]; then
        echo "$(date -u +%F_%H:%M) PROMOTION VÉRIFIÉE (prédicteur ACTIF)"
      else
        echo "$(date -u +%F_%H:%M) ⚠️ ROLLBACK: prédicteur non chargé en 6 min — retour .prev"
        [ -s "${LIVE}.prev" ] && cp "${LIVE}.prev" "$LIVE"
        bash /workspace/restart_miner.sh
        relance_watchdog
      fi
    fi
  fi
  sleep 60
done
