#!/bin/bash
# Watchdog mineur v2 (2026-08-16).
# v1 : restart si process présent MAIS aucun 'stream_fire' depuis 15 min
#      (incident 2026-08-06 : reload x bake a tué la géné, process VIVANT, 11h nul).
# v2 : + restart si >=5 'pre_bake failed' en 5 min — incident 2026-08-16 13:28 :
#      reload raté → processus principal gonflé (23→27,5 GiB) → TOUTES les
#      preuves GRAIL en OOM pendant que la génération continuait. Aucun
#      silence → v1 aveugle, revenu zéro. 5/5min est pathologique quelle que
#      soit la cause (régime normal : 0-3 par JOUR).
# GARDE anti-warmup v2.1 : `ps -o etimes=` renvoie une valeur ABERRANTE pour
# TOUS les processus sur cette box (bug conteneur, ~4.12e9 s constants) — la
# garde par âge du process n'a JAMAIS fonctionné (v1 incluse ; incident
# 2026-08-16 15:00 : v2 a compté les OOM d'AVANT un restart et a tué un mineur
# en warmup). Remplacée par un fichier d'état : 15 min de silence total après
# tout restart (le nôtre ou celui d'un déploiement qui écrit le fichier).
LOG=/workspace/miner.log; WLOG=/workspace/watchdog.log
STATE=/workspace/.watchdog_last_restart
OOM_SEUIL=5           # pre_bake failed
OOM_FENETRE_MIN=5     # minutes
date -u +%s > "$STATE"   # le démarrage du watchdog compte comme un restart
echo "$(date -u +%FT%TZ) watchdog v2.1 démarré (wedge 15min + oom ${OOM_SEUIL}/${OOM_FENETRE_MIN}min, garde=fichier état)" >> "$WLOG"

restart_miner() {  # $1 = raison
  echo "$(date -u +%FT%TZ) $1 — restart" >> "$WLOG"
  tmux kill-session -t miner 2>/dev/null
  pkill -9 -f "reliquary.cli.main mine"; pkill -9 -f EngineCore; sleep 8
  LAUNCHER=$(cat /workspace/.miner_launcher 2>/dev/null || echo /workspace/start_miner.sh)
  tmux new-session -d -s miner "bash $LAUNCHER 2>&1 | tee -a $LOG"
  date -u +%s > "$STATE"
}

while true; do
  sleep 60
  mpid=$(pgrep -f "reliquary.cli.main mine" | head -1) || continue
  [ -z "$mpid" ] && continue
  last_restart=$(cat "$STATE" 2>/dev/null || echo 0)
  since=$(( $(date -u +%s) - last_restart ))
  [ "$since" -lt 900 ] && continue   # <15min après un restart : warmup, ignorer

  # v2 — preuves en échec en rafale (log en UTC, préfixe timestamp trié lexico)
  cut=$(date -u -d "-${OOM_FENETRE_MIN} min" '+%Y-%m-%d %H:%M:%S')
  oomn=$(tail -c 400000 "$LOG" | grep -a "pre_bake failed" \
         | awk -v c="$cut" 'substr($0,1,19) >= c' | wc -l)
  if [ "$oomn" -ge "$OOM_SEUIL" ]; then
    restart_miner "OOM_PREUVES ($oomn pre_bake failed en ${OOM_FENETRE_MIN}min, depuis_restart ${since}s)"
    sleep 600; continue
  fi

  # v1 — wedge : plus aucun groupe généré
  # signes de vie : bake OU reload de checkpoint en cours (un pull nouveau
  # repo = ~8 Go + double chargement modèle : LÉGITIMEMENT >15 min sans bake —
  # incident 18/08 18:19, le watchdog a tué un mineur en plein reload)
  last=$(grep -oE "^2026-[0-9-]+ [0-9:]+" <(tail -c 400000 "$LOG" | grep -aE "stream_fire: groupe|Loading checkpoint|Loading weights|submitted window") | tail -1)
  if [ -z "$last" ]; then
    restart_miner "WEDGE (aucun stream_fire dans le tail, depuis_restart ${since}s)"
    sleep 600; continue
  fi
  last_s=$(date -d "$last" +%s 2>/dev/null) || continue
  age=$(( $(date +%s) - last_s ))
  if [ "$age" -gt 900 ]; then
    restart_miner "WEDGE (dernier groupe ${age}s, depuis_restart ${since}s)"
    sleep 600
  fi
done
