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
  mpid=$(pgrep -f "reliquary.cli.main mine" | head -1)
  # v4 (20/08) : PROCESS ABSENT = panne silencieuse. L'ancienne boucle faisait
  # `continue` et n'a donc PAS rattrapé l'arrêt du 20/08 18:21 (launcher ABORT
  # sur validateur en 502) : 37 min hors ligne. On relance, en laissant au
  # launcher le soin d'attendre le validateur.
  if [ -z "$mpid" ]; then
    last_restart=$(cat "$STATE" 2>/dev/null || echo 0)
    if [ $(( $(date -u +%s) - last_restart )) -ge 120 ]; then
      restart_miner "PROCESS ABSENT (aucun mineur en vie)"
      sleep 120
    fi
    continue
  fi
  last_restart=$(cat "$STATE" 2>/dev/null || echo 0)
  since=$(( $(date -u +%s) - last_restart ))
  [ "$since" -lt 900 ] && continue   # <15min après un restart : warmup, ignorer

  # GARDE VALIDATEUR (fix 20/08 19h) : si le validateur ne répond pas, le
  # mineur ne PEUT pas travailler — ni générer (pas de randomness) ni soumettre.
  # Le redémarrer ne répare rien et risque de le laisser mort (le launcher
  # attend le validateur). Incident 18:21 : 37 min hors ligne pour cette raison.
  vcode=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 \
          "${RELIQUARY_VALIDATOR_URL:-http://209.20.157.231:8080}/health" 2>/dev/null)
  if [ "$vcode" != "200" ]; then
    echo "$(date -u +%FT%TZ) validateur HTTP $vcode — surveillance en pause (aucun restart)" >> "$WLOG"
    sleep 60; continue
  fi

  # v2 — preuves en échec en rafale (log en UTC, préfixe timestamp trié lexico)
  cut=$(date -u -d "-${OOM_FENETRE_MIN} min" '+%Y-%m-%d %H:%M:%S')
  oomn=$(tail -c 400000 "$LOG" | grep -a "pre_bake failed" \
         | awk -v c="$cut" 'substr($0,1,19) >= c' | wc -l)
  if [ "$oomn" -ge "$OOM_SEUIL" ]; then
    restart_miner "OOM_PREUVES ($oomn pre_bake failed en ${OOM_FENETRE_MIN}min, depuis_restart ${since}s)"
    sleep 600; continue
  fi

  # v3 (20/08) — FUITE VRAM : la nuit 19→20/08, la VRAM a dérivé 118→134,6 Go
  # en 8h30 (fragmentation de l'allocateur sur les formes variées des forwards
  # de preuve) → vLLM étranglé → génération 3-5× plus lente → 7 h à ZÉRO
  # acceptée, sans aucun crash ni silence (v1/v2 aveugles). Seuil 128 Go =
  # au-dessus du régime sain (118-120) et sous la zone d'étranglement.
  vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  # Seuil relevé 20/08 19h : 128 000 était trop serré (régime sain = 119-120 Go)
  # — un pic de 57 Mo a déclenché un restart le 20/08 18:21 PENDANT une panne
  # validateur, d'où 37 min hors ligne. La fuite d'origine (processus de
  # grading zombies) est corrigée ; 135 Go ne peut venir que d'une vraie dérive.
  # GARDE ANTI-FAUX-POSITIF (21/08) : pendant un rechargement de checkpoint,
  # l ancien et le nouveau modele coexistent brievement en VRAM et le pic
  # depasse le seuil. Mesure : 3 rechargements sur 7 ont declenche un restart
  # INUTILE (00:40, 03:20, 05:54 — 135,7 Go chacun, retombes ensuite), chacun
  # coutant 2-3 fenetres EN PLUS des 8 min du rechargement lui-meme.
  # On ignore donc le controle VRAM dans les 3 min qui suivent un
  # "Loading checkpoint". Une vraie derive, elle, met des heures a monter et
  # sera vue au controle suivant.
  ck_recent=0
  ck_line=$(grep -a "Loading checkpoint from" "$LOG" 2>/dev/null | tail -1 | cut -c1-19)
  if [ -n "$ck_line" ]; then
    ck_ts=$(date -d "$ck_line" +%s 2>/dev/null || echo 0)
    [ "$ck_ts" -gt 0 ] && [ $(( $(date +%s) - ck_ts )) -lt 180 ] && ck_recent=1
  fi
  if [ -n "$vram" ] && [ "$vram" -gt "${WATCHDOG_VRAM_MAX:-135000}" ] 2>/dev/null; then
    if [ "$ck_recent" = "1" ]; then
      echo "$(date -u +%FT%TZ) VRAM ${vram} MiB > seuil MAIS rechargement de checkpoint il y a <3 min — pic normal, aucun restart" >> "$WLOG"
    else
      restart_miner "FUITE_VRAM (${vram} MiB > ${WATCHDOG_VRAM_MAX:-135000}, depuis_restart ${since}s)"
      sleep 600; continue
    fi
  fi

  # v4.1 (20/08 23h) — FAMINE DE GRADING, le point aveugle de toutes les
  # versions precedentes. La nuit du 19 au 20, le mineur a passe 7 HEURES a
  # zero : les processus de grading fuyaient a 100 % de CPU (quota conteneur
  # 24, pas 192), vLLM etait affame, mais le mineur GENERAIT toujours. Donc
  # aucun silence, aucun OOM, aucune VRAM anormale -> v1 et v2 aveugles.
  #
  # Signature : le 8e groupe du lot s'effondre (7,5 s -> 21 s) PENDANT que les
  # processus de grading s'accumulent. Les deux ENSEMBLE, jamais l'un seul :
  # un gros lot fait monter gen8 a lui seul (mesure du 20/08 : 18,7 s pour
  # 9600 tokens contre 7,5 s pour 4400 — c'est une BONNE nouvelle, pas une
  # panne). Seuils volontairement hauts pour ne jamais tuer un bon lot.
  fant=$(pgrep -fc code_grader_driver 2>/dev/null || echo 0)
  if [ "${fant:-0}" -ge 8 ]; then
    g8=$(tail -c 400000 "$LOG" | grep -aoE "groupe 8/8 pr.t . [0-9.]+s" \
         | tail -8 | grep -oE "[0-9.]+" | sort -n | awk '{a[NR]=$1} END{if(NR)print a[int((NR+1)/2)]}')
    if [ -n "$g8" ] && awk "BEGIN{exit !($g8 > 20)}"; then
      restart_miner "FAMINE DE GRADING (gen8 ${g8}s, ${fant} processus de grading)"
      sleep 900; continue
    fi
    echo "$(date -u +%FT%TZ) ${fant} processus de grading, gen8 ${g8:-?}s — surveille, pas de restart" >> "$WLOG"
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
