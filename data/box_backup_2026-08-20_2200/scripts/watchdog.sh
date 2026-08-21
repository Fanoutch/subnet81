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
  # ══ GARDE VRAM — DESACTIVEE le 21/08 sur decision utilisateur ══
  #
  # BILAN DE CETTE GARDE DEPUIS SA CREATION : 1 incident de 37 min hors ligne
  # (20/08 18:21, seuil 128 Go declenche a 57 Mo pres PENDANT une panne
  # validateur), 4 redemarrages inutiles en une nuit (00:40, 03:20, 05:54,
  # 08:42), ~6 fenetres degradees, ~8 payees perdues. Et ZERO panne reellement
  # attrapee.
  #
  # POURQUOI ELLE NE POUVAIT PAS MARCHER : chaque rechargement de checkpoint
  # laisse un PALIER PERMANENT d environ 15 Go (fragmentation de l allocateur) :
  # 120 Go apres un demarrage, ~135,6 Go apres un rechargement, et qui monte
  # de ~70 Mo/min. Ce n est PAS un pic transitoire — ma fenetre de grace de
  # 3 min attendait une redescente qui n arrive jamais.
  #
  # ET SURTOUT : a 135,6 Go le mineur allait TRES BIEN — gen1 2,1 s, gen8 5,4 s,
  # ses meilleures valeurs de la journee, juste avant que la garde ne le tue.
  # Le seuil protegeait contre un regime sain.
  #
  # LA FUITE VRAM D ORIGINE (nuit 19->20, 118->134,6 Go en 8h30) avait pour
  # cause les processus de grading zombies. Elle est corrigee A LA RACINE
  # (finally: proc.kill() dans code_grader.py) et couverte par le detecteur de
  # famine ci-dessous, qui exige DEUX signaux concordants.
  #
  # Reactivation possible via WATCHDOG_VRAM_GUARD=1, mais alors relire ce qui
  # precede : le seuil devrait etre >=140 Go ET exiger plusieurs relevés
  # consecutifs, sinon elle renuira.
  if [ "${WATCHDOG_VRAM_GUARD:-0}" = "1" ] && [ -n "$vram" ] \
     && [ "$vram" -gt "${WATCHDOG_VRAM_MAX:-140000}" ] 2>/dev/null; then
    restart_miner "FUITE_VRAM (${vram} MiB > ${WATCHDOG_VRAM_MAX:-140000})"
    sleep 600; continue
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
  # BUG CORRIGE (21/08) : `pgrep -fc` AFFICHE deja 0 quand il ne trouve rien ET
  # renvoie un code d erreur -> le `|| echo 0` ajoutait un SECOND zero, la
  # variable valait "0\n0" et le test echouait avec "integer expression
  # expected". Le detecteur de famine ne s executait donc JAMAIS depuis sa pose
  # le 20/08 au soir. On prend la premiere ligne et on force un entier.
  fant=$(pgrep -fc code_grader_driver 2>/dev/null | head -1)
  case "$fant" in ''|*[!0-9]*) fant=0 ;; esac
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
