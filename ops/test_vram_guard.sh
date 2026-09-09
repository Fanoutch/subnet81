#!/bin/bash
# Éprouve la garde VRAM anti-faux-positif sur 4 scénarios, hors production.
# On rejoue exactement la logique du watchdog avec des entrées contrôlées.
WATCHDOG_VRAM_MAX=135000
decide() {   # $1 = VRAM mesurée, $2 = âge du dernier "Loading checkpoint" en s
  local vram="$1" age="$2" LOG=/tmp/fake_miner.log
  : > "$LOG"
  if [ "$age" -ge 0 ]; then
    date -d "@$(( $(date +%s) - age ))" "+%Y-%m-%d %H:%M:%S | Loading checkpoint from /x" >> "$LOG"
  fi
  ck_recent=0
  ck_line=$(grep -a "Loading checkpoint from" "$LOG" 2>/dev/null | tail -1 | cut -c1-19)
  if [ -n "$ck_line" ]; then
    ck_ts=$(date -d "$ck_line" +%s 2>/dev/null || echo 0)
    [ "$ck_ts" -gt 0 ] && [ $(( $(date +%s) - ck_ts )) -lt 180 ] && ck_recent=1
  fi
  if [ -n "$vram" ] && [ "$vram" -gt "$WATCHDOG_VRAM_MAX" ] 2>/dev/null; then
    [ "$ck_recent" = "1" ] && echo "TOLERE" || echo "RESTART"
  else
    echo "RIEN"
  fi
}
ok=0; ko=0
t() { r=$(decide "$2" "$3"); if [ "$r" = "$4" ]; then echo "  ✓ $1 → $r"; ok=$((ok+1));
      else echo "  ✗ $1 → $r (attendu $4)"; ko=$((ko+1)); fi; }

echo "── les 3 faux positifs réellement survenus cette nuit ──"
t "135709 Mo, rechargement il y a 20 s (cas du 05:54)" 135709 20   TOLERE
t "135721 Mo, rechargement il y a 3 s  (cas du 03:20)" 135721 3    TOLERE
t "135729 Mo, rechargement il y a 11 s (cas du 00:40)" 135729 11   TOLERE
echo
echo "── la vraie fuite doit TOUJOURS déclencher ──"
t "136000 Mo, aucun rechargement récent (4 h)"          136000 14400 RESTART
t "140000 Mo, aucun rechargement du tout"               140000 -1    RESTART
t "135500 Mo, rechargement il y a 200 s (>3 min)"       135500 200   RESTART
echo
echo "── régime sain : rien ne doit bouger ──"
t "119000 Mo, aucun rechargement"                       119000 -1    RIEN
t "119000 Mo, rechargement il y a 10 s"                 119000 10    RIEN
t "135000 Mo pile (au seuil, pas au-dessus)"            135000 -1    RIEN
echo
echo "  $ok réussis, $ko échoués"
[ "$ko" -eq 0 ] || exit 1
