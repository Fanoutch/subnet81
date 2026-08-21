#!/bin/bash
# Éprouve que la garde VRAM est bien INERTE par défaut, et réactivable.
decide() {  # $1 = VRAM, $2 = WATCHDOG_VRAM_GUARD
  local vram="$1" g="${2:-0}"
  if [ "${g}" = "1" ] && [ -n "$vram" ] && [ "$vram" -gt "${WATCHDOG_VRAM_MAX:-140000}" ] 2>/dev/null; then
    echo RESTART; else echo RIEN; fi
}
ok=0; ko=0
t() { r=$(decide "$2" "$3"); [ "$r" = "$4" ] && { echo "  ✓ $1 → $r"; ok=$((ok+1)); } \
      || { echo "  ✗ $1 → $r (attendu $4)"; ko=$((ko+1)); }; }
echo "── par défaut (garde désactivée) : plus AUCUN redémarrage ──"
t "135683 Mo (le cas qui a tué le mineur à 08:42)" 135683 "" RIEN
t "135729 Mo (cas de 00:40)"                        135729 "" RIEN
t "142000 Mo (très haut)"                           142000 "" RIEN
t "120089 Mo (régime sain)"                         120089 "" RIEN
echo "── si on la réactive explicitement (seuil 140 Go) ──"
t "135683 Mo — palier normal post-rechargement"     135683 1  RIEN
t "141000 Mo — au-dessus du nouveau seuil"          141000 1  RESTART
echo
echo "  $ok réussis, $ko échoués"
[ "$ko" -eq 0 ] || exit 1
