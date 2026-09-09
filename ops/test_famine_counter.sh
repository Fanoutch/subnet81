#!/bin/bash
# Éprouve la lecture du compteur de processus de grading dans les 3 cas.
essai() {  # $1 = ce que pgrep renvoie, $2 = code retour, $3 = attendu
  fant=$(printf '%s' "$1" | head -1)
  case "$fant" in ''|*[!0-9]*) fant=0 ;; esac
  if [ "$fant" -ge 8 ] 2>/dev/null; then r=DECLENCHE; else r=SILENCE; fi
  [ "$r" = "$3" ] && echo "  ✓ pgrep='$1' → $r" || echo "  ✗ pgrep='$1' → $r (attendu $3)"
}
echo "── ancien comportement (le bug) ──"
f=$(printf '0\n0'); case "$f" in ''|*[!0-9]*) : ;; esac
[ "$f" -ge 8 ] 2>/dev/null && echo "  ancien: DECLENCHE" || echo "  ancien: ERREUR ou SILENCE (c'était le bug)"
echo "── nouveau comportement ──"
essai "0"    0 SILENCE
essai "2"    0 SILENCE
essai "12"   0 DECLENCHE
essai ""     1 SILENCE
essai "abc"  1 SILENCE
