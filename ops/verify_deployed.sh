#!/bin/bash
# À lancer SUR LA BOX, avant de démarrer le mineur.
#
# Pourquoi : l'explication la plus probable des 21 `bad_termination` du
# 2026-08-03 est que les gardes de terminaison n'étaient PAS sur la machine
# (elles vivaient dans un arbre de travail non committé). Vérifier que le code
# déployé contient bien les correctifs coûte 5 secondes ; le découvrir après
# coup coûte des fenêtres.
#
#   bash /workspace/reliquary-miner-priv/ops/verify_deployed.sh
set -u
MP="${1:-/workspace/reliquary-miner-priv}"
FAIL=0
ok()  { echo "  ✅ $1"; }
bad() { echo "  ❌ $1"; FAIL=1; }

echo "=== Vérification du code DÉPLOYÉ ($MP) ==="

# --- les correctifs de terminaison sont-ils présents ? ---
grep -q "def validator_termination_ok" "$MP/reliquary/miner/engine.py" \
  && ok "prédicat validateur (validator_termination_ok)" \
  || bad "prédicat validateur ABSENT — le mineur ne verra pas un EOS mal placé"

grep -q "def truncate_at_first_eos" "$MP/reliquary/miner/engine.py" \
  && ok "troncature au 1er EOS" || bad "troncature ABSENTE"

# la garde doit être CÂBLÉE, pas seulement définie
N=$(grep -c "in_eos = validator_termination_ok" "$MP/reliquary/miner/engine.py")
[ "${N:-0}" -ge 2 ] && ok "garde câblée aux $N sites (sync + async)" \
  || bad "garde câblée à seulement ${N:-0} site(s) — attendu 2"

grep -q "truncate_at_first_eos(gen, self._eos_ids)" "$MP/reliquary/miner/engine.py" \
  && ok "troncature câblée au chemin vLLM (production)" \
  || bad "troncature NON câblée au chemin vLLM"

grep -q "def terminating_rollouts" "$MP/reliquary/miner/engine.py" \
  && ok "filtre non-EOS dans le sélecteur" || bad "filtre sélecteur ABSENT"

grep -q "class DropTracker" "$MP/reliquary/miner/engine.py" \
  && ok "alarme panne silencieuse" || bad "alarme ABSENTE"

grep -q "generation_contract" "$MP/reliquary/protocol/submission.py" \
  && ok "parse /state v3" || bad "parse /state v3 ABSENT — le mineur ne générera JAMAIS"

# --- valeurs de wire effectives (import réel, pas grep) ---
# G8 (balayage 18/08) : version-aware — le check suit RELIQUARY_PROTOCOL_VERSION
# de l'env courant (v4 : cap 8192, domaine v4, profil dapo-v4 ; BFT_THINKING_
# BUDGET reste 15616 = constante morte sous v4).
PV="${RELIQUARY_PROTOCOL_VERSION:-3}"
echo "--- contrat v$PV tel que le code le calcule ---"
OUT=$(cd "$MP" && PYTHONPATH=. python3 -c "
from reliquary import constants as c
print(c.FORCED_SEED_PROTOCOL_VERSION, c.FORCED_SEED_DOMAIN,
      c.BFT_THINKING_BUDGET, c.MAX_NEW_TOKENS_PROTOCOL_CAP,
      c.GENERATION_PROFILE_ID)" 2>&1 | tail -1)
if [ "$PV" -ge 4 ]; then
  EXPECT="4 reliquary-forced-seed-v4 15616 8192 qwen3-4b-base-dapo-v4"
else
  EXPECT="3 reliquary-forced-seed-v3 15616 16384 qwen35-4b-auction-v3"
fi
[ "$OUT" = "$EXPECT" ] \
  && ok "contrat v$PV conforme ($OUT)" || bad "contrat v$PV inattendu: $OUT (attendu: $EXPECT)"

# --- le validateur live sert-il bien ce profil ? ---
URL="${RELIQUARY_VALIDATOR_URL:-http://127.0.0.1:8080}"
PROF=$(curl -s -m 8 "$URL/health" 2>/dev/null \
  | python3 -c "import json,sys; print(json.load(sys.stdin).get('generation_profile_id','?'))" 2>/dev/null)
if [ "$PV" -ge 4 ]; then EXPECT_PROF="qwen3-4b-base-dapo-v4"; else EXPECT_PROF="qwen35-4b-auction-v3"; fi
[ "$PROF" = "$EXPECT_PROF" ] \
  && ok "validateur live sur $PROF" \
  || echo "  ⚠️  profil validateur = '${PROF:-injoignable}' (attendu $EXPECT_PROF — tunnel monté ? bascule faite ?)"

echo "=================================================="
[ $FAIL -eq 0 ] \
  && echo "✅ CODE À JOUR — lancement autorisé.
   Surveiller ensuite :
     grep -i 'PANNE SILENCIEUSE' miner.log     # doit rester VIDE
     grep 'submitted window' miner.log          # doit apparaître
     curl \$URL/verdicts/<hotkey>                # bad_termination doit DISPARAÎTRE" \
  || echo "❌ CODE INCOMPLET — re-rsync AVANT de lancer (c'est l'explication la
   plus probable des 21 rejets du 2026-08-03)."
exit $FAIL
