#!/bin/bash
# Vérificateur du port 4B/v3 — gates G1..G7 (G8 = GPU, hors scope dev box).
# Usage : bash scripts/check_4b_port.sh [--tests]
# Contrat cible = validateur live 2026-08-02 (profil qwen35-4b-auction-v3) :
#   curl 209.20.157.231:8080/health -> generation_contract.
# Chaque gate compare reliquary-miner-priv/ au vrai code origin/main du clone
# upstream (pas à des valeurs recopiées à la main quand la parité est exigée).
set -u
MP=/root/subnet81/reliquary-miner-priv
UP=/root/subnet81/reliquary
FAIL=0
gate() { # gate <id> <ok:0|1> <libellé> [détail]
  if [ "$2" -eq 0 ]; then echo "  ✅ $1 PASS — $3"
  else echo "  ❌ $1 FAIL — $3${4:+ : $4}"; FAIL=1; fi
}

echo "================ CHECK PORT 4B/v3 ($(date -u +%FT%TZ)) ================"

# ---------- G1 : dérivation domaine/version (approche fonctionnelle acceptée) ----------
# La session de code a choisi des fonctions env-overridables (défaut = v3 live)
# plutôt que le profiles.py upstream. Conformité wire = valeurs émises, pas le
# style ; on vérifie donc la dérivation + le chemin de rollback.
echo "--- G1 dérivation version/domaine ---"
grep -qE 'FORCED_SEED_DOMAIN = forced_seed_domain\(|FORCED_SEED_DOMAIN = f"reliquary-forced-seed-v\{' \
  "$MP/reliquary/constants.py" 2>/dev/null
gate G1a $? "FORCED_SEED_DOMAIN dérivé dynamiquement de la version (pas en dur)"
RB=$(cd "$MP" && RELIQUARY_PROTOCOL_VERSION=2 PYTHONPATH=. python3 -c \
  "from reliquary import constants as c; print(c.FORCED_SEED_PROTOCOL_VERSION, c.FORCED_SEED_DOMAIN)" 2>&1 | tail -1)
gate G1b $([ "$RB" = "2 reliquary-forced-seed-v2" ] && echo 0 || echo 1) \
  "rollback RELIQUARY_PROTOCOL_VERSION=2 fonctionne" "donne: $RB"

# ---------- G2 : valeurs v3 par DÉFAUT (sans env var — import Python réel) ----------
echo "--- G2 valeurs v3 par défaut ---"
G2OUT=$(cd "$MP" && PYTHONPATH=. python3 - <<'EOF' 2>&1
from reliquary import constants as c
errs = []
def chk(name, got, want):
    if got != want: errs.append(f"{name}={got!r} (attendu {want!r})")
chk("FORCED_SEED_PROTOCOL_VERSION", c.FORCED_SEED_PROTOCOL_VERSION, 3)
chk("FORCED_SEED_DOMAIN", c.FORCED_SEED_DOMAIN, "reliquary-forced-seed-v3")
chk("BFT_THINKING_BUDGET", c.BFT_THINKING_BUDGET, 15616)
chk("BFT_ANSWER_BUDGET", c.BFT_ANSWER_BUDGET, 512)
chk("MAX_NEW_TOKENS_PROTOCOL_CAP", c.MAX_NEW_TOKENS_PROTOCOL_CAP, 16384)
print("ERRS:" + ("; ".join(errs) if errs else "NONE"))
EOF
)
if echo "$G2OUT" | grep -q "^ERRS:NONE$"; then gate G2 0 "défaut sans env var = contrat v3 live (3 / domaine v3 / 15616 / 512 / 16384)"
else gate G2 1 "valeurs v3 par défaut" "$(echo "$G2OUT" | tail -3)"; fi

# ---------- G3 : enum RejectReason — seuls les MANQUANTS cassent le parse ----------
# (un membre en trop chez nous est inoffensif : le validateur ne l'envoie jamais)
echo "--- G3 enum RejectReason ---"
MISSING=$(comm -23 \
  <(git -C "$UP" show origin/main:reliquary/protocol/submission.py \
     | sed -n '/class RejectReason/,/^class /p' | grep -oE '"[a-z_0-9]+"' | sort) \
  <(sed -n '/class RejectReason/,/^class /p' "$MP/reliquary/protocol/submission.py" \
     | grep -oE '"[a-z_0-9]+"' | sort))
gate G3 $([ -z "$MISSING" ] && echo 0 || echo 1) "aucun membre upstream manquant (parse-safe)" "$MISSING"
EXTRA=$(comm -13 \
  <(git -C "$UP" show origin/main:reliquary/protocol/submission.py \
     | sed -n '/class RejectReason/,/^class /p' | grep -oE '"[a-z_0-9]+"' | sort) \
  <(sed -n '/class RejectReason/,/^class /p' "$MP/reliquary/protocol/submission.py" \
     | grep -oE '"[a-z_0-9]+"' | sort))
[ -n "$EXTRA" ] && echo "  ℹ️  membres en plus chez nous (inoffensif): $(echo $EXTRA | tr '\n' ' ')"

# ---------- G4 : enveloppe + precommit v3 ----------
echo "--- G4 enveloppe/precommit ---"
grep -q "generation_profile_id" "$MP/reliquary/protocol/signatures.py"
gate G4a $? "build_envelope_binding accepte generation_profile_id"
# Parité FONCTIONNELLE byte-exacte (upstream exécuté vs nous, vecteurs multiples,
# enveloppe + precommit) — remplace le diff textuel (notre impl garde une
# branche wire-v1/v2 historique, seuls les BYTES émis comptent).
PAR=$(python3 /root/subnet81/scripts/parity_bindings_v3.py 2>&1)
echo "$PAR" | grep -q "RESULT: PARITY-PASS"
gate G4b $? "préimages enveloppe+precommit byte-exactes (banc fonctionnel)" \
  "$(echo "$PAR" | grep '✗✗' | head -4)"
grep -q '_PRECOMMIT_HEADER' "$MP/reliquary/miner/submitter.py"
gate G4c $? "submitter : _PRECOMMIT_HEADER X-Reliquary-Precommit"
if grep -q '_DRAND_BOUNDARY_SAFETY_SECONDS' "$MP/reliquary/miner/submitter.py"; then
  echo "  ✅ G4d PASS — garde-fous drand boundary portés"
else
  echo "  ⚠️ G4d WARN — drand boundary guards absents (robustesse timing, pas conformité ; moins critique à collection=300s)"
fi
# Câblage engine : le profile part dans la signature ET la requête.
grep -qE 'sign_envelope' "$MP/reliquary/miner/engine.py" && \
  sed -n '/envelope_sig = sign_envelope/,/\.hex()/p' "$MP/reliquary/miner/engine.py" | grep -q 'generation_profile_id'
gate G4e $? "engine : generation_profile_id passé à sign_envelope"
sed -n '/request = BatchSubmissionRequest/,/^\s*)/p' "$MP/reliquary/miner/engine.py" | grep -q 'generation_profile_id'
gate G4f $? "engine : generation_profile_id sur BatchSubmissionRequest"

# ---------- G5 : engine v3 ----------
echo "--- G5 engine ---"
# Le profil v3 live a force_answer=true → notre phase-2 inconditionnelle est
# conforme AUJOURD'HUI. Le gate BFT_FORCE_ANSWER n'est requis que si le
# validateur flippe au clean-cap plus tard → WARN, pas FAIL.
if grep -qE "bft_applicable and BFT_FORCE_ANSWER" "$MP/reliquary/miner/engine.py"; then
  echo "  ✅ G5a PASS — gate phase-2 BFT_FORCE_ANSWER porté (prêt pour clean-cap)"
else
  echo "  ⚠️ G5a WARN — gate BFT_FORCE_ANSWER absent (OK pour le live force_answer=true ; à porter si le validateur flippe clean-cap)"
fi
H=$(grep -rn '"reliquary-forced-seed-v2"' "$MP/reliquary/" --include='*.py' \
    | grep -v tests | grep -v "f\"reliquary-forced-seed" | head -3)
gate G5b $([ -z "$H" ] && echo 0 || echo 1) "aucun domaine v2 en dur hors tests" "$H"
# Test FONCTIONNEL : la valeur réellement émise sur le wire (engine.
# wire_protocol_version, utilisée par BatchSubmissionRequest) doit être 3 sous
# le profil v3. Attrape aussi le piège wire_v2_enabled() qui forcerait 2.
G5OUT=$(cd "$MP" && RELIQUARY_PROTOCOL_PROFILE=qwen35-4b-auction-v3 PYTHONPATH=. \
  python3 -c "from reliquary.miner.engine import wire_protocol_version; print(wire_protocol_version())" 2>&1 | tail -1)
gate G5c $([ "$G5OUT" = "3" ] && echo 0 || echo 1) "wire_protocol_version()==3 sous profil v3" "renvoie: $G5OUT"

# ---------- G6 : config lancement ----------
echo "--- G6 launchers (scripts/ + ops/) ---"
for L in /root/subnet81/scripts/launch_miner.sh "$MP/ops/launch_miner.sh"; do
  N=$(basename $(dirname "$L"))/launch_miner.sh
  # Défaut du code = v3 ; le launcher ne doit PAS forcer un rollback v2.
  RBV=$(grep -oE 'RELIQUARY_PROTOCOL_VERSION=[0-9]+' "$L" | grep -oE '[0-9]+' | head -1)
  gate "G6a[$N]" $([ -z "$RBV" ] || [ "$RBV" = "3" ] && echo 0 || echo 1) \
    "pas de rollback v2 forcé (RELIQUARY_PROTOCOL_VERSION absent ou 3)" "trouvé=$RBV"
  grep -q 'qwen3.5-4b-reliquary-v4' "$L"
  gate "G6b[$N]" $? "checkpoint 4B v4"
  # Deux stratégies valides sous v3 :
  #  (a) cap plein 16384 ;
  #  (b) cap COURT délibéré (étude §5 : gagnants 600-1000 tok) — sûr SEULEMENT
  #      avec la garde locale de troncature (max_truncated_allowed défaut 0 :
  #      on ne soumet que des groupes 100% EOS → jamais BAD_TERMINATION).
  MNT=$(grep -oE 'MAX_NEW_TOKENS[=:-]+[0-9]+' "$L" | grep -oE '[0-9]+' | tail -1)
  if [ "${MNT:-0}" -ge 16384 ]; then
    gate "G6c[$N]" 0 "cap plein (${MNT})"
  elif grep -q "def max_truncated_allowed" "$MP/reliquary/miner/engine.py" && \
       grep -q "too_many_truncated" "$MP/reliquary/miner/engine.py"; then
    gate "G6c[$N]" 0 "cap court ${MNT:-défaut} + garde troncature locale (étude §5)"
  else
    gate "G6c[$N]" 1 "cap court SANS garde troncature" "cap=${MNT:-absent}"
  fi
done

# ---------- G9 : simulateur validateur hors-ligne ----------
# Fait juger la sortie RÉELLE de notre fix par le CODE du validateur upstream
# (pas une ré-implémentation). Reproduit les 3 modes de panne des 21 rejets
# bad_termination et vérifie qu'ils sont rejetés, et que notre correctif passe.
echo "--- G9 simulateur validateur (code upstream réel) ---"
SIM=$(python3 /root/subnet81/scripts/simulate_validator.py 2>&1)
echo "$SIM" | grep -q "Tous les scénarios se comportent comme attendu"
gate G9 $? "modes de panne rejetés + sortie du fix acceptée" \
  "$(echo "$SIM" | grep -E "ATTENDU|Traceback" | head -3)"

# ---------- G7 : tests (optionnel, --tests) ----------
if [ "${1:-}" = "--tests" ]; then
  echo "--- G7 pytest sous profil v3 ---"
  (cd "$MP" && RELIQUARY_PROTOCOL_PROFILE=qwen35-4b-auction-v3 PYTHONPATH=. \
    python3 -m pytest tests/test_forced_seed_*.py tests/test_bft_*.py \
    tests/test_generate_m_rollouts_bft.py tests/test_wire_v2_port.py \
    -q 2>&1 | tail -5)
else
  echo "--- G7 : lancer avec --tests pour la suite pytest ---"
fi

echo "======================================================================"
[ $FAIL -eq 0 ] && echo "VERDICT GLOBAL : ✅ TOUS GATES AUTOMATIQUES PASS (reste G7 --tests et G8 GPU)" \
                || echo "VERDICT GLOBAL : ❌ DES GATES ÉCHOUENT — port incomplet"
exit $FAIL
