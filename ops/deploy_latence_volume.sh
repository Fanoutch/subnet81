#!/usr/bin/env bash
# Déploiement groupé du 20/08 : latence + file d'envoi + bonus de volume.
#
# UN SEUL redémarrage pour les deux leviers (le mineur perd 2-3 fenêtres au
# rechargement du moteur vLLM ; les grouper évite d'en payer le double).
#
# À LANCER DEPUIS LA DEV BOX. Ne fait RIEN d'irréversible avant le restart :
# les fichiers sont copiés, la sauvegarde est prise, puis le mineur redémarre.
#
#   bash ops/deploy_latence_volume.sh            # déploie
#   bash ops/deploy_latence_volume.sh --dry-run  # copie + vérifie, SANS restart
#
# ROLLBACK : les fichiers d'origine sont dans /workspace/backup_<horodatage>/,
# et le point de retour git est le commit 545f4e1 (vérifié identique à la prod).
set -euo pipefail

BOX="${BOX:-root@38.255.28.21}"
PORT="${PORT:-20089}"
SRC="${SRC:-/root/subnet81/.worktrees/miner-priv-port-v4-dapo}"
DST=/workspace/reliquary-miner-priv
DRY=0
VOLUME=1
for a in "$@"; do
  case "$a" in
    --dry-run)     DRY=1 ;;
    --sans-volume) VOLUME=0 ;;   # ne déploie QUE latence + file d'envoi
  esac
done

ssh_() { ssh -o StrictHostKeyChecking=no -p "$PORT" "$BOX" "$@"; }
scp_() { scp -o StrictHostKeyChecking=no -P "$PORT" "$@"; }

STAMP=$(date -u +%Y%m%d_%H%M%S)
BK="/workspace/backup_$STAMP"

echo "══ 1. Le validateur répond-il ? (ne pas redémarrer pendant sa panne) ══"
code=$(ssh_ "curl -s -o /dev/null -w '%{http_code}' --max-time 5 \
        http://209.20.157.231:8080/health" || echo 000)
echo "   /health -> HTTP $code"
if [ "$code" != "200" ]; then
  echo "   ⛔ validateur indisponible — déploiement ANNULÉ (rien n'a été touché)."
  exit 1
fi

echo "══ 2. Sauvegarde de la version en place ══"
ssh_ "mkdir -p $BK && cp $DST/reliquary/miner/engine.py \
      $DST/reliquary/miner/submitter.py \
      $DST/reliquary/miner/prompt_predictor.py \
      /workspace/launch_miner_v4.sh $BK/"
echo "   -> $BK"

echo "══ 3. Copie des fichiers ══"
scp_ "$SRC/reliquary/miner/engine.py"           "$BOX:$DST/reliquary/miner/engine.py"
scp_ "$SRC/reliquary/miner/submitter.py"        "$BOX:$DST/reliquary/miner/submitter.py"
scp_ "$SRC/reliquary/miner/prompt_predictor.py" "$BOX:$DST/reliquary/miner/prompt_predictor.py"
scp_ "$SRC/ops/launch_miner_v4.sh"              "$BOX:/workspace/launch_miner_v4.sh"
if [ "$VOLUME" = "1" ]; then
  scp_ "/root/subnet81/data/volume_v1.json"     "$BOX:/workspace/volume_v1.json"
else
  # Sans le fichier, le chargeur renvoie None et le tri reste byte-identique :
  # on isole ainsi la mesure des 2 fix de latence/file d'envoi.
  ssh_ "rm -f /workspace/volume_v1.json"
  echo "   --sans-volume : bonus de volume INACTIF (tri inchangé)"
fi

echo "══ 4. Vérifications AVANT redémarrage ══"
ssh_ "cd $DST && python3 -c 'import ast,sys
for f in (\"reliquary/miner/engine.py\",\"reliquary/miner/submitter.py\",
          \"reliquary/miner/prompt_predictor.py\"):
    ast.parse(open(f).read())
print(\"   syntaxe OK\")'"
if [ "$VOLUME" = "1" ]; then
ssh_ "cd $DST && RELIQUARY_VOLUME_MODEL=/workspace/volume_v1.json python3 -c '
import json, sys
sys.path.insert(0, \".\")
from reliquary.miner.prompt_predictor import volume_score
m = json.load(open(\"/workspace/volume_v1.json\"))
a = volume_score(m, \"Implement a segment tree with lazy propagation for range minimum queries\")
b = volume_score(m, \"Return the sum of two integers.\")
assert a > b, (a, b)
print(f\"   modèle de volume OK ({len(m[\"weights\"])} poids, long={a:.2f} > court={b:.2f})\")'"
fi

if [ "$DRY" = "1" ]; then
  echo "══ --dry-run : fichiers en place, mineur NON redémarré ══"
  echo "   (la version en cours d'exécution tourne encore sur l'ancien code)"
  exit 0
fi

echo "══ 5. Redémarrage ══"
ssh_ "bash /workspace/restart_miner.sh" || {
  echo "   ⛔ restart en échec — restaurer avec :"
  echo "      ssh -p $PORT $BOX 'cp $BK/* $DST/reliquary/miner/ && bash /workspace/restart_miner.sh'"
  exit 1
}

echo "══ 6. Confirmation que les deux leviers sont ACTIFS ══"
sleep 45
ssh_ "grep -aE 'bonus de volume ACTIF|malus anti-court ACTIF' /workspace/miner.log | tail -3"
ssh_ "grep -acE 'MAX_INFLIGHT|fire_on_append' /workspace/miner.log | tail -1" || true
echo
echo "Déployé. Sauvegarde : $BK"
echo "À surveiller sur ~40 fenêtres : tokens/groupe, part >=6000 tok, écart"
echo "prête->partie (doit s'effondrer), payées/fenêtre."
