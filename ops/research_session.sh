#!/bin/bash
# Session de recherche forced-seed sur la box de PROD pendant une panne du
# validateur (22/09). Le mineur est arrêté, les bancs tournent, et un CHIEN DE
# GARDE relance le mineur dès que le validateur répond (toutes les 10 s).
# Le mineur est TOUJOURS relancé à la fin, même si les bancs échouent.
# Lancer détaché : setsid nohup bash /workspace/research_session.sh &
LOG=/workspace/research_fs.log
OUT=/workspace/bench_fs_decomp.jsonl
U="http://62.238.81.36:8000/state?env=opencodeinstruct"
PAT="cli.""main mine"
FLAG=/tmp/research_validator_back
rm -f "$FLAG"
say(){ echo "=== $(date -u +%FT%TZ) $*" >> "$LOG"; }

PID=$(pgrep -f "$PAT" | head -1)
ENVF=/workspace/miner_environ_research.bin
[ -n "$PID" ] && cp /proc/$PID/environ "$ENVF"
MODEL=$(grep -ao "model='[^']*'" /workspace/miner.log | tail -1 | cut -d"'" -f2)
say "début (pid mineur ${PID:-absent}) modèle=$MODEL"
[ -s "$ENVF" ] && [ -n "$MODEL" ] || { say "environ/modèle introuvable -> abandon"; exit 1; }
pkill -9 -f "$PAT"; pkill -9 -f EngineCore; sleep 5
nvidia-smi --query-gpu=memory.used --format=csv,noheader >> "$LOG"

# chien de garde : validateur de retour -> tout couper et relancer le mineur
(
  while :; do
    c=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$U")
    if [ "$c" = "200" ]; then
      touch "$FLAG"; say "VALIDATEUR DE RETOUR -> arrêt des bancs, relance du mineur"
      pkill -9 -f "ops/bench_fs_h100.py"; pkill -9 -f EngineCore; sleep 3
      bash /workspace/restart_miner.sh >> "$LOG" 2>&1; say "mineur relancé (garde)"
      exit 0
    fi
    [ -f /tmp/research_done ] && exit 0
    sleep 10
  done
) &
GUARD=$!

mapfile -d '' ENVV < "$ENVF"
run(){  # run <étiquette> VAR=val ...  — journal complet + garde anti-blocage
  local tag=$1; shift
  [ -f "$FLAG" ] && return
  mkdir -p /workspace/research_runs
  local L=/workspace/research_runs/$tag.log
  say "banc $tag : $* (journal $L)"
  (cd /workspace/reliquary-miner-priv && exec env -i "${ENVV[@]}" PYTHONPATH=. \
     BENCH_MODEL="$MODEL" BENCH_OUT="$OUT" "$@" \
     timeout -k 5 1500 /workspace/venv/bin/python ops/bench_fs_h100.py) > "$L" 2>&1 &
  local BP=$! last=$(date +%s) n0 n1
  n0=$(wc -l < "$OUT" 2>/dev/null || echo 0)
  while kill -0 $BP 2>/dev/null; do
    sleep 10
    [ -f "$FLAG" ] && break
    n1=$(wc -l < "$OUT" 2>/dev/null || echo 0)
    if [ "$n1" != "$n0" ]; then n0=$n1; last=$(date +%s); fi
    # chargement du moteur (~30-60 s) toléré : 240 s sans nouvelle mesure = blocage
    if [ $(( $(date +%s) - last )) -gt 240 ]; then
      say "banc $tag BLOQUÉ (aucune mesure depuis 240 s) -> coupé"
      pkill -9 -f "ops/bench_fs_h100.py"; break
    fi
  done
  wait $BP 2>/dev/null
  grep -a "\[bench\]" "$L" | grep -a -E "meilleur|total" >> "$LOG"
  pkill -9 -f EngineCore; sleep 3
}
# 1. décomposition à 160 séquences (taille des bakes)
run B1-fs  BENCH_FAMILY=fs BENCH_MODES=libre_t1,libre_greedy,noop,fs,fast BENCH_PROMPTS=10 BENCH_LENS=512 BENCH_REPS=3
run B1-nu  BENCH_FAMILY=nu BENCH_MODES=nu_greedy,nu_t1 BENCH_PROMPTS=10 BENCH_LENS=512 BENCH_REPS=3
# 2. grille tailles × longueurs
run B2-fs  BENCH_FAMILY=fs BENCH_MODES=libre_greedy,noop,fs,fast BENCH_PROMPTS=2,4,10,16 BENCH_LENS=512,2048 BENCH_REPS=2
run B2-nu  BENCH_FAMILY=nu BENCH_MODES=nu_greedy BENCH_PROMPTS=2,4,10,16 BENCH_LENS=512,2048 BENCH_REPS=2

touch /tmp/research_done
if [ ! -f "$FLAG" ]; then
  say "bancs terminés -> relance du mineur (validateur toujours absent : il attendra)"
  kill $GUARD 2>/dev/null
  bash /workspace/restart_miner.sh >> "$LOG" 2>&1
fi
say "fin"
