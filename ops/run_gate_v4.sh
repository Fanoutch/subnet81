#!/bin/bash
# Gate forced-seed v4 sur CETTE carte — contrôle OBLIGATOIRE avant de miner.
# Elle vérifie que nos tokens sont ceux que le validateur reconstruira par
# teacher-forcing. Planchers protocole : 0.80 groupe / 0.75 pire rollout.
# ⚠️ On ne source PAS bench_env.sh : il force SMOKE_CKPT sur le 2B périmé.
export PYTHONPATH=/workspace/reliquary-miner-priv
export HF_HOME=/workspace/hf
export GRAIL_ATTN_IMPL=sdpa
export CUDA_HOME=/workspace/venv/lib/python3.12/site-packages/nvidia/cu13
export PATH=/workspace/venv/bin:$CUDA_HOME/bin:$PATH
export VLLM_USE_DEEP_GEMM=0
export VLLM_DEEP_GEMM_WARMUP=skip
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# v5 depuis le 24/08 : le domaine forced-seed est
# `reliquary-forced-seed-v{PROTOCOL_VERSION}` (verifie en source upstream
# cba84ce, constants.py:1292). Lancer le gate en v4 validerait le domaine v4
# alors qu'on mine en v5 — le controle passerait pour de mauvaises raisons.
# ⚠️ MAJ 12/09 — V1/fill-closed : on mine en 6. Une gate lancée en 5 validerait
# le domaine `reliquary-forced-seed-v5` et passerait pour de mauvaises raisons.
export RELIQUARY_PROTOCOL_VERSION=${RELIQUARY_PROTOCOL_VERSION:-6}
unset SMOKE_CKPT SMOKE_REV          # -> modèle par défaut du protocole v4
export GATE_MAX_NUM_SEQS=64         # >64 dépasse les blocs Mamba du 4B
P=/workspace/venv/bin/python
# 12/09 : la gate vivait en fichier NON VERSIONNÉ dans /workspace. Sur une box
# neuve elle est absente et le script se terminait en 20 s, sections VIDES,
# sans un seul message d'erreur (stderr filtré par le grep) — un contrôle
# OBLIGATOIRE qui passe à vide est pire que pas de contrôle. On prend la copie
# versionnée du dépôt, et on ÉCHOUE bruyamment si elle manque.
G=/workspace/reliquary-miner-priv/ops/validate_vllm_forced_seed_group.py
[ -f "$G" ] || { echo "ABORT: gate introuvable ($G)" >&2; exit 1; }

echo "##### GATE EAGER (référence de conformité) #####"
GATE_EAGER=1 $P "$G" 2>&1 \
  | grep -aE "\[gate\]|model|Error|Traceback|Assertion|No such file|ModuleNotFound"
echo
echo "##### GATE CUDA GRAPHS (le mode de production) #####"
GATE_EAGER=0 $P "$G" 2>&1 \
  | grep -aE "\[gate\]|model|Error|Traceback|Assertion|No such file|ModuleNotFound"
echo "GATES_TERMINEES"
