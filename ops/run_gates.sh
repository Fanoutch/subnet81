#!/bin/bash
# Les deux gates de conformite forced-seed, journalises correctement
# (le tee DOIT etre dans la commande, pas autour de tmux).
source /workspace/bench_env.sh
export SMOKE_CKPT=ReliquaryForge/qwen3.5-2b-reliquary-v3
export SMOKE_REV=5db7a1f55218c68fd3dff1d927b02c7508684c9e
P=/workspace/venv/bin/python

echo "##### GATE 1A : vLLM forced-seed, EAGER (reference prod) #####"
GATE_EAGER=1 $P /workspace/validate_vllm_forced_seed_group.py 2>&1 | grep -E "\[gate\]|Error|Traceback"
echo "exit=$?"

echo
echo "##### GATE 1B : vLLM forced-seed, CUDA GRAPHS (+56% au banc) #####"
GATE_EAGER=0 $P /workspace/validate_vllm_forced_seed_group.py 2>&1 | grep -E "\[gate\]|Error|Traceback"
echo "exit=$?"

echo
echo "##### GATE 2 : phase-2 batchee entre prompts (vrai critere) #####"
$P /workspace/validate_bft_multi_seedconsistency.py 2>&1 | grep -E "\[gate\]|Error|Traceback"
echo "exit=$?"
echo "##### FIN #####"
