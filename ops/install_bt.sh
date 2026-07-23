#!/bin/bash
# Stack bittensor par-dessus le venv vLLM (recette validée 2026-07-17, H100).
# 10.5.0 obligatoire: v11 supprime bt.AsyncSubtensor; 10.0-10.2 échouent au
# metagraph.sync() (netuid attendu en Composite NetUid(u16) par le runtime finney).
V=/workspace/venv/bin
$V/pip install -q "bittensor==10.5.0" pyarrow 2>&1 | tail -3
$V/pip install -q "flash-linear-attention==0.5.0" 2>&1 | tail -2
# async-substrate-interface 2.2.1 exige cyscale et refuse scalecodec
# (RuntimeError conflit de namespace à l'import) -> virer scalecodec.
$V/pip uninstall -y -q scalecodec 2>&1 | tail -1
$V/pip install -q --force-reinstall --no-deps cyscale 2>&1 | tail -1
echo "=== versions ==="
$V/pip list 2>/dev/null | grep -iE "^(bittensor|async-substrate-interface|cyscale|scalecodec|pyarrow|flash-linear-attention|torch|transformers|vllm) "
