#!/bin/bash
set -e
source /workspace/venv/bin/activate
F=/workspace/reliquary-miner-priv/ops/prod_backup_2026-08-07/pip_freeze.txt
# env figé COMPLET, versions exactes, pas de résolveur, index pytorch cu130 pour torch/nvidia-cu13
pip install --no-deps --extra-index-url https://download.pytorch.org/whl/cu130 -r <(grep -vE "^(-e |.* @ )" "$F")
python -c "import torch,vllm,transformers,bittensor,flashinfer; print('OK torch',torch.__version__,'vllm',vllm.__version__,'tf',transformers.__version__,'bt',bittensor.__version__)"
echo "=====INSTALL_DONE====="
