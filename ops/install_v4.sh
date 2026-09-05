#!/bin/bash
# Reconstruction complète après effacement de la box (20/08 21h20).
# Modèle CORRIGÉ : Qwen/Qwen3-4B-Base @906bfd4b (le validateur live l'exige ;
# l'ancien install.sh visait Qwen3.5-4B, périmé depuis le passage en v4).
set -e
cd /workspace
echo "=== venv ==="
python3 -m venv /workspace/venv
source /workspace/venv/bin/activate
pip install -q --upgrade pip
echo "=== dépendances (sans torch/nvidia) ==="
pip install --no-input -q --no-deps -r <(grep -vE "^(nvidia-|torch)" \
  /workspace/reliquary-miner-priv/ops/prod_backup_2026-08-07/pip_freeze.txt)
echo "=== torch cu130 ==="
pip install --no-input -q torch==2.11.0+cu130 --index-url https://download.pytorch.org/whl/cu130
pip install --no-input -q --no-deps --extra-index-url https://download.pytorch.org/whl/cu130 \
  torchvision==0.26.0 torchaudio==2.11.0 torch_c_dlpack_ext==0.1.5 \
  nvidia-cuda-cccl==13.3.3.4.1 nvidia-cuda-crt==13.3.73 nvidia-cuda-nvcc==13.0.88 \
  nvidia-cuda-tileiras==13.2.86 nvidia-cudnn-frontend==1.26.0 \
  nvidia-cutlass-dsl==4.5.2 nvidia-cutlass-dsl-libs-base==4.5.2 nvidia-cutlass-dsl-libs-cu13==4.5.2 \
  nvidia-ml-py==13.610.43 nvidia-nvvm==13.2.86
echo "=== contrôle des imports ==="
python -c "import torch, vllm, transformers, bittensor; print('IMPORTS_OK', torch.__version__, vllm.__version__)"
ls /workspace/venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc >/dev/null && echo NVCC_OK
echo "=== modèle Qwen3-4B-Base ==="
export HF_HOME=/workspace/hf
python - <<'PYEOF'
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen3-4B-Base",
                      revision="906bfd4b4dc7f14ee4320094d8b41684abff8539")
print("MODEL_PREFETCHED", p)
PYEOF
echo INSTALL_DONE
