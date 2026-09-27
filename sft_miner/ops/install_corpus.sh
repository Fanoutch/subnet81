#!/bin/bash
export HF_HOME=/workspace/hf PIP_CONFIG_FILE=/dev/null PIP_BREAK_SYSTEM_PACKAGES=1
log(){ echo "$(date -u +%T) $*"; }
dl(){
  python3 -m pip install -q "huggingface_hub[hf_xet]" >/dev/null 2>&1
  python3 - <<P
from huggingface_hub import snapshot_download
p=snapshot_download("Qwen/Qwen3.8-27B", revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
print("MODEL_OK", p)
P
}
venv(){
  python3 -m venv /workspace/venv_corpus && . /workspace/venv_corpus/bin/activate
  pip install -q --upgrade pip
  pip install -q vllm==0.30.0 || { echo VENV_ECHEC_vllm; return 1; }
  pip install -q --no-deps -e /workspace/reliquary_upstream
  pip install -q "transformers==5.10.4" "bittensor==10.5.0" "async-substrate-interface==2.2.1" "cyscale==0.5.0" aiobotocore boto3 typer rich tenacity pyarrow datasets "uvicorn[standard]>=0.52,<0.53" fastapi bitsandbytes httpx pytest pytest-asyncio 2>&1 | tail -3
  python -c "import vllm,torch,transformers,reliquary.miner.corpus_miner as c; print(\"VENV_OK\", vllm.__version__, torch.__version__, transformers.__version__, torch.cuda.is_available())"
}
log début
dl > /workspace/dl_corpus.log 2>&1 &
venv > /workspace/venv_corpus.log 2>&1
log "venv: $(grep -E \"VENV_OK|ECHEC\" /workspace/venv_corpus.log)"
wait; log "modèle: $(grep MODEL_OK /workspace/dl_corpus.log)"
log INSTALL_DONE
