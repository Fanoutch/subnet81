#!/bin/bash
# ============================================================================
# vLLM bring-up for ReliquaryForge/qwen3.5-2b-reliquary (subnet 81, cot-2b/v7)
# ----------------------------------------------------------------------------
# The 2B is a MULTIMODAL HYBRID (Qwen3_5ForConditionalGeneration: vision tower
# + GDN linear-attention text backbone). vLLM 0.10.x can't load it at all;
# 0.22/0.24 load it but the GDN kernel JIT-compiles via flashinfer and dies on
# an nvcc<->headers mismatch. This script encodes the 7 root causes from
# reference_vllm_qwen35_2b_bringup + the priority FIX (align nvcc to cu130).
#
# vLLM does NOT affect GRAIL parity (validator proves in pure HF, no vllm pin),
# so we are FREE to use the cu130 / vllm 0.24 / transformers 5.9 stack here.
# vLLM = generation acceleration only.
#
# RUN ON THE GPU BOX, from a directory containing vllm_smoke_test.py.
# Everything is ephemeral on a fresh box (venv/model-dir/ninja) -> re-run whole.
# ============================================================================
set -u
# Paths are overridable: on Lium boxes /root is a small (~30G) loop device while
# the overlay under /workspace has the real space. venv + HF cache + weights do
# not fit in 30G.
VENV=${VENV:-/root/venv}
PY=$VENV/bin/python
PIP=$VENV/bin/pip
CKPT_LOCAL=${CKPT_LOCAL:-/root/qwen3.5-2b-reliquary-local}

log(){ echo "[bringup $(date -u +%H:%M:%S)] $*"; }

# --- 0. sanity: GPU + driver recent enough for cu130 (needs ~R580+) ---------
log "nvidia-smi:"; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || {
  log "FATAL: no GPU visible"; exit 1; }

# --- 1. venv (reuse if torch+cu130 already present, else create) ------------
if [ ! -x "$PY" ]; then
  log "creating venv $VENV"; python3 -m venv "$VENV"; "$PIP" install -U pip
fi

# --- 2. the cu130 stack that LOADS qwen3_5 ----------------------------------
#     vLLM 0.24.0 is the only line allowing transformers>=5.5.3 (needed for
#     Qwen3.5). It pulls a compatible flashinfer-python + torch 2.11+cu130.
NEED=$("$PY" -c "import vllm,transformers,torch;print(vllm.__version__,transformers.__version__,'cu130' in torch.__version__)" 2>/dev/null)
if [ "$NEED" != "0.24.0 5.9.0 True" ]; then
  log "installing torch cu130 + vllm 0.24 + transformers 5.9 (~10 min)"
  "$PIP" install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu130 2>&1 | tail -3
  "$PIP" install "vllm==0.24.0" "transformers==5.9.0" 2>&1 | tail -5
else
  log "stack already present: $NEED"
fi

# --- 3. THE FIX: align nvcc to the installed cccl/crt HEADERS ---------------
#     flashinfer's GDN JIT compile dies with "CUDA compiler and CUDA toolkit
#     headers are incompatible". EMPIRICAL (2026-07-17 box): the pip stack ships
#     nvidia-cuda-nvcc==13.2.86 but nvidia-cuda-crt/cccl==13.3.x headers -> 13.2
#     compiler vs 13.3 headers. Pin nvcc to the header version. (The old
#     `nvidia-cuda-nvcc-cu13` name is a 0.0.1 stub; the real pkg is
#     `nvidia-cuda-nvcc`.) HDR_VER auto-detects the crt version so this survives
#     a stack bump.
# EMPIRICAL (2026-07-17): the guard in flashinfer's BUNDLED cccl header
# (flashinfer/data/cccl/.../cuda_toolkit.h) is `!_CCCL_CUDACC_EQUAL(CUDART/1000,
# ...)` i.e. it requires  version(nvcc) == CUDART_VERSION  where CUDART comes
# from the nvidia-cuda-RUNTIME pkg (13.0.96 -> CUDART_VERSION 13000 -> 13.0).
# So nvcc must equal the RUNTIME version, NOT the crt/cccl header version.
# (Aligning to crt 13.3 was wrong -> still failed 13.3 != CUDART 13.0.)
RT_VER=$("$PIP" show nvidia-cuda-runtime 2>/dev/null | awk '/^Version:/{print $2}')
RT_MM=$(echo "${RT_VER:-13.0.96}" | cut -d. -f1,2)   # e.g. 13.0
NVCC_VER=$("$PIP" index versions nvidia-cuda-nvcc 2>/dev/null | grep -oE "${RT_MM}\.[0-9]+" | head -1)
NVCC_VER=${NVCC_VER:-13.0.88}
log "aligning nvcc to CUDART/runtime version ${RT_MM} -> pinning nvidia-cuda-nvcc==${NVCC_VER}"
"$PIP" install "nvidia-cuda-nvcc==${NVCC_VER}" 2>&1 | tail -3
"$PIP" install ninja 2>&1 | tail -1   # root cause #5

# --- 4. toolchain env (root causes #3 #4 #6) --------------------------------
export CUDA_HOME=$VENV/lib/python3.12/site-packages/nvidia/cu13
export PATH=$VENV/bin:$CUDA_HOME/bin:$PATH
export VLLM_USE_DEEP_GEMM=0 VLLM_DEEP_GEMM_WARMUP=skip   # no FP8 (bf16 model); vLLM 0.24 makes WARMUP an enum: skip|full|relax (0.22 accepted 0/1)
export VLLM_USE_FLASHINFER_SAMPLER=0                  # native pytorch sampler
log "nvcc in use: $(command -v nvcc) -> $(nvcc --version 2>/dev/null | grep -oE 'release [0-9.]+' || echo MISSING)"

# --- 5. diagnostics ---------------------------------------------------------
"$PY" -c "import vllm,transformers,torch;print('[ver] vllm',vllm.__version__,'tf',transformers.__version__,'torch',torch.__version__)"
"$PY" -c "from vllm.model_executor.models.registry import ModelRegistry as R;print('[ver] qwen3_5 registered:', 'Qwen3_5ForConditionalGeneration' in R.get_supported_archs())"

# --- 6. run the smoke test (loads via condgen path + generates) -------------
#     preprocessor_config.json injection (root cause #1) + text-only mode
#     (root cause #2) are handled INSIDE vllm_smoke_test.py.
log "=== SMOKE TEST ==="
"$PY" "$(dirname "$0")/vllm_smoke_test.py"
RC=$?
log "smoke test exit=$RC"

# --- 7. fallback ladder if step 6 still dies on GDN compile -----------------
if [ $RC -ne 0 ]; then
  cat <<'EOF'
[bringup] SMOKE TEST FAILED. Fallback ladder (try in order):
  A. Confirm the fix took: `nvcc --version` must match the crt/cccl header
     version (see `pip show nvidia-cuda-crt`). If they differ:
     `pip install "nvidia-cuda-nvcc==<that version>"` then re-export PATH.
  B. Prebuilt flashinfer cu13 wheel (skip JIT entirely):
        pip install flashinfer-python --index-url \
          https://flashinfer.ai/whl/cu130/torch2.11/
  C. Nuke JIT cache if a stale bad compile is cached:
        rm -rf ~/.cache/flashinfer ~/.cache/vllm
  D. Last resort: official vLLM Docker image (self-consistent toolchain),
     mount the model dir, run the same smoke test inside.
EOF
fi
exit $RC
