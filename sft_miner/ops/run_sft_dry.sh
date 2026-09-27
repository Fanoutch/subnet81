#!/bin/bash
export HF_HOME=/workspace/hf VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_V2_MODEL_RUNNER=0
export RELIQUARY_TASK_CONTRACT=/workspace/corpus-code-v1.contract.json PYTHONPATH=/workspace/sft_src
cd /workspace/reliquary_upstream
/workspace/venv_corpus/bin/python -m sft_miner.miner --dry-run --duration 1800 --max-in-flight 32 --max-ahead 256 --admit-below 0.6 --dump 64
echo SFT_DRY_EXIT=$?
export BENCH_MODEL=$(ls -d /workspace/hf/hub/models--Qwen--Qwen3.8-27B/snapshots/*/ | head -1) BENCH_OUT=/workspace/sft_dump.json
/workspace/venv_corpus/bin/python /workspace/corpus_bench.py check
