"""vLLM smoke test for ReliquaryForge/qwen3.5-2b-reliquary (subnet 81, cot-2b/v7).

Loads the checkpoint as the FULL Qwen3_5ForConditionalGeneration (weights map
correctly, same class family the validator's HF path uses), after injecting the
base model's preprocessor_config.json so vLLM's processor load succeeds
(root cause #1). We never pass images and force text-only mode via
limit_mm_per_prompt (root cause #2) -> vision tower loaded but dormant.

Run via a .py file, NOT heredoc/stdin: vLLM v1 spawns an engine-core subprocess
that re-imports argv[0]; stdin becomes FileNotFoundError in the worker.
Env vars (CUDA_HOME/PATH/VLLM_* ) must be exported by the caller (bringup.sh)
so the spawned workers inherit them.
"""
import os
import shutil

from huggingface_hub import hf_hub_download, snapshot_download

# Defaults track the live validator's published checkpoint (GET /state ->
# checkpoint_repo_id / checkpoint_revision), so the smoke test downloads the
# same weights the probe and the miner will use. Override to pin an older one.
CKPT = os.environ.get("SMOKE_CKPT", "ReliquaryForge/qwen3.5-2b-reliquary-v2")
REV = os.environ.get("SMOKE_REV", "cdc9daee91a8f00b649202fd4c45bd90a1b3f3d6")
BASE = os.environ.get("SMOKE_BASE", "Qwen/Qwen3.5-2B")


def main():
    local = snapshot_download(CKPT, revision=REV)
    print("[smoke] checkpoint at", local, flush=True)

    # root cause #1: inject the base image-processor config if absent.
    pp = os.path.join(local, "preprocessor_config.json")
    if not os.path.exists(pp):
        try:
            base_pp = hf_hub_download(BASE, "preprocessor_config.json")
            shutil.copy(base_pp, pp)
            print("[smoke] injected preprocessor_config.json from base", flush=True)
        except Exception as e:  # noqa: BLE001
            print("[smoke] WARN base preprocessor fetch failed:", e, flush=True)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=local,
        gpu_memory_utilization=0.6,
        max_model_len=4096,
        dtype="bfloat16",
        enforce_eager=True,              # validated env; avoids cudagraph capture
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 0, "video": 0},  # root cause #2: text-only
        # root cause #8 (2026-07-17): the FlashInfer GDN prefill kernel is
        # JIT-compiled via ninja/ptxas and dies (headers-incompatible on nvcc!=
        # CUDART, then ptxas-fatal). Force the Triton/FLA backend to skip the JIT
        # entirely. This is ALSO the validator's own path: /runtime-contract
        # reports fla 0.5.0 + qwen35_fla_chunk=true, i.e. GRAIL proves via FLA,
        # so triton is the parity-aligned backend, not a fallback.
        additional_config={"gdn_prefill_backend": "triton"},
    )
    print("[smoke] ENGINE LOADED OK", flush=True)

    # v7 sampler params (cot-2b): T=0.6 / top_p=0.95 / top_k=20.
    out = llm.generate(
        ["What is 12 times 8? Answer:"],
        SamplingParams(max_tokens=64, temperature=0.6, top_p=0.95, top_k=20),
    )
    print("[smoke] OUTPUT:", repr(out[0].outputs[0].text[:300]), flush=True)
    print("[smoke] SUCCESS — vLLM generation works end-to-end", flush=True)


if __name__ == "__main__":
    main()
