"""Task 4 GATE (robust): generate a full M-rollout group with vLLM forced-seed,
teacher-force each rollout through HF, report per-rollout AND group-aggregate
seed_consistency — exactly the floors the validator enforces (0.75 rollout /
0.80 group). Decides whether a real submission would survive the vLLM path.
"""
import os
import torch

from huggingface_hub import snapshot_download
from reliquary.shared.modeling import load_text_generation_model, load_tokenizer
from reliquary.environment.forced_sampling import u_at, seed_consistency
from reliquary.miner.vllm_forced_seed import (
    build_forced_seed_logitsproc_class, FORCED_SEED_EXTRA_KEY, forced_seed_extra_args)
from reliquary import constants as c

CKPT = os.environ.get("SMOKE_CKPT", "ReliquaryForge/qwen3.5-2b-reliquary-v2")
REV = os.environ.get("SMOKE_REV", "cdc9daee91a8f00b649202fd4c45bd90a1b3f3d6")
RANDOMNESS = "deadbeef" * 8
PROMPT_IDX = 424242
CKPT_HASH = "vllm-gate-hash"
M = 8
MAX_NEW = 512
PROMPT = ("A train travels 60 km in the first hour, then increases its speed by "
          "15 km/h each subsequent hour. Reason step by step how far it has "
          "traveled after 5, 8, and 12 hours. Show all arithmetic. Answer:")


def main():
    local = snapshot_download(CKPT, revision=REV)
    tok = load_tokenizer(local)
    prompt_ids = tok(PROMPT, return_tensors="pt").input_ids[0].tolist()
    prompt_len = len(prompt_ids)

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    llm = LLM(model=local, gpu_memory_utilization=0.35, max_model_len=1536,
              dtype="bfloat16",
              # CUDA graphs : +56% de debit mesure au banc (2026-07-22), mais
              # ils changent l'execution du calcul. Sous forced-seed une
              # difference numerique infime fait basculer un pick inverse-CDF
              # et fait diverger toute la sequence -> ce gate DOIT etre repasse
              # avant de desactiver enforce_eager en production.
              enforce_eager=(os.environ.get("GATE_EAGER", "1") == "1"),
              trust_remote_code=True,
              limit_mm_per_prompt={"image": 0, "video": 0},
              additional_config={"gdn_prefill_backend": "triton"},
              logits_processors=[build_forced_seed_logitsproc_class()])
    gens = []
    for r in range(M):
        sp = SamplingParams(n=1, temperature=0.0, max_tokens=MAX_NEW,
            extra_args={FORCED_SEED_EXTRA_KEY: forced_seed_extra_args(
                randomness=RANDOMNESS, prompt_idx=PROMPT_IDX, checkpoint_hash=CKPT_HASH,
                rollout_index=r, base_offset=0, start_len=prompt_len)})
        out = llm.generate([TokensPrompt(prompt_token_ids=prompt_ids)], sp)
        gens.append(list(out[0].outputs[0].token_ids))
        print(f"[gate] rollout {r}: {len(gens[-1])} tokens", flush=True)
    del llm
    import gc; gc.collect(); torch.cuda.empty_cache()

    model = load_text_generation_model(local, torch_dtype=torch.bfloat16,
        attn_implementation=os.environ.get("GRAIL_ATTN_IMPL", "sdpa")).to("cuda").eval()

    tot_stoch = tot_match = 0
    rollout_rates = []
    for r, gen_ids in enumerate(gens):
        full = torch.tensor([prompt_ids + gen_ids], device="cuda")
        with torch.no_grad():
            logits = model(full).logits[0]
        step = logits[prompt_len - 1: prompt_len - 1 + len(gen_ids)].float()
        us = [u_at(RANDOMNESS, PROMPT_IDX, CKPT_HASH, r, t) for t in range(len(gen_ids))]
        ns, nm = seed_consistency(step, gen_ids, us, t=c.T_PROTO, top_k=c.TOP_K_PROTO,
            top_p=c.TOP_P_PROTO, stochastic_threshold=c.FORCED_SEED_STOCHASTIC_MAXPROB)
        rate = (nm / ns) if ns else float("nan")
        rollout_rates.append((ns, nm, rate))
        tot_stoch += ns; tot_match += nm
        print(f"[gate] rollout {r}: n_stoch={ns} n_match={nm} rate={rate:.4f}", flush=True)

    group_rate = (tot_match / tot_stoch) if tot_stoch else float("nan")
    min_rollout = min((rt for _, _, rt in rollout_rates if rt == rt), default=float("nan"))
    print(f"[gate] GROUP: n_stoch={tot_stoch} n_match={tot_match} rate={group_rate:.4f}", flush=True)
    print(f"[gate] worst rollout rate={min_rollout:.4f}", flush=True)
    group_ok = group_rate >= c.FORCED_SEED_CONSISTENCY_FLOOR   # 0.80
    roll_ok = min_rollout >= c.FORCED_SEED_ROLLOUT_FLOOR       # 0.75
    print(f"[gate] group_floor={c.FORCED_SEED_CONSISTENCY_FLOOR} "
          f"rollout_floor={c.FORCED_SEED_ROLLOUT_FLOOR}", flush=True)
    ok = group_ok and roll_ok
    print(f"[gate] RESULT: {'PASS ✓' if ok else 'FAIL ✗'} "
          f"(group {group_rate:.4f}>={c.FORCED_SEED_CONSISTENCY_FLOOR} & "
          f"worst {min_rollout:.4f}>={c.FORCED_SEED_ROLLOUT_FLOOR})", flush=True)
    print(f"[gate] MACHINE: group={group_rate:.4f} worst={min_rollout:.4f} ok={ok}", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
