"""Runtime validation of the forced-seed v2 port on the REAL 2B checkpoint.

Loads the live checkpoint via the miner's own canonical HF loader
(`load_text_generation_model` — the same class the validator's GRAIL proof
uses), generates a completion under the ported ForcedSeedLogitsProcessor (v2
u_at, no hotkey), then teacher-forces `seed_consistency` over the produced
logits. An honest v2 miner must score ~1.0 (every stochastic position matches
the public inverse-CDF pick). This is deploy-checklist step 6, run BEFORE any
hotkey re-registration so a broken forced stream never costs a registration.

Run on the GPU box:  PYTHONPATH=/workspace/reliquary-miner-priv \
    python validate_forced_seed_v2.py
"""
import os
import torch

CKPT = os.environ.get("SMOKE_CKPT", "ReliquaryForge/qwen3.5-2b-reliquary-v2")
REV = os.environ.get("SMOKE_REV", "cdc9daee91a8f00b649202fd4c45bd90a1b3f3d6")

from huggingface_hub import snapshot_download
from reliquary.shared.modeling import load_text_generation_model, load_tokenizer
from reliquary.environment.forced_sampling import u_at, warp, pick, seed_consistency
from reliquary.miner.forced_seed_sampler import ForcedSeedLogitsProcessor
from reliquary import constants as c

T, TOP_P, TOP_K = 0.6, 0.95, 20
RANDOMNESS = "deadbeef" * 8      # stand-in window randomness
PROMPT_IDX = 12345
CKPT_HASH = "validation-ckpt-hash"
ROLLOUT = 0
# 96 was too short to reach FORCED_SEED_MIN_STOCH_POSITIONS (30): the run then
# "fails" on sample size rather than on consistency. Override for a real verdict.
MAX_NEW = int(os.environ.get("VAL_MAX_NEW", "96"))


def main():
    print(f"[val] domain={c.FORCED_SEED_DOMAIN} proto={c.FORCED_SEED_PROTOCOL_VERSION}", flush=True)
    assert c.FORCED_SEED_DOMAIN == "reliquary-forced-seed-v2"
    assert c.FORCED_SEED_PROTOCOL_VERSION == 2

    local = snapshot_download(CKPT, revision=REV)
    print("[val] checkpoint at", local, flush=True)
    tok = load_tokenizer(local)
    model = load_text_generation_model(
        local, torch_dtype=torch.bfloat16,
        attn_implementation=os.environ.get("GRAIL_ATTN_IMPL", "flash_attention_2"),
    ).to("cuda").eval()
    print("[val] model loaded", flush=True)

    prompt = "Compute 17 * 23 step by step. Answer:"
    ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
    prompt_len = ids.shape[1]

    proc = ForcedSeedLogitsProcessor(
        randomness=RANDOMNESS, hotkey="unused-under-v2", prompt_idx=PROMPT_IDX,
        checkpoint_hash=CKPT_HASH, rollout_indices=[ROLLOUT], base_offsets=[0],
        start_len=prompt_len,
    )
    from transformers import LogitsProcessorList
    import time
    _t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=MAX_NEW, do_sample=False,
            logits_processor=LogitsProcessorList([proc]),
            repetition_penalty=1.0, pad_token_id=tok.pad_token_id,
        )
    _elapsed = time.perf_counter() - _t0
    gen_ids = out[0, prompt_len:].tolist()
    # Timing matters now: upstream 8835a95 cut WINDOW_COLLECTION_SECONDS 300 -> 100,
    # so a full BFT group (8 rollouts x up to 2048+512 tokens) must fit in 100s.
    print(
        f"[val] generated {len(gen_ids)} tokens in {_elapsed:.1f}s "
        f"({len(gen_ids)/_elapsed:.1f} tok/s, 1 rollout sequential)",
        flush=True,
    )
    print("[val] text:", repr(tok.decode(gen_ids)[:200]), flush=True)

    # Teacher-force: recompute logits over the produced sequence, score consistency.
    with torch.no_grad():
        full = out[:, : prompt_len + len(gen_ids)]
        logits = model(full).logits[0]  # [seq, vocab]
    # position t predicts token at prompt_len + t, from logits at prompt_len + t - 1
    step_logits = logits[prompt_len - 1 : prompt_len - 1 + len(gen_ids)].float()
    us = [u_at(RANDOMNESS, PROMPT_IDX, CKPT_HASH, ROLLOUT, t) for t in range(len(gen_ids))]
    n_stoch, n_match = seed_consistency(
        step_logits, gen_ids, us, t=T, top_k=TOP_K, top_p=TOP_P,
        stochastic_threshold=c.FORCED_SEED_STOCHASTIC_MAXPROB,
    )
    rate = (n_match / n_stoch) if n_stoch else float("nan")
    print(f"[val] seed_consistency: n_stoch={n_stoch} n_match={n_match} rate={rate:.4f}", flush=True)
    print(f"[val] group floor={c.FORCED_SEED_CONSISTENCY_FLOOR} min_stoch={c.FORCED_SEED_MIN_STOCH_POSITIONS}", flush=True)
    ok = n_stoch >= c.FORCED_SEED_MIN_STOCH_POSITIONS and rate >= c.FORCED_SEED_CONSISTENCY_FLOOR
    print("[val] RESULT:", "PASS ✓ forced-seed v2 is runtime-correct" if ok else "FAIL ✗", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
