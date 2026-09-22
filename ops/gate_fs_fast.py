"""Gate de conformité du chemin forced-seed rapide (22/09).

Juge = la règle du validateur : ``seed_consistency`` (logits HF recalculés par
teacher-forcing sur NOS tokens) avec les planchers 0,80 groupe / 0,75 pire
rollout. On génère avec le moteur de PROD (``VLLMBackend``, processeur batché)
en mode actuel (``fs``) et rapide (``fast``), puis on vérifie les deux.
Deux phases (la VRAM de vLLM doit être rendue avant de charger HF) :
  GATE_PHASE=gen   -> GATE_TOKENS (json)
  GATE_PHASE=check -> verdict ; code retour 0 si fast PASS et ≥ fs − 0,01.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RND, CKPT_HASH = "5a" * 32, "bench"


def gen(path: str) -> None:
    from transformers import AutoTokenizer
    from bench_fs_h100 import _prompts, _run
    from reliquary.cli.main import vllm_max_model_len
    from reliquary.miner.vllm_backend import VLLMBackend
    model = os.environ["BENCH_MODEL"]
    n, length = int(os.environ.get("GATE_PROMPTS", "10")), int(os.environ.get("GATE_LEN", "512"))
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Base")
    be = VLLMBackend(model_path=model, tokenizer_path="Qwen/Qwen3-4B-Base", gpu_id=0,
                     gpu_memory_utilization=float(os.environ.get("RELIQUARY_VLLM_GPU_FRACTION", "0.45")),
                     max_model_len=vllm_max_model_len(), dtype="bfloat16", forced_seed=True)
    be._ensure_loaded()
    prompts = _prompts(tok, n)
    out = {"prompts": prompts, "modes": {}}
    for mode in ("fs", "fast"):
        _, toks = _run(be._llm, prompts, m=16, length=length, mode=mode, rnd=RND)
        out["modes"][mode] = toks
        print(f"[gate] {mode}: {len(toks)} séquences générées", flush=True)
    with open(path, "w") as fh:
        json.dump(out, fh)


def check(path: str) -> int:
    import torch
    from reliquary import constants as c
    from reliquary.environment.forced_sampling import seed_consistency, u_at
    from reliquary.shared.modeling import load_text_generation_model
    data = json.load(open(path))
    model = load_text_generation_model(os.environ["BENCH_MODEL"], torch_dtype=torch.bfloat16,
                                       attn_implementation=os.environ.get("GRAIL_ATTN_IMPL", "sdpa")
                                       ).to("cuda").eval()
    step_every = int(os.environ.get("GATE_SAMPLE_EVERY", "5"))   # 1 séquence sur 5
    res = {}
    for mode, seqs in data["modes"].items():
        tot_s = tot_m = 0
        worst = 1.0
        for i in range(0, len(seqs), step_every):
            pi, r = divmod(i, 16)
            p, g = data["prompts"][pi], seqs[i]
            full = torch.tensor([p + g], device="cuda")
            with torch.no_grad():
                lg = model(full).logits[0][len(p) - 1: len(p) - 1 + len(g)].float()
            us = [u_at(RND, 10_000 + pi, CKPT_HASH, r, t) for t in range(len(g))]
            ns, nm = seed_consistency(lg, g, us, t=c.T_PROTO, top_k=c.TOP_K_PROTO,
                                      top_p=c.TOP_P_PROTO,
                                      stochastic_threshold=c.FORCED_SEED_STOCHASTIC_MAXPROB)
            tot_s += ns; tot_m += nm
            if ns:
                worst = min(worst, nm / ns)
        res[mode] = (tot_m / tot_s if tot_s else float("nan"), worst, tot_s)
        print(f"[gate] {mode}: groupe {res[mode][0]:.4f} · pire séquence {worst:.4f} · "
              f"{tot_s} positions stochastiques", flush=True)
    g_ok = res["fast"][0] >= c.FORCED_SEED_CONSISTENCY_FLOOR and res["fast"][1] >= c.FORCED_SEED_ROLLOUT_FLOOR
    same = res["fast"][0] >= res["fs"][0] - 0.01
    ok = g_ok and same
    print(f"[gate] planchers {c.FORCED_SEED_CONSISTENCY_FLOOR}/{c.FORCED_SEED_ROLLOUT_FLOOR} -> "
          f"RÉSULTAT {'PASS' if ok else 'FAIL'} (fast≥planchers {g_ok}, fast≥fs−0,01 {same})", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    path = os.environ.get("GATE_TOKENS", "/workspace/gate_fs_fast.json")
    if os.environ.get("GATE_PHASE", "gen") == "gen":
        gen(path); sys.exit(0)
    sys.exit(check(path))
