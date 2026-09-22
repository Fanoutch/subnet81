"""Test de bout en bout du forced-seed rapide sur le CHEMIN DE PROD (22/09).

Sans validateur : moteur construit comme le mineur (``VLLMBackend``,
environ du launcher), vrai bake de 10 prompts × 16 rollouts via
``generate_forced_phase1_multi_stream`` (le driver en streaming du mineur),
vrais prompts (dataset opencodeinstruct, gabarit v6), arrêt réel à l'EOS,
plafond de tokens de prod. On relève l'heure de livraison de chaque groupe.

  E2E_PHASE=gen   (lancer une fois avec RELIQUARY_FS_FAST=0, une fois avec =1)
                  -> E2E_OUT_<fs|fast>.json : temps par groupe + tokens
  E2E_PHASE=check -> cohérence forced-seed HF (règle du validateur) sur la
                  sortie rapide + taux de tokens identiques rapide/actuel.
S'arrête de lui-même si le validateur revient (le mineur doit avoir la carte).
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROMPTS = [1076031, 1074350, 1075731, 1076391, 1075567,
           1076987, 1078402, 1078321, 1076452, 1074857]   # un vrai 1er bake du 22/09
RND, CKPT_HASH = "c3" * 32, "e2e-bench"
STATE_URL = "http://62.238.81.36:8000/state?env=opencodeinstruct"


def _validator_back() -> bool:
    try:
        return urllib.request.urlopen(STATE_URL, timeout=5).status == 200
    except Exception:
        return False


def gen() -> None:
    from transformers import AutoTokenizer, GenerationConfig
    from reliquary.cli.main import vllm_max_model_len
    from reliquary.environment import load_environment
    from reliquary.miner.bft import phase1_max_new_tokens
    from reliquary.miner.vllm_backend import VLLMBackend
    from reliquary.protocol.tokens import encode_prompt
    from reliquary.shared.modeling import resolve_eos_token_ids
    if _validator_back():
        raise SystemExit("validateur revenu : test annulé, le mineur prend la carte")
    model = os.environ["BENCH_MODEL"]
    variant = "fast" if os.environ.get("RELIQUARY_FS_FAST") == "1" else "fs"
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Base")
    env = load_environment("opencodeinstruct")
    problems = [env.get_problem(i) for i in PROMPTS]
    prompts_tokens = [encode_prompt(tok, p["prompt"]) for p in problems]
    eos = sorted(resolve_eos_token_ids(tokenizer=tok))
    try:
        prim = GenerationConfig.from_pretrained(model).eos_token_id
        prim = prim if isinstance(prim, int) else None
    except Exception:
        prim = None
    max_new = phase1_max_new_tokens(8192, "opencodeinstruct")
    be = VLLMBackend(model_path=model, tokenizer_path="Qwen/Qwen3-4B-Base", gpu_id=0,
                     gpu_memory_utilization=float(os.environ.get("RELIQUARY_VLLM_GPU_FRACTION", "0.45")),
                     max_model_len=vllm_max_model_len(), dtype="bfloat16", forced_seed=True)
    be._ensure_loaded()

    def bake(rnd, prompt_idx_offset=0):
        t0 = time.perf_counter()
        ready, groups = {}, {}

        def on_group(pos, idx, comps):
            ready[pos] = time.perf_counter() - t0
            groups[pos] = [list(c) for c in comps]
        be.generate_forced_phase1_multi_stream(
            prompts_tokens, prompt_indices=[i + prompt_idx_offset for i in PROMPTS],
            randomness=rnd, checkpoint_hash=CKPT_HASH, m_rollouts=16,
            max_tokens=max_new, stop_token_ids=eos, primary_eos_id=prim,
            on_group=on_group, should_abort=lambda: False)
        return ready, groups

    bake("d4" * 32, prompt_idx_offset=7)                   # chauffe (autres tokens)
    label = os.environ.get("E2E_LABEL", variant)
    runs = {}
    for seed in os.environ.get("E2E_SEEDS", RND).split(","):
        if _validator_back():
            raise SystemExit("validateur revenu : test annulé")
        ready, groups = bake(seed)
        times = sorted(ready.values())
        lens = [len(c) for g in groups.values() for c in g]
        print(f"[e2e] {label} tirage {seed[:4]}: groupes prêts à "
              + " ".join(f"{t:.1f}" for t in times)
              + f" s | tokens {sum(lens)} (max {max(lens)}) | "
              f"{sum(lens) / times[-1]:.0f} tok/s", flush=True)
        runs[seed] = {"ready": {str(k): v for k, v in ready.items()},
                      "groups": {str(k): v for k, v in groups.items()}}
    out = os.environ.get("E2E_DIR", "/workspace") + f"/e2e_{label}.json"
    first = next(iter(runs.values()))
    json.dump({"variant": variant, "prompts": PROMPTS, "prompt_tokens": prompts_tokens,
               "ready": first["ready"], "groups": first["groups"], "runs": runs},
              open(out, "w"))


def check() -> int:
    import torch
    from reliquary import constants as c
    from reliquary.environment.forced_sampling import seed_consistency, u_at
    from reliquary.shared.modeling import load_text_generation_model
    from bench_fs_h100 import identity_rates
    d = os.environ.get("E2E_DIR", "/workspace")
    fs, fast = (json.load(open(f"{d}/e2e_{v}.json")) for v in ("fs", "fast"))
    ref = [c_ for p in range(10) for c_ in fs["groups"][str(p)]]
    new = [c_ for p in range(10) for c_ in fast["groups"][str(p)]]
    si, pi = identity_rates(ref, new)
    print(f"[e2e] tokens identiques rapide/actuel : {si:.3f} des séquences, "
          f"{pi:.3f} des positions", flush=True)
    model = load_text_generation_model(os.environ["BENCH_MODEL"], torch_dtype=torch.bfloat16,
                                       attn_implementation=os.environ.get("GRAIL_ATTN_IMPL", "sdpa")
                                       ).to("cuda").eval()
    res = {}
    for name, data in (("fs", fs), ("fast", fast)):
        ts = tm = 0
        worst = 1.0
        for pos in range(10):
            p = data["prompt_tokens"][pos]
            for r in range(0, 16, 4):                     # 4 rollouts / prompt
                g = data["groups"][str(pos)][r]
                if not g:
                    continue
                full = torch.tensor([p + g], device="cuda")
                with torch.no_grad():
                    lg = model(full).logits[0][len(p) - 1: len(p) - 1 + len(g)].float()
                us = [u_at(RND, PROMPTS[pos], CKPT_HASH, r, t) for t in range(len(g))]
                ns, nm = seed_consistency(lg, g, us, t=c.T_PROTO, top_k=c.TOP_K_PROTO,
                                          top_p=c.TOP_P_PROTO,
                                          stochastic_threshold=c.FORCED_SEED_STOCHASTIC_MAXPROB)
                ts += ns; tm += nm
                if ns:
                    worst = min(worst, nm / ns)
        res[name] = (tm / ts if ts else float("nan"), worst)
        print(f"[e2e] cohérence {name} : groupe {res[name][0]:.4f} · pire {worst:.4f}", flush=True)
    ok = (res["fast"][0] >= c.FORCED_SEED_CONSISTENCY_FLOOR
          and res["fast"][1] >= c.FORCED_SEED_ROLLOUT_FLOOR
          and res["fast"][0] >= res["fs"][0] - 0.01)
    print(f"[e2e] RÉSULTAT {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    if os.environ.get("E2E_PHASE", "gen") == "gen":
        gen(); sys.exit(0)
    sys.exit(check())
