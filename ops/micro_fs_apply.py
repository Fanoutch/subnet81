"""Micro-banc du processeur forced-seed SEUL sur GPU (22/09), sans vLLM.

La décomposition du 22/09 dit : mécanisme vLLM ~0 ms, notre calcul ~4,2 ms
par pas à 160 séquences. Ici on chronomètre ``ForcedRowsState.apply`` et
chacune de ses étapes sur des logits [n, 151936] fp32 résidents, avec des
événements CUDA (temps GPU) et le temps CPU d'appel (lancement), pour savoir
si le coût est du calcul GPU, du CPU, ou une synchronisation cachée.
"""
from __future__ import annotations

import os
import sys
import time

import torch


def _cuda_ms(fn, reps=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t_cpu = time.perf_counter()
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    cpu_ms = 1000 * (time.perf_counter() - t_cpu) / reps
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps, cpu_ms


def main() -> int:
    from reliquary.miner import vllm_forced_seed as vfs
    from reliquary.environment.forced_sampling import force_rows_batched
    vfs.T_PROTO, vfs.TOP_K_PROTO, vfs.TOP_P_PROTO = 1.0, 0, 1.0
    dev = torch.device("cuda")
    V = 151936
    for n in [int(x) for x in os.environ.get("MICRO_N", "32,160,256").split(",")]:
        logits = torch.randn(n, V, device=dev, dtype=torch.float32) * 3
        outs = [[0] * 100 for _ in range(n)]
        fsd = lambda i, fast: {"randomness": "ab" * 32, "prompt_idx": 1000 + i // 16,
                               "checkpoint_hash": "c", "rollout_index": i % 16,
                               "base_offset": 0, "start_len": 0,
                               **({"fast": True} if fast else {})}
        res = {}
        for fast in (False, True):
            st = vfs.ForcedRowsState()
            st.rebuild({i: (fsd(i, fast), outs[i]) for i in range(n)}, device=dev)
            res["apply_fast" if fast else "apply_actuel"] = _cuda_ms(
                lambda: st.apply(logits.clone()))
        u = torch.rand(n, device=dev)
        lg = logits.clone()
        res["clone_seul"] = _cuda_ms(lambda: logits.clone())
        res["softmax"] = _cuda_ms(lambda: torch.softmax(lg, dim=-1))
        p = torch.softmax(lg, dim=-1)
        res["somme+div"] = _cuda_ms(lambda: p / p.sum(dim=-1, keepdim=True))
        res["cumsum"] = _cuda_ms(lambda: torch.cumsum(p, dim=-1))
        c = torch.cumsum(p, dim=-1)
        res["searchsorted"] = _cuda_ms(lambda: torch.searchsorted(c, u.unsqueeze(-1), right=True))
        res["pick_fast(total)"] = _cuda_ms(lambda: vfs.fast_pick_t1(lg, u))
        res["pick_ref(total)"] = _cuda_ms(lambda: force_rows_batched(lg, u, t=1.0, top_k=0, top_p=1.0))
        print(f"[micro] n={n}", flush=True)
        for k, (g, cpu) in res.items():
            print(f"[micro]   {k:18s} GPU {g:7.3f} ms   CPU(appel) {cpu:7.3f} ms", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
