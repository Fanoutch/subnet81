"""Gate GPU du chantier fs-graph (2026-08-15) — À EXÉCUTER SUR LA BOX.

1. ÉQUIVALENCE BIT-EXACTE : pour plusieurs formes (n∈{1,2,8,16,128}) et
   plusieurs replays par forme (buffers statiques réutilisés — le piège),
   le bloc masqué produit par le CUDA graph doit être identique bit à bit
   au chemin eager (torch.equal, aucune tolérance).
2. MICRO-BANC : coût par appel de la passe eager vs graph (sync inclus),
   à n=8 (régime sprint) et n=128 (balayage).
"""
import os, sys, time
sys.path.insert(0, "/workspace/reliquary-miner-priv")
os.environ["RELIQUARY_FS_GRAPH"] = "1"

import torch
from reliquary.miner import vllm_forced_seed as vfs
from reliquary.environment.forced_sampling import force_rows_batched

assert torch.cuda.is_available(), "gate GPU uniquement"
dev = torch.device("cuda")
VOCAB = 151_936
torch.manual_seed(1234)

gp = vfs._FsGraphPass()
fails = 0
for n in (1, 2, 8, 16, 32):  # 128 > paliers -> eager par design (v2)
    for rep in range(5):
        # logits réalistes : bf16 upcastés (comme la prod), pics marqués
        lg = (torch.randn(n, VOCAB, device=dev, dtype=torch.bfloat16)
              .float() * 3.0)
        u = torch.rand(n, device=dev, dtype=torch.float32)
        # eager de référence
        toks = force_rows_batched(lg, u, t=vfs.T_PROTO,
                                  top_k=vfs.TOP_K_PROTO, top_p=vfs.TOP_P_PROTO)
        ref = torch.full_like(lg, float("-inf"))
        ref.scatter_(1, toks.unsqueeze(1), 0.0)
        # graph
        out = gp.run(lg, u)
        if out is None:
            print(f"[gate] n={n} rep={rep}: CAPTURE MORTE (repli eager)")
            fails += 1
            break
        if not torch.equal(out, ref):
            diff = (out != ref).sum().item()
            print(f"[gate] n={n} rep={rep}: MISMATCH {diff} éléments")
            fails += 1
if fails == 0:
    print("[gate] ÉQUIVALENCE: PASS ✓ (5 formes × 5 replays, bit-exact)")
else:
    print(f"[gate] ÉQUIVALENCE: FAIL ({fails})")
    sys.exit(1)

def bench(n, mode, iters=300):
    lg = torch.randn(n, VOCAB, device=dev).float() * 3.0
    u = torch.rand(n, device=dev)
    def eager():
        toks = force_rows_batched(lg, u, t=vfs.T_PROTO,
                                  top_k=vfs.TOP_K_PROTO, top_p=vfs.TOP_P_PROTO)
        out = torch.full_like(lg, float("-inf"))
        out.scatter_(1, toks.unsqueeze(1), 0.0)
        return out
    fn = eager if mode == "eager" else (lambda: gp.run(lg, u))
    for _ in range(20): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3

for n in (8, 32):
    e = bench(n, "eager"); g = bench(n, "graph")
    print(f"[bench] n={n}: eager {e:.3f} ms/appel | graph {g:.3f} ms/appel "
          f"| gain {100*(1-g/e):.0f}%")
print("[gate] DONE")


# --- v2 (2026-08-15) : gate anti-fuite — n VARIABLE comme en production ---
print("[gate] v2: n variable (paliers + padding)")
gp2 = vfs._FsGraphPass()
fails2 = 0
for n in (1, 2, 3, 5, 7, 8, 11, 15, 16, 19, 24, 29, 32, 31, 6, 2):
    lg = (torch.randn(n, VOCAB, device=dev, dtype=torch.bfloat16).float() * 3.0)
    u = torch.rand(n, device=dev, dtype=torch.float32)
    toks = force_rows_batched(lg, u, t=vfs.T_PROTO,
                              top_k=vfs.TOP_K_PROTO, top_p=vfs.TOP_P_PROTO)
    ref = torch.full_like(lg, float("-inf"))
    ref.scatter_(1, toks.unsqueeze(1), 0.0)
    out = gp2.run(lg, u)
    if out is None or not torch.equal(out, ref):
        print(f"[gate] v2 n={n}: {'MORT' if out is None else 'MISMATCH'}")
        fails2 += 1
ncaps = len(gp2._graphs)
big = gp2.run(torch.randn(60, VOCAB, device=dev).float(), torch.rand(60, device=dev))
print(f"[gate] v2: captures={ncaps} (attendu <= {len(vfs._FsGraphPass.BUCKETS)}), n=60 -> {'eager (None)' if big is None else 'ERREUR: capturé'}")
if fails2 == 0 and ncaps <= len(vfs._FsGraphPass.BUCKETS) and big is None:
    print("[gate] V2 RESULT: PASS ✓ (16 n variables bit-exact, captures bornées, >32 eager)")
else:
    print("[gate] V2 RESULT: FAIL"); sys.exit(1)
