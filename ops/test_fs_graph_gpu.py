"""Gate GPU du chantier fs-graph (2026-08-15) — À EXÉCUTER SUR LA BOX.

0. PALIERS : le jeu testé est celui de `RELIQUARY_FS_GRAPH_BUCKETS` s'il est
   posé, sinon le défaut du code — les formes testées et la borne de captures
   en sont DÉRIVÉES (08/09), pour que la gate valide aussi un palier ajouté
   (ex. 48, requis par un sprint de 3 groupes = 48 séquences).
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
BK = list(gp.buckets)
BMAX = BK[-1]
print(f"[gate] paliers testés : {BK}")
fails = 0
# formes = chaque palier + un cran sous le plus grand (padding)
_FORMES = tuple(sorted({1, 2, 8, 16, BMAX, max(1, BMAX - 1)}))
for n in _FORMES:
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

for n in sorted({8, 32, BMAX}):
    e = bench(n, "eager"); g = bench(n, "graph")
    print(f"[bench] n={n}: eager {e:.3f} ms/appel | graph {g:.3f} ms/appel "
          f"| gain {100*(1-g/e):.0f}%")
print("[gate] DONE")


# --- v2 (2026-08-15) : gate anti-fuite — n VARIABLE comme en production ---
print("[gate] v2: n variable (paliers + padding)")
gp2 = vfs._FsGraphPass()
fails2 = 0
# n variables : autour de chaque palier (b-1, b) + quelques valeurs libres,
# puis retour en arrière (réutilisation des buffers statiques = le piège)
_VAR = [1, 2, 3, 5, 7, 8, 11, 15, 16, 19, 24, 29]
for _b in BK:
    _VAR += [max(1, _b - 1), _b]
_VAR += [BMAX - 1, 6, 2]
for n in _VAR:
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
_NBIG = BMAX + 8
big = gp2.run(torch.randn(_NBIG, VOCAB, device=dev).float(), torch.rand(_NBIG, device=dev))
_free, _tot = torch.cuda.mem_get_info()
print(f"[gate] v2: captures={ncaps} (attendu <= {len(BK)}), n={_NBIG} -> "
      f"{'eager (None)' if big is None else 'ERREUR: capturé'} | "
      f"VRAM libre {_free/2**30:.1f} Go / {_tot/2**30:.1f} Go")
if fails2 == 0 and ncaps <= len(BK) and big is None:
    print(f"[gate] V2 RESULT: PASS ✓ ({len(_VAR)} n variables bit-exact, "
          f"captures bornées, >{BMAX} eager)")
else:
    print("[gate] V2 RESULT: FAIL"); sys.exit(1)
