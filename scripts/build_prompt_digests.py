import hashlib, sys, os, time, numpy as np
from reliquary.environment.opencodeinstruct import OpenCodeInstructEnvironment
env=OpenCodeInstructEnvironment(); n=len(env); out=sys.argv[1]
print("n prompts", n, flush=True)
arr=np.zeros((n,32),dtype=np.uint8); t0=time.time()
for idx in range(n):
    p=(env.get_problem(idx) or {}).get("prompt","")
    h=hashlib.sha256(); h.update(b"reliquary/prompt-content/v1\0"); h.update(b"opencodeinstruct"); h.update(b"\0"); h.update(p.encode("utf-8"))
    arr[idx]=np.frombuffer(h.digest(),dtype=np.uint8)
    if idx%200000==0: print(idx, f"{time.time()-t0:.0f}s", flush=True)
np.save(out+".tmp.npy", arr); os.replace(out+".tmp.npy", out)
print("done", out, f"{time.time()-t0:.0f}s", flush=True)
