import json, gzip, random, sys
sys.path.insert(0, "/workspace/reliquary-miner-priv")
from reliquary.shared.prompt_range import window_prompt_range
from reliquary.environment.opencodeinstruct import OpenCodeInstructEnvironment
env = OpenCodeInstructEnvironment()
S = json.load(open("/workspace/audit_0917/r2_taken.json"))
rng = random.Random(5)
out = gzip.open("/workspace/audit_0917/taken_ds.jsonl.gz", "wt"); n = 0
for w, v in sorted(S.items()):
    lo, hi = window_prompt_range(v["randomness"], "opencodeinstruct", 2481806, 5000)
    taken = {int(p): t for p, t in v["taken"].items() if lo <= int(p) < hi}
    rest = [p for p in range(lo, hi) if p not in taken]
    for p in list(taken) + rng.sample(rest, 400):
        t = (env.get_problem(p) or {}).get("prompt", "")
        out.write(json.dumps({"w": int(w), "p": p, "taken": p in taken, "t_arr": taken.get(p), "t": t}) + "\n"); n += 1
out.close(); print("lignes", n)
