"""Table « V1c × (1 − P(collision))^0,5 » (18/09). Garde risk/volume/empreinte de --old."""
import argparse, sys, time, numpy as np
sys.path.insert(0, "/workspace/reliquary-miner-priv/ops/prior_v1")
from inzone_scorer import InZoneScorer
from reliquary.environment.opencodeinstruct import OpenCodeInstructEnvironment
ap = argparse.ArgumentParser(); ap.add_argument("--zone"); ap.add_argument("--coll"); ap.add_argument("--alpha", type=float, default=0.5)
ap.add_argument("--old"); ap.add_argument("--out"); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
old = np.load(a.old); n = len(old["score"]) if not a.limit else a.limit
Z, K = InZoneScorer(a.zone), InZoneScorer(a.coll); env = OpenCodeInstructEnvironment()
score = np.zeros(len(old["score"]), dtype="float32"); t0 = time.time(); miss = 0
for i in range(n):
    try:
        t = (env.get_problem(i) or {}).get("prompt", "")
        score[i] = Z.proba(t) * (1.0 - K.proba(t)) ** a.alpha
    except Exception:
        miss += 1
    if i % 200000 == 0: print("  %d/%d %.1f min" % (i, n, (time.time() - t0) / 60), flush=True)
if a.limit:
    print("essai %d prompts %.1f s, manquants %d, moyenne %.3f" % (n, time.time() - t0, miss, score[:n].mean())); sys.exit()
np.savez(a.out, score=score, risk=old["risk"], volume=old["volume"], fingerprint=old["fingerprint"])
print("écrit %s | manquants %d | moyenne %.3f | empreinte %s" % (a.out, miss, score.mean(), str(old["fingerprint"])[:12]))
