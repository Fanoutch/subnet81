import json, gzip, random, collections, sys, numpy as np
sys.path.insert(0, "/workspace/reliquary-miner-priv"); sys.path.insert(0, "/workspace/reliquary-miner-priv/ops/prior_v1")
from reliquary.shared.prompt_range import window_prompt_range
from inzone_scorer import InZoneScorer
from reliquary.environment.opencodeinstruct import OpenCodeInstructEnvironment
env = OpenCodeInstructEnvironment()
S = json.load(open("/workspace/audit_0917/r2_slices.json")); TK = json.load(open("/workspace/audit_0917/r2_taken.json"))
V59 = np.load("/workspace/prompt_scores_v59seul_v1.npz")["score"]
Z = InZoneScorer("/workspace/eval_prior/inzone_v1c_mid.json"); K = InZoneScorer("/workspace/eval_prior/collision_mid.json")
loc = collections.defaultdict(dict)
for l in gzip.open("/workspace/audit_0917/v59era_labels.jsonl.gz", "rt"):
    r = json.loads(l)
    if r["p"] not in loc[r["w"]] or not r["z"]: loc[r["w"]][r["p"]] = bool(r["z"])
rng = random.Random(81)
ALPHAS = [0, 0.5, 1, 2, 4]
agg = collections.defaultdict(collections.Counter)
for w, v in sorted(S.items()):
    wi = int(w)
    if wi <= 46225 or w not in TK: continue
    lo, hi = window_prompt_range(v["randomness"], "opencodeinstruct", 2481806, 5000)
    ours = {p: z for p, z in loc.get(wi, {}).items() if lo <= p < hi}
    early = {int(p) for p, t in TK[w]["taken"].items() if t < 16 and lo <= int(p) < hi}
    anyt = {int(p) for p in TK[w]["taken"] if lo <= int(p) < hi}
    known = set(ours) | anyt
    cand = list(known) + rng.sample([p for p in range(lo, hi) if p not in known], 1500 - len(known))
    txt = {p: (env.get_problem(p) or {}).get("prompt", "") for p in cand}
    pz = np.array([Z.proba(txt[p]) for p in cand]); pk = np.array([K.proba(txt[p]) for p in cand])
    arms = {"v5.9": np.array([V59[p] for p in cand])}
    for a in ALPHAS: arms["V1c a=%g" % a] = pz * (1 - pk) ** a
    for name, s in arms.items():
        top = [cand[i] for i in np.argsort(-s)[:50]]
        c = agg[name]
        c["zone"] += sum(1 for p in top if ours.get(p) is True); c["hors"] += sum(1 for p in top if ours.get(p) is False)
        c["pris16"] += sum(1 for p in top if p in early); c["pris"] += sum(1 for p in top if p in anyt); c["n"] += 50
    agg["_fen"]["n"] += 1
print("fenêtres", agg["_fen"]["n"], "| top-50 sur 1500 par fenêtre")
for name, c in agg.items():
    if name == "_fen": continue
    print("%-10s en zone (nos étiquettes) %.1f%% (%d/%d) | pris par un autre avant 16 s : %.1f%% | pris un jour : %.1f%%" % (
        name, 100 * c["zone"] / max(c["zone"] + c["hors"], 1), c["zone"], c["zone"] + c["hors"], 100 * c["pris16"] / c["n"], 100 * c["pris"] / c["n"]))
