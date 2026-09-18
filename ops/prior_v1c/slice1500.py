import json, gzip, random, collections, sys, numpy as np
sys.path.insert(0, "/workspace/reliquary-miner-priv"); sys.path.insert(0, "/workspace/reliquary-miner-priv/ops/prior_v1")
from reliquary.shared.prompt_range import window_prompt_range
from inzone_scorer import InZoneScorer
from reliquary.environment.opencodeinstruct import OpenCodeInstructEnvironment
env = OpenCodeInstructEnvironment()
S = json.load(open("/workspace/audit_0917/r2_slices.json"))
V59 = np.load("/workspace/prompt_scores_v59seul_v1.npz")["score"]
C = InZoneScorer("/workspace/eval_prior/inzone_v1c_mid.json")
loc = collections.defaultdict(dict)
for l in gzip.open("/workspace/audit_0917/v59era_labels.jsonl.gz", "rt"):
    r = json.loads(l)
    if r["p"] not in loc[r["w"]] or not r["z"]: loc[r["w"]][r["p"]] = bool(r["z"])
rng = random.Random(81)
res = collections.defaultdict(list)       # (source, label) -> [(rang v59, rang V1c, rang hasard)]
allpairs = collections.defaultdict(list)
for w, v in sorted(S.items()):
    w = int(w)
    if w <= 46225: continue
    lo, hi = window_prompt_range(v["randomness"], "opencodeinstruct", 2481806, 5000)
    lab = {}
    for p, z in loc.get(w, {}).items():
        if lo <= p < hi: lab[p] = ("nous", z)
    for p in v["pos"]:
        if lo <= p < hi and p not in lab: lab[p] = ("marche", True)
    for p in v["neg"]:
        if lo <= p < hi and p not in lab: lab[p] = ("marche", False)
    others = [p for p in range(lo, hi) if p not in lab]
    cand = list(lab) + rng.sample(others, max(0, 1500 - len(lab)))
    txt = {p: (env.get_problem(p) or {}).get("prompt", "") for p in cand}
    s59 = np.array([V59[p] for p in cand]); sc = np.array([C.proba(txt[p]) for p in cand]); sr = np.array([rng.random() for _ in cand])
    rk = lambda s: {p: r for r, p in enumerate(np.array(cand)[np.argsort(-s)], 1)}
    R59, RC, RR = rk(s59), rk(sc), rk(sr)
    for p, (src, z) in lab.items():
        res[(src, z)].append((R59[p], RC[p], RR[p]))
    print("fen %d : %d étiquetés (%d nous, %d marché) + %d inconnus" % (w, len(lab), sum(1 for x in lab.values() if x[0] == "nous"), sum(1 for x in lab.values() if x[0] == "marche"), 1500 - len(lab)), flush=True)
def summ(v, i):
    a = np.array([x[i] for x in v]); return "rang méd %4d | dans top 50 : %4.1f%% | top 150 : %4.1f%%" % (np.median(a), 100 * (a <= 50).mean(), 100 * (a <= 150).mean())
print("\n=== rang (sur 1500) des prompts dont le verdict est connu ===")
for key in sorted(res):
    v = res[key]
    print("-- source %-6s %-8s n=%d" % (key[0], "EN ZONE" if key[1] else "HORS", len(v)))
    for i, n in enumerate(("v5.9", "V1c", "hasard")): print("     %-7s %s" % (n, summ(v, i)))
# précision dans le top-50 : parmi les étiquetés qui y tombent, part en zone
for i, n in enumerate(("v5.9", "V1c", "hasard")):
    pos = sum(1 for k, v in res.items() if k[1] for x in v if x[i] <= 50)
    neg = sum(1 for k, v in res.items() if not k[1] for x in v if x[i] <= 50)
    posn = sum(1 for k, v in res.items() if k[0] == "nous" and k[1] for x in v if x[i] <= 50)
    negn = sum(1 for k, v in res.items() if k[0] == "nous" and not k[1] for x in v if x[i] <= 50)
    print("%-7s top-50 : %d en zone / %d hors (toutes sources) ; sur NOS étiquettes : %d / %d (%.1f%% en zone)" % (n, pos, neg, posn, negn, 100 * posn / max(posn + negn, 1)))
