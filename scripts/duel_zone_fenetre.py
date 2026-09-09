import json, gzip, sys, statistics as st
from collections import defaultdict
import numpy as np
sys.path.insert(0, "/root/subnet81/.worktrees/miner-priv-port-v4-dapo")
from reliquary.miner.prompt_predictor import tokenize, risk_short, score_prompt

zone = json.load(open("/root/subnet81/data/risk_zone_v1.json"))
v59  = json.load(gzip.open("/root/subnet81/.worktrees/miner-priv-port-v4-dapo/data/box_backup_2026-08-26_final/models/predictor_v59.json.gz"))

ERA = 37200
byw = defaultdict(dict)
for l in open("/root/subnet81/data_backups/2026-08-31/samples_v4.jsonl"):
    try: r = json.loads(l)
    except: continue
    w = r.get("window_n") or 0
    if w < ERA or w >= 37963 or r.get("env") != "opencodeinstruct" or not r.get("prompt"): continue
    byw[w][r["prompt_idx"]] = dict(txt=r["prompt"], sz=1 if (r.get("sigma") or 0)==0 else 0,
                                   val=float(r.get("score") or 0.0),
                                   vol=sum(r.get("completion_lens") or [0]))
# fenêtres avec assez de candidats pour qu'un échange soit possible
wins = {w: list(d.values()) for w, d in byw.items() if len(d) >= 10}
print("fenêtres exploitables (>=10 candidats):", len(wins))
cache = {}
def sc(r):
    k = id(r)
    if k not in cache:
        cache[k] = (score_prompt(v59, r["txt"]), risk_short(zone, r["txt"]))
    return cache[k]

print("%-22s %8s %8s %10s %10s" % ("bras (choix top-2/fen)", "sz@tete", "val@tete", "vol@tete", "fen. tete saine"))
for name, lam in [("v5.9 seul", 0.0), ("- 0.05*Psz", .05), ("- 0.10*Psz", .10),
                  ("- 0.20*Psz", .20), ("- 0.40*Psz", .40), ("- 0.80*Psz", .80)]:
    szs = []; vals = []; vols = []; intact = 0
    for w, cands in wins.items():
        top = sorted(cands, key=lambda r: -(sc(r)[0] - lam * sc(r)[1]))[:2]
        s = sum(r["sz"] for r in top)
        szs.append(s); vals += [r["val"] for r in top]; vols += [r["vol"] for r in top]
        intact += 1 if s == 0 else 0
    n = len(wins)
    print("%-22s %7.1f%% %9.4f %10.0f %11.0f%%" % (
        name, 100*sum(szs)/(2*n), st.mean(vals), st.mean(vols), 100*intact/n))
# référence : ce que le mineur a RÉELLEMENT mis en tête (ordre de bake réel indisponible
# dans le dump -> approx = les 2 premiers par score v5.9, déjà le bras 1)
