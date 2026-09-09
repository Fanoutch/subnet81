import json, gzip, hashlib, sys, statistics as st
from collections import defaultdict
import numpy as np
sys.path.insert(0, "/root/subnet81/.worktrees/miner-priv-port-v4-dapo")
from reliquary.miner.prompt_predictor import tokenize, risk_short, score_prompt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ERA = 37200
# ---- corpus (dédupliqué par (fenêtre, prompt)) ----
rows, seen = [], set()
for l in open("/root/subnet81/data_backups/2026-08-31/samples_v4.jsonl"):
    try: r = json.loads(l)
    except: continue
    w = r.get("window_n") or 0
    if w < ERA or r.get("env") != "opencodeinstruct" or not r.get("prompt"): continue
    key = (w, r["prompt_idx"])
    if key in seen: continue
    seen.add(key)
    rows.append(dict(w=w, p=r["prompt_idx"], txt=r["prompt"],
                     sz=1 if (r.get("sigma") or 0) == 0 else 0,
                     val=float(r.get("score") or 0.0),
                     vol=sum(r.get("completion_lens") or [0]), k=r.get("k")))
print("corpus:", len(rows), "| sigma=0: %.1f%%" % (100*sum(r['sz'] for r in rows)/len(rows)))

is_test = lambda p: int(hashlib.md5(str(p).encode()).hexdigest(), 16) % 5 == 0
train = [r for r in rows if not is_test(r["p"])]
test  = [r for r in rows if is_test(r["p"])]
print("train %d prompts-groupes / test %d (hash de prompt, 0 recouvrement)" % (len(train), len(test)))

# ---- entraînement (BoW binaire, tokenizer DU MINEUR) ----
vec = CountVectorizer(binary=True, tokenizer=lambda t: list(set(tokenize(t))),
                      token_pattern=None, min_df=5)
Xtr = vec.fit_transform([r["txt"] for r in train]); ytr = np.array([r["sz"] for r in train])
Xte = vec.transform([r["txt"] for r in test]);      yte = np.array([r["sz"] for r in test])
clf = LogisticRegression(max_iter=3000, C=1.0).fit(Xtr, ytr)
auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
print("AUC sklearn (prompts jamais vus): %.3f" % auc)

# ---- export format risk_short + contrôle de parité via la fonction du mineur ----
vocab = vec.get_feature_names_out()
model = {"bias": float(clf.intercept_[0]),
         "w": {vocab[i]: float(c) for i, c in enumerate(clf.coef_[0]) if abs(c) > 1e-6},
         "meta": {"target": "P(sigma=0)", "era": ">=37200", "auc_test": round(auc, 3),
                  "n_train": len(train), "date": "2026-08-31", "version": "zone_v1",
                  "note": "malus de zone — deploiement via RELIQUARY_SHORT_RISK_MODEL"}}
pr_miner = np.array([risk_short(model, r["txt"]) for r in test])
print("AUC via risk_short() du mineur : %.3f  (parité)" % roc_auc_score(yte, pr_miner))
json.dump(model, open("/root/subnet81/data/risk_zone_v1.json", "w"))
print("modèle -> data/risk_zone_v1.json (%d poids)" % len(model["w"]))

# ---- duel de sélection : v5.9 seul vs v5.9 - lambda*P_sz, viviers appariés ----
v59 = json.load(gzip.open("/root/subnet81/.worktrees/miner-priv-port-v4-dapo/data/box_backup_2026-08-26_final/models/predictor_v59.json.gz"))
for r in test:
    r["s59"] = score_prompt(v59, r["txt"])
    r["psz"] = risk_short(model, r["txt"])
test.sort(key=lambda r: r["w"])
POOL = 40
pools = [test[i:i+POOL] for i in range(0, len(test) - POOL + 1, POOL)]
print("\n=== DUEL (%d viviers de %d, appariés) ===" % (len(pools), POOL))
print("%-22s %8s %8s %10s %10s %9s" % ("bras", "sz@top2", "sz@top8", "valeur@8", "vol@8", "val-tête"))
for name, key in [("v5.9 seul", lambda r: -r["s59"])] + [
        ("v5.9 - %.2f*Psz" % lam, (lambda lam: lambda r: -(r["s59"] - lam * r["psz"]))(lam))
        for lam in (0.05, 0.10, 0.20, 0.40, 0.80)]:
    sz2 = sz8 = 0; vals = []; vols = []; vhead = []
    for pool in pools:
        top = sorted(pool, key=key)[:8]
        sz2 += sum(r["sz"] for r in top[:2]); sz8 += sum(r["sz"] for r in top)
        vals += [r["val"] for r in top]; vols += [r["vol"] for r in top]
        vhead += [r["val"] for r in top[:2]]
    n = len(pools)
    print("%-22s %7.1f%% %7.1f%% %10.4f %10.0f %9.4f" % (
        name, 100*sz2/(2*n), 100*sz8/(8*n), st.mean(vals), st.mean(vols), st.mean(vhead)))
