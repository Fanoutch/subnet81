import json, hashlib, statistics as st
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ERA = 37200; SPLIT = 37800
rows = []
for l in open("/root/subnet81/data_backups/2026-08-31/samples_v4.jsonl"):
    try: r = json.loads(l)
    except: continue
    w = r.get("window_n") or 0
    if w < ERA or r.get("env") != "opencodeinstruct": continue
    if not r.get("prompt"): continue
    rows.append((w, r["prompt_idx"], r["prompt"], float(r.get("sigma") or 0),
                 r.get("k"), r.get("checkpoint_n")))
print("groupes ère fraîche (code):", len(rows))
oz = sum(1 for r in rows if r[3] == 0)
print("taux sigma=0 global: %.1f%%" % (100*oz/len(rows)))
k16 = sum(1 for r in rows if r[3]==0 and r[4]==16)
k0  = sum(1 for r in rows if r[3]==0 and (r[4] or 0)==0)
print("  dont k=16 (trop facile): %d   k=0 (trop dur): %d" % (k16, k0))

# ---------- 1. HISTORIQUE PAR PROMPT (répétabilité de sigma=0) ----------
byp = defaultdict(list)
for w,p,txt,s,k,c in rows: byp[p].append((w,s,k))
trans = {"apres_sz":[0,0], "apres_inzone":[0,0]}  # [n, n_sz_next]
trans_k16 = [0,0]; trans_k0 = [0,0]
for p, occ in byp.items():
    occ.sort()
    for i in range(1, len(occ)):
        prev_s, prev_k = occ[i-1][1], occ[i-1][2]
        nxt_sz = 1 if occ[i][1] == 0 else 0
        key = "apres_sz" if prev_s == 0 else "apres_inzone"
        trans[key][0] += 1; trans[key][1] += nxt_sz
        if prev_s == 0 and prev_k == 16: trans_k16[0]+=1; trans_k16[1]+=nxt_sz
        if prev_s == 0 and (prev_k or 0) == 0: trans_k0[0]+=1; trans_k0[1]+=nxt_sz
multi = sum(1 for o in byp.values() if len(o) >= 2)
print("\n--- 1. HISTORIQUE (prompts revus: %d/%d) ---" % (multi, len(byp)))
for k_, (n, nsz) in trans.items():
    if n: print("  P(sigma=0 | %-12s) = %.1f%%  (n=%d)" % (k_, 100*nsz/n, n))
if trans_k16[0]: print("  P(sigma=0 | avant k=16) = %.1f%% (n=%d)" % (100*trans_k16[1]/trans_k16[0], trans_k16[0]))
if trans_k0[0]:  print("  P(sigma=0 | avant k=0 ) = %.1f%% (n=%d)" % (100*trans_k0[1]/trans_k0[0], trans_k0[0]))

# ---------- 2. TF-IDF texte, coupe temporelle + prompts JAMAIS vus ----------
train = [(t,s,k) for w,p,t,s,k,c in rows if w < SPLIT]
train_p = {p for w,p,t,s,k,c in rows if w < SPLIT}
test  = [(t,s,k) for w,p,t,s,k,c in rows if w >= SPLIT and p not in train_p]
test_seen = [(t,s) for w,p,t,s,k,c in rows if w >= SPLIT and p in train_p]
print("\n--- 2. TEXTE (train %d < fen %d <= test %d jamais-vus, %d revus) ---"
      % (len(train), SPLIT, len(test), len(test_seen)))
vec = TfidfVectorizer(max_features=30000, ngram_range=(1,1), sublinear_tf=True)
Xtr = vec.fit_transform([t for t,s,k in train]); ytr = np.array([1 if s==0 else 0 for t,s,k in train])
Xte = vec.transform([t for t,s,k in test]);      yte = np.array([1 if s==0 else 0 for t,s,k in test])
clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
pr = clf.predict_proba(Xte)[:,1]
print("  AUC(sigma=0) prompts jamais vus : %.3f" % roc_auc_score(yte, pr))
# usage filtre : si on veto le pire quintile predit, que perd/gagne-t-on ?
q = np.quantile(pr, 0.8)
veto = pr >= q
print("  veto top-20%% risque : évite %.0f%% des sigma=0, sacrifie %.0f%% des in-zone"
      % (100*yte[veto].sum()/max(yte.sum(),1),
         100*(1-yte[veto]).sum()/max((1-yte).sum(),1)))
# cibles séparées
for lab, cond in (("k=16", lambda s,k: 1 if (s==0 and k==16) else 0),
                  ("k=0",  lambda s,k: 1 if (s==0 and (k or 0)==0) else 0)):
    ytr2 = np.array([cond(s,k) for t,s,k in train]); yte2 = np.array([cond(s,k) for t,s,k in test])
    if yte2.sum() >= 20 and ytr2.sum() >= 50:
        c2 = LogisticRegression(max_iter=2000).fit(Xtr, ytr2)
        print("  AUC(%s seul) jamais vus : %.3f  (n test %d)" % (lab, roc_auc_score(yte2, c2.predict_proba(Xte)[:,1]), yte2.sum()))

# ---------- 3. LE PROCÈS DE mu=10 : longueur réelle vs sigma=0 ----------
print("\n--- 3. VOLUME RÉEL vs sigma=0 (le procès de mu=10) ---")
vols = [(sum(json.loads(l)["completion_lens"] or [0]), 0) for l in []]  # placeholder
by_vol = defaultdict(lambda: [0,0])
for l in open("/root/subnet81/data_backups/2026-08-31/samples_v4.jsonl"):
    try: r = json.loads(l)
    except: continue
    if (r.get("window_n") or 0) < ERA or r.get("env") != "opencodeinstruct": continue
    cl = r.get("completion_lens")
    if not cl: continue
    v = sum(cl)
    b = "<6k" if v<6000 else "6-10k" if v<10000 else "10-14k" if v<14000 else ">=14k"
    by_vol[b][0]+=1
    if (r.get("sigma") or 0)==0: by_vol[b][1]+=1
for b in ("<6k","6-10k","10-14k",">=14k"):
    n,z = by_vol[b]
    if n: print("  volume %-6s : sigma=0 %.1f%%  (n=%d)" % (b, 100*z/n, n))
