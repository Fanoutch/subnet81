import gzip, json, sys, numpy as np, collections
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack, csr_matrix
sys.path.insert(0, "data/prior_v1"); sys.path.insert(0, ".worktrees/miner-priv-fifo")
from inzone_scorer import InZoneScorer, PRE
from reliquary.miner import prompt_predictor as pp
txt = lambda t: t[len(PRE):] if t.startswith(PRE) else t
E = [json.loads(l) for l in gzip.open("data/prior_v1c/v59era_labels.jsonl.gz", "rt")]
# un label par (fenêtre, prompt) ; ooz prioritaire s'il existe (groupe jeté)
lab = {}
for r in E:
    k = (r["w"], r["p"])
    if k not in lab or not r["z"]: lab[k] = r
E = list(lab.values())
old = [dict(p=r["p"], t=r["t"], z=bool(r["z"])) for f in ("v1_code_samples", "v1_code_negatives") for r in (json.loads(l) for l in gzip.open(f"data/prior_v1/{f}.jsonl.gz", "rt"))]
v1b = [json.loads(l) for l in gzip.open("data/prior_v1b/v1_labels_0916.jsonl.gz", "rt")]
seen_v1b = {r["p"] for r in old} | {r["p"] for r in v1b}
ws = sorted({r["w"] for r in E}); mid = ws[len(ws) // 2]
print("ère v5.9 : groupes", len(E), "fenêtres", ws[0], "-", ws[-1], "coupe", mid, "hors zone %.1f%%" % (100 * (1 - np.mean([r["z"] for r in E]))))
IZ = InZoneScorer("data/prior_v1/inzone_v1.json")
V = json.load(open("data/predictor_v5.9_2026-08-30_1420.json"))
B = InZoneScorer("data/prior_v1b/inzone_v1b.json")
def fit(rows):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000, sublinear_tf=True)
    X = vec.fit_transform([txt(r["t"]) for r in rows])
    L = csr_matrix(np.log1p([[len(txt(r["t"]))] for r in rows]) / 10.0)
    m = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(hstack([X, L]).tocsr(), [int(r["z"]) for r in rows])
    return lambda ts: m.predict_proba(hstack([vec.transform([txt(t) for t in ts]), csr_matrix(np.log1p([[len(txt(t))] for t in ts]) / 10.0)]).tocsr())[:, 1]
rng = np.random.default_rng(81)
def top(y, s, it=3000):
    o = []
    for _ in range(it):
        i = rng.choice(len(y), 22, replace=False); o.append(y[i[np.argsort(-s[i])[:10]]].mean())
    return 100 * np.mean(o)
def report(title, T, extra):
    T = [r for r in T]
    y = np.array([int(r["z"]) for r in T]); ts = [r["t"] for r in T]
    S = {"inzone_v1": np.array([IZ.proba(t) for t in ts]), "v5.9 (PROD)": np.array([pp.score_prompt(V, t) for t in ts]),
         "V1b": np.array([B.proba(t) for t in ts])}
    S.update({k: f(ts) for k, f in extra.items()})
    print("==", title, "| n=%d en zone %.1f%%" % (len(y), 100 * y.mean()))
    for k, s in S.items():
        print("   %-14s AUC %.3f | en zone top10/22 %.1f%%" % (k, roc_auc_score(y, s), top(y, s)))
# 1) V1b sur l'ère v5.9, prompts jamais vus par V1b
T1 = [r for r in E if r["p"] not in seen_v1b]
report("TEST PROPRE V1b sur l'ère v5.9 (jamais vus)", T1, {})
# 2) V1c = données V1b + 1re moitié ère v5.9 ; test 2e moitié
test2 = [r for r in E if r["w"] > mid]
tp = {r["p"] for r in test2}
train = [r for r in old + [dict(p=r["p"], t=r["t"], z=r["z"]) for r in v1b] + [r for r in E if r["w"] <= mid] if r["p"] not in tp]
fc = fit(train)
report("V1c (entraîné jusqu'à %d) sur 2e moitié" % mid, [r for r in test2 if r["p"] not in seen_v1b], {"V1c": fc})
# 3) mélange V1c + v5.9 (rangs) et sous-ensemble « têtes » (groupes courts = premiers prêts)
from scipy.stats import rankdata
T3 = [r for r in test2 if r["p"] not in seen_v1b]
y = np.array([int(r["z"]) for r in T3]); ts = [r["t"] for r in T3]; ml = np.array([r["ml"] for r in T3])
s59 = np.array([pp.score_prompt(V, t) for t in ts]); sc = fc(ts)
blend = rankdata(sc) + rankdata(s59)
print("== mélange et têtes (2e moitié)")
for k, s in (("v5.9", s59), ("V1c", sc), ("V1c+v5.9", blend)):
    print("   %-10s AUC %.3f | top10/22 %.1f%%" % (k, roc_auc_score(y, s), top(y, s)))
q = np.quantile(ml, 0.33)
sh = ml <= q
print("   groupes courts (traînard <= %d tok, n=%d) : hors zone %.1f%% contre %.1f%% pour les autres" % (q, sh.sum(), 100 * (1 - y[sh].mean()), 100 * (1 - y[~sh].mean())))
for k, s in (("v5.9", s59), ("V1c", sc), ("V1c+v5.9", blend)):
    print("   courts %-10s AUC %.3f" % (k, roc_auc_score(y[sh], s[sh])))
# 4) SIMULATION par fenêtre : 1er bake réel (v5.9) contre 1er bake que V1c aurait choisi
#    dans le vivier de la fenêtre (tous les prompts bakés, labels exacts sous la même
#    randomness). Ordre de sortie ≈ traînard croissant (même règle pour les deux).
bk = [json.loads(l) for l in open("data/prior_v1c/bakes.jsonl")]
labw = {}
for r in E: labw[(r["w"], r["p"])] = r
first_bake = collections.defaultdict(dict)
for b in bk:
    first_bake[b["w"]].setdefault("min", b["bake"])
    first_bake[b["w"]]["min"] = min(first_bake[b["w"]]["min"], b["bake"])
real1 = collections.defaultdict(list)
for b in bk:
    if b["bake"] == first_bake[b["w"]]["min"] and (b["w"], b["p"]) in labw:
        real1[b["w"]].append((b["ready_s"], labw[(b["w"], b["p"])]))
pool = collections.defaultdict(list)
for (w, p), r in labw.items(): pool[w].append(r)
def head(rs, k=3):
    rs = sorted(rs, key=lambda r: r["ml"])
    return sum(r["z"] for r in rs[:k]), sum(r["z"] for r in rs)
res = []
for w in sorted(real1):
    if w <= mid or len(real1[w]) < 8 or len(pool[w]) < 25: continue
    act = [r for _, r in real1[w]]
    act_ready = [r for _, r in sorted(real1[w], key=lambda x: x[0])]
    h_act_true = sum(r["z"] for r in act_ready[:3])            # ordre RÉEL de sortie
    h_act, v_act = head(act)
    cand = pool[w]; s = fc([r["t"] for r in cand])
    pick = [cand[i] for i in np.argsort(-s)[:len(act)]]
    h_v1c, v_v1c = head(pick)
    res.append((w, h_act_true, h_act, v_act, h_v1c, v_v1c, len(act)))
R = np.array([r[1:] for r in res], dtype=float)
print("== SIMULATION 1er bake, fenêtres %d-%d (n=%d)" % (res[0][0], res[-1][0], len(res)))
print("   valides parmi les 3 premiers sortis : réel (ordre réel) %.2f | réel (proxy traînard) %.2f | V1c (proxy) %.2f" % (R[:, 0].mean(), R[:, 1].mean(), R[:, 3].mean()))
print("   valides sur le bake entier         : réel %.2f | V1c %.2f  (sur %.1f groupes)" % (R[:, 2].mean(), R[:, 4].mean(), R[:, 5].mean()))
bad = R[:, 0] <= 1
print("   fenêtres à tête MAUVAISE (≤1 valide parmi les 3 premiers, ordre réel) : %d/%d" % (bad.sum(), len(R)))
print("     dans ces fenêtres : têtes valides réel %.2f -> V1c %.2f | bake entier réel %.2f -> V1c %.2f" % (R[bad, 0].mean(), R[bad, 3].mean(), R[bad, 2].mean(), R[bad, 4].mean()))
print("     V1c aurait eu >=2 valides en tête dans %d/%d de ces fenêtres" % ((R[bad, 3] >= 2).sum(), bad.sum()))
for r in res:
    if r[1] <= 1: print("     fen %d : tête réelle %d/3 -> V1c %d/3 | bake %d -> %d /%d" % (r[0], r[1], r[4], r[3], r[5], r[6]))
# 5) garde-fous : hasard dans le même vivier, et longueur des picks (tête plus lente ?)
rnd = np.random.default_rng(7); hr = []; br = []; mla = []; mlv = []
for w in sorted(real1):
    if w <= mid or len(real1[w]) < 8 or len(pool[w]) < 25: continue
    act = [r for _, r in real1[w]]; cand = pool[w]
    for _ in range(50):
        pk = [cand[i] for i in rnd.choice(len(cand), len(act), replace=False)]
        h, b = head(pk); hr.append(h); br.append(b)
    s = fc([r["t"] for r in cand]); pick = [cand[i] for i in np.argsort(-s)[:len(act)]]
    mla.append(np.median([r["ml"] for r in act])); mlv.append(np.median([r["ml"] for r in pick]))
    mla.append(sorted(r["ml"] for r in act)[2]); mlv.append(sorted(r["ml"] for r in pick)[2])
print("== garde-fous : HASARD dans le vivier : tête %.2f / bake %.2f" % (np.mean(hr), np.mean(br)))
print("   traînard médian du bake : réel %.0f | V1c %.0f ; 3e plus court (tête) : réel %.0f | V1c %.0f" % (np.median(mla[0::2]), np.median(mlv[0::2]), np.median(mla[1::2]), np.median(mlv[1::2])))
