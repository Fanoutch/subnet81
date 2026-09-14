import gzip, json, sys, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack, csr_matrix
sys.path.insert(0, "data/prior_v1")
from inzone_scorer import InZoneScorer, PRE

pos = [json.loads(l) for l in gzip.open("data/prior_v1/v1_code_samples.jsonl.gz", "rt")]
neg = [json.loads(l) for l in gzip.open("data/prior_v1/v1_code_negatives.jsonl.gz", "rt")]
rows = [r for r in pos if r["w"] < 45890] + [r for r in pos if r["w"] >= 45900 and r["z"]] + neg
txt = lambda r: r["t"][len(PRE):] if r["t"].startswith(PRE) else r["t"]
T = [txt(r) for r in rows]; y = np.array([int(bool(r["z"])) for r in rows])
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000, sublinear_tf=True)
X = vec.fit_transform(T)
L = csr_matrix(np.log1p([[len(t)] for t in T]) / 10.0)
m = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(hstack([X, L]).tocsr(), y)
coef = m.coef_[0]
nv = len(vec.vocabulary_)
out = {"vocab": {k: int(v) for k, v in vec.vocabulary_.items()}, "idf": vec.idf_.tolist(),
       "coef": coef[:nv].tolist(), "len_coef": float(coef[nv]), "intercept": float(m.intercept_[0]),
       "meta": {"n": len(rows), "hors_zone": float(1 - y.mean()), "fenetres": [45650, 45919],
                "C": 1.0, "cible": "in_zone sigma>=0.24 (local), V1 code"}}
json.dump(out, open("data/prior_v1/inzone_v1.json", "w"))
# parité scikit-learn <-> scoreur pur
sc = InZoneScorer("data/prior_v1/inzone_v1.json")
p_sk = m.predict_proba(hstack([vec.transform(T), L]).tocsr())[:, 1]
p_py = np.array([sc.proba(r["t"]) for r in rows])
print("n", len(rows), "écart max scikit/pur:", float(np.abs(p_sk - p_py).max()))
