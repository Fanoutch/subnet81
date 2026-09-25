#!/usr/bin/env python3
"""Prior MATH v2 (25/09) = prior v1 (mots) − λ·indice de marge (mots + méta).

- v1 : régression logistique positifs R2 / non étiquetés OMI (mots seuls),
  ``train_prior_math_pu.py`` — le prior en production depuis le 24/09.
- indice de marge : sur les groupes math SÉLECTIONNÉS (R2, donc en zone),
  P(k ≥ 14) — problème « presque trop facile », souvent 16/16 dans une autre
  fenêtre. Traits : mots + ``prompt_predictor.math_meta_tokens`` (source OMI,
  réponse attendue, longueur de l'énoncé).
- combinaison : z(v1) − λ·z(marge), z calculé sur la population tirée au
  hasard (``unlabeled_omi.jsonl``) ; les deux étant linéaires, le résultat est
  UN modèle linéaire ``{"type": "linear", "meta": "math_v1"}`` lu par
  ``prompt_predictor.score_problem``.

Éval hors ligne du 25/09 sur 1 064 tirages au hasard : AUC 0,646 → 0,675,
top-5 % 58,5 % → 67,9 % (intervalles chevauchants). L'éval ci-dessous passe
par ``score_problem``, exactement comme le mineur.

Entrées (dev box) : data/math_box/{samples_math,ooz_math,unlabeled_omi,
meta_idx,meta_r2}.jsonl + le pickle R2 (prompt -> idx, k, réponse, maxlen).
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from reliquary.miner import prompt_predictor as pp  # noqa: E402
from train_prior_math import auc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/root/subnet81/data/math_box")
    ap.add_argument("--r2-pickle", required=True)
    ap.add_argument("--v1", default="/root/subnet81/data/predictor_math_pu_C0.02.json")
    ap.add_argument("--lam", type=float, default=0.6)
    ap.add_argument("--C", type=float, default=0.05)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from scipy.sparse import hstack

    D = Path(a.data)
    meta = {}
    for l in open(D / "meta_idx.jsonl"):
        r = json.loads(l)
        meta[r["i"]] = (r["src"], r["ans"])
    r2src = {}
    for l in open(D / "meta_r2.jsonl"):
        r = json.loads(l)
        r2src[r["i"]] = r["src"]
    r2 = pickle.load(open(a.r2_pickle, "rb"))

    ev = []
    for f, z in (("samples_math.jsonl", 1), ("ooz_math.jsonl", 0)):
        for l in open(D / f):
            r = json.loads(l)
            if r.get("prompt") and r.get("source") in ("scan", "explore") and r["prompt_idx"] in meta:
                src, ans = meta[r["prompt_idx"]]
                ev.append(({"prompt": r["prompt"], "source": src, "ground_truth": ans},
                           1 if (z and r.get("in_zone")) else 0))
    evt = {p["prompt"] for p, _ in ev}
    pos = [(t, r2src[v[0]], v[2], v[1]) for t, v in r2.items()
           if t not in evt and v[0] in r2src]
    unl = []
    for l in open(D / "unlabeled_omi.jsonl"):
        r = json.loads(l)
        if r["prompt_idx"] in meta:
            src, ans = meta[r["prompt_idx"]]
            unl.append({"prompt": r["prompt"], "source": src, "ground_truth": ans})
    print(f"éval (hasard) {len(ev)} | R2 positifs {len(pos)} | population {len(unl)}")

    # indice de marge : P(k >= 14) sur les positifs R2, mots + méta
    def feats(prompt, src, ans):
        return list(set(pp.tokenize(prompt))) + pp.math_meta_tokens(prompt, src, ans)

    vec = CountVectorizer(analyzer=lambda x: x, binary=True, min_df=5)
    X = vec.fit_transform([feats(t, s, g) for t, s, g, _ in pos])
    yk = np.array([1 if k >= 14 else 0 for *_, k in pos])
    mk = LogisticRegression(C=a.C, max_iter=3000, solver="liblinear").fit(X, yk)
    wm = dict(zip(vec.get_feature_names_out(), mk.coef_[0]))
    marge = {"type": "linear", "meta": "math_v1", "bias": float(mk.intercept_[0]),
             "weights": {str(t): float(w) for t, w in wm.items()}}

    v1 = pp.load_model(a.v1)
    s1 = np.array([pp.score_problem(v1, p) for p in unl])
    sm = np.array([pp.score_problem(marge, p) for p in unl])
    m1, d1, mm, dm = s1.mean(), s1.std(), sm.mean(), sm.std()
    w = {}
    for t, x in v1["weights"].items():
        w[t] = w.get(t, 0.0) + x / d1
    for t, x in marge["weights"].items():
        w[t] = w.get(t, 0.0) - a.lam * x / dm
    model = {"type": "linear", "meta": "math_v1",
             "bias": float((v1.get("bias", 0.0) - m1) / d1
                           - a.lam * (marge["bias"] - mm) / dm),
             "weights": {t: round(x, 6) for t, x in w.items() if abs(x) >= 1e-6}}

    ye = np.array([y for _, y in ev])
    for name, m in (("v1 (production)", v1), (f"v2 (λ={a.lam})", model)):
        s = np.array([pp.score_problem(m, p) for p, _ in ev])
        out = [f"AUC {auc(list(zip(s, ye))):.3f}"]
        for fr in (0.10, 0.05):
            k = int(len(s) * fr)
            out.append(f"top{int(fr * 100)}% {ye[np.argsort(-s)[:k]].mean():.1%}")
        print(f"{name:18s} base {ye.mean():.1%} | " + " | ".join(out))
    model["meta_info"] = {"built": "2026-09-25", "lambda": a.lam, "C": a.C,
                          "v1": a.v1, "positives": len(pos), "population": len(unl)}
    if a.out:
        pp.save_model(model, a.out)
        print(f"modèle → {a.out} ({len(model['weights'])} poids)")


if __name__ == "__main__":
    main()
