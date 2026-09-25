#!/usr/bin/env python3
"""Prior MATH par apprentissage positifs/non-étiquetés (24/09).

Positifs  : énoncés math SÉLECTIONNÉS par le validateur (archives R2, donc en
            zone) — sauf ceux de NOTRE hotkey math (fuite vers l'éval).
Non étiq. : énoncés OMI tirés au hasard dans l'univers (``unlabeled_omi.jsonl``,
            extrait sur la box) ≈ la population que le mineur tire.
Modèle    : régression logistique sur la présence des unigrammes+bigrammes de
            ``prompt_predictor.tokenize`` ; le score P(positif vs non étiqueté)
            est monotone en P(en zone) (Elkan & Noto 2008) — seul l'ORDRE sert
            au tirage.
Éval      : TOUTES nos étiquettes (samples_math = en zone, ooz_math = 16/16 ou
            0/16) — jamais vues à l'entraînement, et tirées uniformément, donc
            sans le biais de sélection des autres mineurs.
Export    : ``{"type": "linear", "bias", "weights"}`` lu par
            ``prompt_predictor.score_prompt`` (aucune dépendance sur la box).

Usage : python3 scripts/train_prior_math_pu.py --data data/math_box \
          --r2 data/r2_cache_0924 data/r2_cache_0919 [--out FICHIER]
"""
import argparse
import glob
import gzip
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from reliquary.miner import prompt_predictor as pp  # noqa: E402
from train_prior_math import auc, load_labels, top_rate  # noqa: E402

MATH_HOTKEY = "5Gy8EzpC6PXZX1XyCfRUmuYp1Qk6ZJwtFE2GHXudiBMtiJaW"  # hotkey81.2


def r2_positives(dirs, exclude):
    out = set()
    for d in dirs:
        for f in sorted(glob.glob(str(Path(d) / "*.json.gz"))):
            try:
                arc = json.load(gzip.open(f))
            except Exception:
                continue
            for e in arc.get("batch") or []:
                if (e.get("env_name") == "openmathinstruct" and e.get("prompt")
                        and e.get("hotkey") != MATH_HOTKEY
                        and e["prompt"] not in exclude):
                    out.add(e["prompt"])
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--r2", nargs="+", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--C", type=float, default=0.5)
    ap.add_argument("--min-df", type=int, default=5)
    a = ap.parse_args()
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression

    labels = load_labels(Path(a.data))
    ev_texts = {r["prompt"] for r in labels}
    pos = r2_positives(a.r2, exclude=ev_texts)
    unl = []
    for line in open(Path(a.data) / "unlabeled_omi.jsonl"):
        t = json.loads(line)["prompt"]
        if t not in ev_texts:
            unl.append(t)
    print(f"positifs R2 {len(pos)} | non étiquetés {len(unl)} | éval (nos étiquettes) "
          f"{len(labels)} dont en zone {sum(r['y'] for r in labels) / max(1, len(labels)):.1%}")

    vec = CountVectorizer(analyzer=pp.tokenize, binary=True, min_df=a.min_df)
    X = vec.fit_transform(pos + unl)
    y = [1] * len(pos) + [0] * len(unl)
    clf = LogisticRegression(C=a.C, class_weight="balanced", max_iter=2000,
                             solver="liblinear")
    t0 = time.time()
    clf.fit(X, y)
    print(f"entraîné en {time.time() - t0:.1f} s, {X.shape[1]} traits")

    vocab = vec.get_feature_names_out()
    coef = clf.coef_[0]
    model = {"type": "linear", "bias": float(clf.intercept_[0]),
             "weights": {str(t): round(float(w), 5)
                         for t, w in zip(vocab, coef) if abs(w) >= 1e-4}}
    # l'éval passe par score_prompt, exactement comme sur la box
    scored = [(pp.score_prompt(model, r["prompt"]), r["y"]) for r in labels]
    base = sum(yy for _, yy in scored) / len(scored)
    t10, n10 = top_rate(scored, 0.10)
    t25, n25 = top_rate(scored, 0.25)
    au = auc(scored)
    print(f"AUC = {au:.3f}")
    print(f"en zone : base {base:.1%} | top-25 % {t25:.1%} (n={n25}) | "
          f"top-10 % {t10:.1%} (n={n10}) → ×{t10 / base:.2f}")
    top = sorted(zip(coef, vocab))
    print("plus négatifs :", ", ".join(t for _, t in top[:15]))
    print("plus positifs :", ", ".join(t for _, t in top[-15:]))
    model["meta"] = {"trained": time.strftime("%F %T"), "env": "openmathinstruct",
                     "target": "pu_r2_selected_vs_omi", "positives": len(pos),
                     "unlabeled": len(unl), "auc_own_labels": round(au, 3),
                     "top10_own": round(t10, 3), "base_own": round(base, 3),
                     "C": a.C, "min_df": a.min_df, "protocol": 6}
    if a.out:
        pp.save_model(model, a.out)
        print(f"modèle → {a.out} ({len(model['weights'])} poids)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
