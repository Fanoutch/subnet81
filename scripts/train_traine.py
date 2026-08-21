#!/usr/bin/env python3
"""Prédit la TRAÎNE d'un groupe — max(completion_lens) / moyenne — depuis le prompt.

POURQUOI — mesuré le 21/08 sur 429 entrées tranchées : à volume ÉGAL, un groupe
« uniforme » (ratio max/moyenne sous la médiane de 2,01) est payé **49 %** du
temps contre **17 %** pour un groupe « à traîne ». L'effet tient dans CHAQUE
bande de volume (36 % vs 8 % sous 4 000 tok ; 68 % vs 36 % entre 5 000 et
6 500), et l'arrivée est identique dans les deux groupes (+9,4 s vs +9,6 s) —
ce n'est donc ni un effet de volume ni un effet de timing.

Rang médian : 17 pour les uniformes, 32 pour ceux à traîne.

⚠️ CE QU'ON PRÉDIT EST LE RATIO, PAS LE MAXIMUM. Pénaliser `max` seul
pénaliserait aussi le volume — or le volume, lui, est BON (le rang est
`tokens // (rounds × 50)`). Le ratio isole ce qui coûte sans rapporter.

Traits identiques à `prompt_predictor.volume_features` pour réutiliser le même
chargeur. Évaluation à DOUBLE coupe : temporelle ET par prompt.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path

DATA = Path("/root/subnet81/data")
TOK_RE = re.compile(r"[a-z_]{3,}")


def feats(text: str) -> dict:
    words = TOK_RE.findall((text or "").lower())
    f: dict[str, float] = {}
    for t in words[:400]:
        f[t] = f.get(t, 0.0) + 1.0
    norm = (sum(v * v for v in f.values()) ** 0.5) or 1.0
    for k in f:
        f[k] /= norm
    f["__len__"] = min(len(text or "") / 2000.0, 3.0)
    return f


def charger(heures: float) -> list[dict]:
    limite = time.time() - heures * 3600
    out = []
    for line in open(DATA / "samples_v4.jsonl"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        cl = r.get("completion_lens")
        if not cl or len(cl) < 4 or not r.get("prompt") or not r.get("ts"):
            continue
        if r["ts"] < limite or min(cl) < 32:
            continue
        moy = sum(cl) / len(cl)
        if moy <= 0:
            continue
        r["_ratio"] = max(cl) / moy
        out.append(r)
    return out


def entrainer(rows, epochs=60, lr=0.5, l2=1e-4, graine=12345):
    random.seed(graine)
    par = defaultdict(list)
    txt = {}
    for r in rows:
        par[r["prompt_idx"]].append(r["_ratio"])
        txt[r["prompt_idx"]] = r["prompt"]
    y = {p: math.log(st.mean(v)) for p, v in par.items()}
    X = {p: feats(txt[p]) for p in y}
    w = defaultdict(float)
    b = st.mean(y.values())
    ordre = list(y)
    for _ in range(epochs):
        random.shuffle(ordre)
        for p in ordre:
            f = X[p]
            e = (b + sum(w[k] * v for k, v in f.items())) - y[p]
            b -= lr * e * 0.01
            for k, v in f.items():
                w[k] -= lr * (e * v + l2 * w[k])
    return w, b, list(y.values())


def spearman(xs, ys):
    def rk(v):
        s = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for i, j in enumerate(s):
            r[j] = i
        return r
    a, c = rk(xs), rk(ys)
    n = len(a)
    return 1 - 6 * sum((a[i] - c[i]) ** 2 for i in range(n)) / (n * (n * n - 1)) if n > 2 else 0.0


def scorer(m, text):
    s = m["intercept"] + sum(m["weights"].get(k, 0.0) * v
                             for k, v in feats(text).items())
    return (s - m["center"]) / (m["scale"] or 1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heures", type=float, default=12.0)
    ap.add_argument("--eval-heures", type=float, default=1.0)
    ap.add_argument("--sortie", default=str(DATA / "traine_v1.json"))
    a = ap.parse_args()

    rows = charger(a.heures + a.eval_heures)
    coupe = time.time() - a.eval_heures * 3600
    tr = [r for r in rows if r["ts"] < coupe]
    prompts_tr = {r["prompt_idx"] for r in tr}
    te = [r for r in rows if r["ts"] >= coupe and r["prompt_idx"] not in prompts_tr]
    print(f"entraînement : {len(tr)} groupes / {len(prompts_tr)} prompts")
    print(f"évaluation   : {len(te)} groupes POSTÉRIEURS et jamais vus")
    if len(tr) < 1000 or len(te) < 150:
        print("corpus insuffisant")
        return 1

    w, b, ys = entrainer(tr)
    m = {
        "kind": "traine_log_ratio",
        "version": f"v1_{time.strftime('%Y-%m-%d_%H%M', time.gmtime())}",
        "intercept": round(b, 6),
        "weights": {k: round(v, 6) for k, v in w.items() if abs(v) > 2e-3},
        "center": round(st.mean(ys), 6),
        "scale": round(st.pstdev(ys) or 1.0, 6),
        "n_prompts": len(prompts_tr),
        "note": ("Prédit log(max/moyenne des completion_lens). À volume égal, un "
                 "groupe uniforme est payé 49 % contre 17 % pour un groupe à "
                 "traîne (429 entrées tranchées, 21/08). S'utilise en MALUS de "
                 "tri, jamais en exclusion."),
    }
    pred = [scorer(m, r["prompt"]) for r in te]
    reel = [r["_ratio"] for r in te]
    print(f"\n  Spearman prédit ~ réel : {spearman(pred, reel):+.3f} "
          f"({len(m['weights'])} poids)")
    paires = sorted(zip(pred, reel))
    k = max(1, len(paires) // 4)
    print(f"  bas 25 % du score (visés) → ratio moyen {st.mean(v for _, v in paires[:k]):.2f}")
    print(f"  haut 25 % (à éviter)      → ratio moyen {st.mean(v for _, v in paires[-k:]):.2f}")
    print(f"  référence (tous)          → ratio moyen {st.mean(reel):.2f}")
    json.dump(m, open(a.sortie, "w"))
    print(f"\n  écrit : {a.sortie}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
