#!/usr/bin/env python3
"""Ré-entraîne le prédicteur de VOLUME sur une fenêtre GLISSANTE de données.

POURQUOI — mesuré le 21/08 par analyse indépendante, sur 4 points de coupe :
entraîner sur les 12 dernières heures bat entraîner sur 24 h, 48 h, ou tout le
corpus. L'ordre est stable partout (12 h > 24 h > 48 h ≈ tout) et TOUS très
au-dessus du prior de valeur en vol. La fraîcheur bat le volume de données.
Le script nocturne existant entraîne sur TOUT le corpus — mauvais réglage.

Gain visé : `volume_v1.json` est figé au 20/08 20h14 et obtient une AUC de
0,622 sur le label « payé » ; un modèle de volume ré-entraîné frais atteint
0,717. En taux de payées du tiers supérieur : 40,7 % → 45,8 %.

ÉVALUATION — deux précautions, chacune payée d'une erreur dans ce projet :
  - coupe TEMPORELLE : on évalue sur des groupes STRICTEMENT POSTÉRIEURS à
    l'entraînement, jamais sur un échantillon aléatoire ;
  - coupe PAR PROMPT en plus : un prompt de l'évaluation ne doit jamais
    apparaître à l'entraînement. Sans elle on mesure la mémorisation — c'est
    ce qui avait invalidé sept duels du prédicteur v4.3.

Les traits sont IDENTIQUES à volume_v1 (fréquences L2-normalisées des 400
premiers mots + longueur bornée) pour que le câblage existant fonctionne sans
modification : `prompt_predictor.volume_score` lit le même format.

Usage :
    python3 scripts/train_volume_v2.py            # 12 h, évalue sur les 2 dernières
    python3 scripts/train_volume_v2.py --heures 24
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path

DATA = Path("/root/subnet81/data")
TOK_RE = None


def feats(text: str) -> dict:
    """IDENTIQUE à prompt_predictor.volume_features — ne pas diverger."""
    global TOK_RE
    if TOK_RE is None:
        import re
        TOK_RE = re.compile(r"[a-z_]{3,}")
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
    """Groupes SAINS de la fenêtre glissante (min rollout >= 32 tok).

    Le gate CHALLENGE_K rejette tout groupe dont un rollout fait moins de
    32 tokens : les inclure apprendrait à viser des groupes inéligibles.
    """
    limite = time.time() - heures * 3600
    out = []
    for line in open(DATA / "samples_v4.jsonl"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        cl = r.get("completion_lens")
        if not cl or not r.get("prompt") or not r.get("ts"):
            continue
        if r["ts"] < limite or min(cl) < 32:
            continue
        out.append(r)
    return out


def entrainer(rows: list[dict], epochs: int = 60, lr: float = 0.5,
              l2: float = 1e-4, graine: int = 12345) -> tuple[dict, float]:
    """Ridge par descente sur log(total tokens), agrégé PAR PROMPT."""
    random.seed(graine)
    par_prompt = defaultdict(list)
    txt = {}
    for r in rows:
        par_prompt[r["prompt_idx"]].append(sum(r["completion_lens"]))
        txt[r["prompt_idx"]] = r["prompt"]
    y = {p: math.log(st.mean(v)) for p, v in par_prompt.items()}
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
    return {"w": dict(w), "b": b, "X": X, "y": y}, b


def spearman(xs, ys) -> float:
    def rk(v):
        s = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for i, j in enumerate(s):
            r[j] = i
        return r
    a, c = rk(xs), rk(ys)
    n = len(a)
    if n < 3:
        return 0.0
    return 1 - 6 * sum((a[i] - c[i]) ** 2 for i in range(n)) / (n * (n * n - 1))


def scorer(modele: dict, text: str) -> float:
    w, b = modele["weights"], modele["intercept"]
    s = b + sum(w.get(k, 0.0) * v for k, v in feats(text).items())
    return (s - modele["center"]) / (modele["scale"] or 1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heures", type=float, default=12.0,
                    help="fenêtre glissante d'entraînement")
    ap.add_argument("--eval-heures", type=float, default=2.0,
                    help="durée finale réservée à l'évaluation")
    ap.add_argument("--sortie", default=str(DATA / "volume_v2.json"))
    a = ap.parse_args()

    rows = charger(a.heures + a.eval_heures)
    if len(rows) < 2000:
        print(f"corpus insuffisant : {len(rows)} groupes")
        return 1
    coupe = time.time() - a.eval_heures * 3600
    tr = [r for r in rows if r["ts"] < coupe]
    te = [r for r in rows if r["ts"] >= coupe]
    prompts_tr = {r["prompt_idx"] for r in tr}
    # coupe PAR PROMPT en plus de la coupe temporelle
    te = [r for r in te if r["prompt_idx"] not in prompts_tr]
    print(f"entraînement : {len(tr)} groupes / {len(prompts_tr)} prompts "
          f"(fenêtre {a.heures} h)")
    print(f"évaluation   : {len(te)} groupes POSTÉRIEURS et jamais vus "
          f"({a.eval_heures} h)")
    if len(te) < 200:
        print("évaluation trop maigre — élargir --eval-heures")
        return 1

    fit, b = entrainer(tr)
    ys = list(fit["y"].values())
    modele = {
        "kind": "volume_log_tokens",
        "version": f"v2_{time.strftime('%Y-%m-%d_%H%M', time.gmtime())}",
        "intercept": round(b, 6),
        "weights": {k: round(v, 6) for k, v in fit["w"].items() if abs(v) > 2e-3},
        "center": round(st.mean(ys), 6),
        "scale": round(st.pstdev(ys) or 1.0, 6),
        "n_prompts": len(fit["y"]),
        "fenetre_h": a.heures,
        "note": ("Ré-entraîné sur fenêtre glissante — la fraîcheur bat le "
                 "volume de données (mesuré sur 4 coupes : 12 h > 24 h > 48 h "
                 "> tout le corpus). Évalué sur des groupes POSTÉRIEURS et "
                 "prompts disjoints."),
    }

    # comparaison honnête avec le modèle en réserve
    reel = [sum(r["completion_lens"]) for r in te]
    pred_v2 = [scorer(modele, r["prompt"]) for r in te]
    print(f"\n  v2 (frais)  Spearman {spearman(pred_v2, reel):+.3f}  "
          f"({len(modele['weights'])} poids)")
    try:
        v1 = json.load(open(DATA / "volume_v1.json"))
        pred_v1 = [scorer(v1, r["prompt"]) for r in te]
        print(f"  v1 (figé)   Spearman {spearman(pred_v1, reel):+.3f}")
    except Exception:
        pass

    # ce qui compte vraiment : le volume qu'on obtiendrait en prenant le haut
    paires = sorted(zip(pred_v2, reel), reverse=True)
    k = max(1, len(paires) // 4)
    print(f"\n  top 25 % du score → {st.mean(v for _, v in paires[:k]):.0f} tok")
    print(f"  bas 25 %          → {st.mean(v for _, v in paires[-k:]):.0f} tok")
    print(f"  référence (tous)  → {st.mean(reel):.0f} tok")

    json.dump(modele, open(a.sortie, "w"))
    print(f"\n  écrit : {a.sortie}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
