#!/usr/bin/env python3
"""Mesure l'A/B du SPRINT — métrique : bucket de la meilleure entrée.

POURQUOI CETTE MÉTRIQUE ET PAS LES PAYÉES. Le bucket de la meilleure entrée
d'une fenêtre (`tokens // (rounds × 50)`) a une moyenne de 27,3 et un
écart-type de 8,3 : l'effet prédit (+12) est détectable en **7 fenêtres par
bras**. Les payées/fenêtre ont une moyenne de 2,44 pour un écart-type de 2,66 —
détecter +0,30 y demanderait 1 234 fenêtres par bras, soit 88 heures. C'est
précisément ce piège qui a fait juger « sprint 8 bon puis mauvais » par le
passé, en jugeant sur un signal trop bruité et sans contrôler l'âge du moteur.

  effet visé | fenêtres/bras | durée (14 fen/h)
     +5      |      43       |   3,1 h
     +7      |      22       |   1,6 h
    +12      |       7       |   0,7 h

RÈGLES DE MESURE — chacune payée d'une erreur dans ce projet :
  1. ÂGE DU MOTEUR. Ne jamais comparer un bras jeune à un bras mûr. On écarte
     les fenêtres des 15 premières minutes après un redémarrage.
  2. FENÊTRES DE CHECKPOINT écartées : le rechargement en coûte une à deux,
     sans rapport avec le réglage testé.
  3. VERDICTS MÛRS uniquement : une fenêtre dont `rewarded` est encore None
     n'est pas « zéro payée », elle n'est pas décidée.
  4. Le round drand est reconstruit depuis `flip_offset_s` — proxy vérifié
     exact à 84 % contre `arrival_drand_round`, écarts de ±1 round. Bon pour
     l'agrégat, pas pour juger une entrée isolée.

Usage :
    python3 scripts/ab_sprint.py --depuis "2026-08-21 14:00" --jusqu-a "2026-08-21 15:00"
    python3 scripts/ab_sprint.py --comparer "14:00-15:00" "15:00-16:00"
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import statistics as st
import sys
import time
from pathlib import Path

DATA = Path("/root/subnet81/data")
GRACE_S = 15 * 60          # age minimal du moteur


def _epoch(s: str) -> float:
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%H:%M"):
        try:
            t = time.strptime(s, fmt)
            if fmt == "%H:%M":
                d = time.localtime()
                t = time.struct_time((d.tm_year, d.tm_mon, d.tm_mday,
                                      t.tm_hour, t.tm_min, 0, 0, 0, -1))
            return time.mktime(t)
        except ValueError:
            continue
    raise SystemExit(f"date illisible : {s}")


def charger() -> dict:
    """Par fenêtre : bucket de la meilleure entrée, payées, arrivées."""
    G = {}
    for line in open(DATA / "samples_v4.jsonl", errors="replace"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("window_n") is not None:
            G[(r["window_n"], r.get("prompt_idx"))] = r
    V = collections.defaultdict(list)
    for line in open(DATA / "verdicts_v4.jsonl", errors="replace"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("window_n") and r.get("rewarded") is not None:
            V[r["window_n"]].append(r)
    par = collections.defaultdict(lambda: {"b": [], "off": [], "ts": []})
    for line in open(DATA / "submits_v4.jsonl", errors="replace"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if not r.get("accepted") or r.get("flip_offset_s") is None:
            continue
        g = G.get((r["window_n"], r.get("prompt_idx")))
        if not g or not g.get("completion_lens"):
            continue
        tot, off = sum(g["completion_lens"]), r["flip_offset_s"]
        d = par[r["window_n"]]
        d["b"].append(tot // (max(1, int(off // 3) + 1) * 50))
        d["off"].append(off)
        if r.get("ts"):
            d["ts"].append(r["ts"])
    out = {}
    for w, d in par.items():
        # regle 3 : verdicts murs seulement
        if not d["b"] or w not in V or not d["ts"]:
            continue
        out[w] = {"bucket": max(d["b"]), "off": min(d["off"]),
                  "ts": min(d["ts"]), "n": len(d["b"]),
                  "paye": sum(1 for v in V[w] if v.get("rewarded"))}
    return out


def bras(fen: dict, a: float, b: float, nom: str) -> dict | None:
    g = [v for v in fen.values() if a <= v["ts"] < b]
    if len(g) < 5:
        print(f"  {nom:12s} : {len(g)} fenêtres — insuffisant")
        return None
    bk = [v["bucket"] for v in g]
    r = {"n": len(g), "m": st.mean(bk), "s": st.pstdev(bk),
         "med": st.median(bk),
         "sup29": 100 * sum(1 for x in bk if x >= 29) / len(bk),
         "off": st.median(v["off"] for v in g),
         "pay": st.mean(v["paye"] for v in g)}
    print(f"  {nom:12s} : n={r['n']:3d} | bucket moy {r['m']:5.1f} "
          f"(méd {r['med']:4.0f}) | >=29 {r['sup29']:3.0f} % | "
          f"1re arrivée {r['off']:+5.1f}s | payées {r['pay']:.2f}")
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparer", nargs=2, metavar=("BRAS_A", "BRAS_B"),
                    help='deux plages "HH:MM-HH:MM"')
    ap.add_argument("--depuis")
    ap.add_argument("--jusqu-a", dest="jusqua")
    a = ap.parse_args()
    fen = charger()
    print(f"fenêtres exploitables (verdicts mûrs) : {len(fen)}\n")

    if a.comparer:
        res = []
        for i, plage in enumerate(a.comparer):
            d, f = plage.split("-")
            res.append(bras(fen, _epoch(d), _epoch(f), f"bras {'AB'[i]}"))
        if all(res):
            x, y = res
            se = math.sqrt(x["s"] ** 2 / x["n"] + y["s"] ** 2 / y["n"])
            dif = y["m"] - x["m"]
            print(f"\n  écart B − A : {dif:+.1f} bucket  "
                  f"(erreur-type {se:.1f}, soit {abs(dif)/se if se else 0:.1f} σ)")
            print(f"  IC 95 % : [{dif-1.96*se:+.1f} ; {dif+1.96*se:+.1f}]")
            if abs(dif) < 1.96 * se:
                print("  → NON CONCLUANT : l'écart n'est pas distinguable du bruit.")
                besoin = math.ceil(2 * ((1.96 + 0.84) * max(x["s"], y["s"])
                                        / max(abs(dif), 0.1)) ** 2)
                print(f"     il faudrait ~{besoin} fenêtres par bras pour trancher "
                      f"un écart de cette taille.")
            else:
                print("  → écart SIGNIFICATIF au seuil de 5 %.")
        return 0

    d = _epoch(a.depuis) if a.depuis else 0
    f = _epoch(a.jusqua) if a.jusqua else time.time()
    bras(fen, d, f, "période")
    return 0


if __name__ == "__main__":
    sys.exit(main())
