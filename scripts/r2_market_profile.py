#!/usr/bin/env python3
"""Profil de marché à partir des archives R2 : qui envoie quoi, quand, et qui est payé.

Construit une table AU NIVEAU DE L'ENTRÉE (une ligne = une soumission d'un mineur
dans une fenêtre) à partir de batch[] + rejected[], puis sort :
  1. la courbe P(payé) vs arrivée, par environnement, poolée sur tout le marché ;
  2. le classement des mineurs (part d'émission, entrées/fenêtre, arrivée, présence) ;
  3. notre position dans ce classement.

Usage:  python3 scripts/r2_market_profile.py --cache /chemin/cache [--nous <ss58>]
"""
import argparse
import glob
import gzip
import json
import statistics as st
from collections import Counter, defaultdict

NOUS = "5DvpFN3QEa9iimQiA5jQaRmx8dbW2uxonM53j51Cw3kBva7q"
BINS = [(0, 6), (6, 8), (8, 10), (10, 12), (12, 15), (15, 18),
        (18, 21), (21, 25), (25, 30), (30, 100)]


def charger(cache):
    """Une ligne par soumission. `age` = arrival_ts - window_opened_wall_ts,
    c'est-à-dire l'arrivée dans le référentiel DU VALIDATEUR."""
    rows, emis, wins = [], defaultdict(float), {}
    for f in sorted(glob.glob(f"{cache}/*.json.gz")):
        d = json.loads(gzip.decompress(open(f, "rb").read()))
        w = d["window_start"]
        wins[w] = d.get("window_opened_wall_ts_by_environment", {})
        for hk, v in d.get("rewards_by_hotkey", {}).items():
            emis[hk] += v
        for e in d["batch"]:
            rows.append(dict(
                w=w, hk=e["hotkey"], env=e["env_name"], age=e.get("arrival_age_seconds"),
                rank=e.get("canonical_rank"), val=e.get("difficulty_auction_value"),
                rew=bool(e.get("rewarded")), reason=None,
                vol=sum(r["completion_length"] for r in e["rollouts"]),
                pidx=e["prompt_idx"]))
        for e in d["rejected"]:
            rows.append(dict(
                w=w, hk=e["hotkey"], env=e["env_name"], age=e.get("arrival_age_seconds"),
                rank=e.get("canonical_rank"), val=e.get("difficulty_auction_value"),
                rew=False, reason=e.get("reason"), vol=None, pidx=e["prompt_idx"]))
    return rows, emis, wins


def courbe(rows, env):
    out = {}
    for lo, hi in BINS:
        s = [r for r in rows if r["env"] == env and r["age"] is not None and lo <= r["age"] < hi]
        out[(lo, hi)] = (len(s), sum(1 for r in s if r["rew"]) / len(s) if s else float("nan"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--nous", default=NOUS)
    a = ap.parse_args()
    rows, emis, wins = charger(a.cache)
    nw = len(wins)
    print(f"{nw} fenêtres, {len(rows)} entrées, {len({r['hk'] for r in rows})} hotkeys\n")

    cc, cm = courbe(rows, "opencodeinstruct"), courbe(rows, "openmathinstruct")
    print("=== P(payé) vs arrivée (référentiel validateur) ===")
    print(f"  {'arrivée':>10s} | {'n code':>7s} {'P code':>7s} | {'n math':>7s} {'P math':>7s}")
    for b in BINS:
        print(f"  {b[0]:3d}-{b[1]:3d} s | {cc[b][0]:7d} {cc[b][1]:7.3f} | {cm[b][0]:7d} {cm[b][1]:7.3f}")

    print("\n=== Contention par environnement ===")
    for env in ("opencodeinstruct", "openmathinstruct"):
        s = [r for r in rows if r["env"] == env]
        perw = Counter(r["w"] for r in s)
        paid = [r for r in s if r["rew"]]
        print(f"  {env:20s} {st.median(perw.values()):5.0f} entrées/fen  "
              f"{len({r['hk'] for r in s}):3d} mineurs  "
              f"{len(paid)/len(perw):5.1f} payées/fen  taux global {len(paid)/len(s):.3f}")

    print("\n=== Classement des mineurs ===")
    ent, seen, paid = defaultdict(list), defaultdict(set), Counter()
    for r in rows:
        ent[r["hk"]].append(r)
        seen[r["hk"]].add(r["w"])
        if r["rew"]:
            paid[r["hk"]] += 1
    tot = sum(emis.values())
    print(f"  {'#':>3} {'hotkey':13s} {'part%':>6s} {'ent/fen':>7s} {'payées/fen':>10s} "
          f"{'P(payé)':>7s} {'arr.méd':>7s} {'présence':>8s}")
    for i, (hk, e) in enumerate(sorted(emis.items(), key=lambda x: -x[1]), 1):
        E, n = ent[hk], len(seen[hk]) or 1
        ages = sorted(r["age"] for r in E if r["age"] is not None)
        if i <= 10 or hk == a.nous:
            print(f"  {i:3d} {hk[:13]} {100*e/tot:6.2f} {len(E)/n:7.2f} {paid[hk]/n:10.2f} "
                  f"{paid[hk]/max(len(E),1):7.2f} {st.median(ages) if ages else -1:7.2f} "
                  f"{100*len(seen[hk])/nw:7.1f}%" + ("  <<< NOUS" if hk == a.nous else ""))


if __name__ == "__main__":
    main()
