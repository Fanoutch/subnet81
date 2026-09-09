#!/usr/bin/env python3
"""A/B de la branche fix/course-2026-08-27 contre la config d'avant.

Chaque fix a sa signature propre — on juge chacun sur la sienne, pas sur un
agrégat unique :

  HOT_SWAP + PREFETCH  -> présence dans les 2 fenêtres qui suivent chaque
                          avancée de checkpoint (référence : 0/22)
  HEADROOM             -> taux de stale_round dans submits_v4.jsonl
                          (référence : 37 %) — invisible dans R2, le precommit
                          rejeté ne produit aucune ligne d'archive
  VOLUME_MU=0          -> volume des groupes sélectionnés (référence : 11 048)
                          + arrivée de la 1re entrée + part de sigma=0
  global               -> payées/fenêtre, à lire APRÈS ≥30 fenêtres mûres
                          (règle CLAUDE.md : trois lectures précoces se sont
                          dégonflées le 24/08)

Usage :
  python3 scripts/r2_pull_windows.py --debut <A> --fin <B> --out <cache>
  python3 scripts/r2_ab_fixes.py --cache <cache> --coupure <fenêtre du restart>
          [--submits data/submits_v4.jsonl]
"""
import argparse
import glob
import gzip
import json
import statistics as st
from collections import Counter, defaultdict

NOUS = "5DvpFN3QEa9iimQiA5jQaRmx8dbW2uxonM53j51Cw3kBva7q"


def charger(cache):
    W = {}
    for f in sorted(glob.glob(f"{cache}/*.json.gz")):
        d = json.loads(gzip.decompress(open(f, "rb").read()))
        W[d["window_start"]] = d
    return W


def avancees(W, cont):
    """Fenêtres où le checkpoint majoritaire du batch change."""
    ck = {}
    for w in cont:
        hs = Counter(e.get("claimed_checkpoint_hash")
                     for e in W[w]["batch"] if e.get("claimed_checkpoint_hash"))
        ck[w] = hs.most_common(1)[0][0] if hs else None
    return [w for a, w in zip(cont, cont[1:])
            if ck.get(a) and ck.get(w) and ck[a] != ck[w]]


def bras(W, cont, nous):
    adv = avancees(W, cont)
    near = set()
    for a in adv:
        i = cont.index(a)
        near |= set(cont[i:i + 2])
    pres = {w for w in cont for e in W[w]["batch"] + W[w]["rejected"]
            if e["hotkey"] == nous and e["env_name"] == "opencodeinstruct"}
    mine, e1, vols, sig0 = [], [], [], 0
    for w in cont:
        d = W[w]
        m = [e for e in d["batch"]
             if e["hotkey"] == nous and e["env_name"] == "opencodeinstruct"]
        r = [e for e in d["rejected"]
             if e["hotkey"] == nous and e["env_name"] == "opencodeinstruct"]
        mine += m
        sig0 += sum(1 for e in r if e.get("reason") == "out_of_zone")
        ages = [e.get("arrival_age_seconds") for e in m + r
                if e.get("arrival_age_seconds") is not None]
        if ages:
            e1.append(min(ages))
        vols += [sum(x["completion_length"] for x in e["rollouts"]) for e in m]
    far = [w for w in cont if w not in near]
    return dict(
        n=len(cont), adv=len(adv),
        pres_far=(sum(1 for w in far if w in pres), len(far)),
        pres_near=(sum(1 for w in near if w in pres), len(near)),
        payees=len(mine), payees_fen=len(mine) / max(len(cont), 1),
        e1=st.median(e1) if e1 else None,
        vol=st.median(vols) if vols else None,
        out_of_zone=sig0,
    )


def stale_depuis_submits(path, coupure):
    """Taux de stale_round avant/après, depuis le dump du mineur."""
    out = {}
    try:
        rows = [json.loads(l) for l in open(path)]
    except OSError:
        return None
    for lbl, sel in [("avant", lambda w: w < coupure),
                     ("apres", lambda w: w >= coupure)]:
        g = [r for r in rows if r.get("window_n") and sel(r["window_n"])]
        n = len(g)
        s = sum(1 for r in g if r.get("reason") == "stale_round")
        out[lbl] = (s, n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--coupure", type=int, required=True,
                    help="première fenêtre de la nouvelle config")
    ap.add_argument("--rodage", type=int, default=3,
                    help="fenêtres écartées après la coupure (moteur froid)")
    ap.add_argument("--submits", default="data/submits_v4.jsonl")
    ap.add_argument("--nous", default=NOUS)
    a = ap.parse_args()

    W = charger(a.cache)
    cont = sorted(W)
    avant = [w for w in cont if w < a.coupure]
    apres = [w for w in cont if w >= a.coupure + a.rodage]
    A, B = bras(W, avant, a.nous), bras(W, apres, a.nous)

    print(f"AVANT : {len(avant)} fen  |  APRÈS : {len(apres)} fen "
          f"(rodage {a.rodage} écarté)")
    if len(apres) < 30:
        print(f"⚠️  {len(apres)} fenêtres après — EN DESSOUS des 30 requises, "
              "lecture indicative seulement")
    print(f"\n{'signature':<38s} {'AVANT':>14s} {'APRÈS':>14s}")

    def ratio(t):
        return f"{t[0]}/{t[1]}" + (f" ({100*t[0]/t[1]:.0f}%)" if t[1] else "")
    print(f"{'[ckpt] présence 2 fen post-avancée':<38s} "
          f"{ratio(A['pres_near']):>14s} {ratio(B['pres_near']):>14s}")
    print(f"{'[ckpt] présence hors avancée':<38s} "
          f"{ratio(A['pres_far']):>14s} {ratio(B['pres_far']):>14s}")
    print(f"{'[volume] vol méd des sélectionnés':<38s} "
          f"{A['vol'] or float('nan'):>14.0f} {B['vol'] or float('nan'):>14.0f}")
    print(f"{'[volume] e1 méd (validateur)':<38s} "
          f"{A['e1'] or float('nan'):>14.1f} {B['e1'] or float('nan'):>14.1f}")
    print(f"{'[volume] rejets out_of_zone':<38s} "
          f"{A['out_of_zone']:>14d} {B['out_of_zone']:>14d}")
    print(f"{'[global] payées/fenêtre':<38s} "
          f"{A['payees_fen']:>14.2f} {B['payees_fen']:>14.2f}")

    stale = stale_depuis_submits(a.submits, a.coupure)
    if stale:
        print(f"{'[headroom] stale_round (submits)':<38s} "
              f"{ratio(stale['avant']):>14s} {ratio(stale['apres']):>14s}")
    else:
        print(f"(submits introuvable : {a.submits} — signature headroom absente)")


if __name__ == "__main__":
    main()
