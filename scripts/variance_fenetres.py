#!/usr/bin/env python3
"""D'où vient la variation des payés d'une fenêtre à l'autre ? (H100, 19/09)

Par fenêtre : payés (R2, vérité), marché (56e place code, payés des AUTRES
avant 17 s — R2) et nous (journal : signal→bake, groupe 1 prêt, hors zone du
1er bake ; soumissions : tirs avant 17/18 s). Puis part de variance expliquée.
Usage : variance_fenetres.py LO HI LOG[,LOG2] SUBMITS [EXCLURE,...]
"""
import collections, glob, gzip, json, re, statistics as st, sys

US = "5DvpFN3QEa9iimQiA5jQaRmx8dbW2uxonM53j51Cw3kBva7q"; ENV = "opencodeinstruct"
lo, hi, logs, subs = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
excl = {int(x) for x in sys.argv[5].split(",")} if len(sys.argv) > 5 else set()

# ── R2 ──
R = {}
for p in glob.glob("data/r2_cache_0919/*.json.gz"):
    w = int(p.split("/")[-1].split(".")[0])
    if not (lo <= w <= hi) or w in excl: continue
    d = json.load(gzip.open(p))
    op = (d.get("window_opened_wall_ts_by_environment") or {}).get(ENV)
    if not op: continue
    paid = [e for e in d.get("batch") or [] if e.get("env_name") == ENV and e.get("rewarded")]
    arr = lambda e: (e.get("precommit_arrival_ts") or e.get("arrival_ts") or 0) - op
    offs = sorted(arr(e) for e in paid)
    if len(offs) < 100: continue
    R[w] = dict(payes=sum(1 for e in paid if e.get("hotkey") == US), m56=offs[55],
                autres17=sum(1 for e in paid if e.get("hotkey") != US and arr(e) < 17),
                open=op)
# ── journal ──
sig, oo, g1, b1 = {}, {}, {}, collections.defaultdict(list); drop = {}; seen = collections.Counter(); cur = None
for lf in logs.split(","):
    for l in open(lf, errors="ignore"):
        if "flip_diag:" in l:
            m = re.search(r"window_n=(\d+).*signal_off=([\d.]+)", l)
            if m: sig[int(m.group(1))] = float(m.group(2))
        elif "bake_start:" in l:
            m = re.search(r"window=(\d+) bake_seq=\d+ prompts=\d+ open_off=([\d.]+)", l)
            if m:
                w = int(m.group(1)); seen[w] += 1
                cur = w if seen[w] == 1 else None
                if cur: oo[w] = float(m.group(2))
        elif "prêt à" in l and cur:
            m = re.search(r"groupe (\d+)/\d+ prêt à ([\d.]+)s .*prompt=(\d+)", l)
            if m:
                b1[cur].append(int(m.group(3)))
                if m.group(1) == "1": g1[cur] = float(m.group(2))
        elif "pre_bake[out_of_zone]" in l:
            m = re.search(r"prompt=(\d+)", l)
            if m: drop[int(m.group(1))] = 1
# ── soumissions : 1er tir par prompt, heure validateur via R2 ──
first = {}
for l in open(subs, errors="ignore"):
    try: s = json.loads(l)
    except Exception: continue
    w = s.get("window_n") or 0
    if w in R and s.get("t_precommit_sent"): first.setdefault((w, s.get("prompt_idx")), s)
fires = collections.defaultdict(list)
for (w, p), s in first.items():
    fires[w].append(s["t_precommit_sent"] - R[w]["open"])
rows = []
for w in sorted(R):
    f = sorted(fires[w])
    rows.append(dict(w=w, **{k: R[w][k] for k in ("payes", "m56", "autres17")},
                     bake=(oo.get(w, 0) - sig.get(w, 0)) if w in oo and w in sig else None,
                     g1=g1.get(w), ooz=sum(1 for p in b1[w][:10] if p in drop),
                     t17=sum(1 for x in f if x < 17), t18=sum(1 for x in f if x < 18),
                     tir3=f[2] if len(f) >= 3 else None))
print(f"{'fen':>6} {'payés':>5} | {'56e':>5} {'autres<17':>9} | {'sig→bake':>8} {'g1':>5} {'ooz b1':>6} {'tirs<17':>7} {'tir 3e':>6}")
for r in rows:
    fm = lambda v, d=1: "—" if v is None else f"{v:.{d}f}"
    print(f"{r['w']:>6} {r['payes']:>5} | {r['m56']:5.1f} {r['autres17']:>9} | {fm(r['bake'],2):>8} {fm(r['g1']):>5} {r['ooz']:>6} {r['t17']:>7} {fm(r['tir3']):>6}")

def r2(y, xs):
    """R² d'une régression linéaire (moindres carrés, numpy)."""
    import numpy as np
    X = np.column_stack([np.ones(len(y))] + [np.array(x, float) for x in xs])
    b, *_ = np.linalg.lstsq(X, np.array(y, float), rcond=None)
    res = np.array(y) - X @ b
    return 1 - res.var() / np.array(y, float).var(), b

ok = [r for r in rows if r["tir3"] is not None]
y = [r["payes"] for r in ok]
print(f"\n{len(ok)} fenêtres ; payés moy {st.mean(y):.2f}, écart-type {st.pstdev(y):.2f}, min {min(y)}, max {max(y)}")
for lab, cols in [("marché seul (autres payés <17 s)", ["autres17"]),
                  ("marché seul (56e place)", ["m56"]),
                  ("nous seuls (tirs <17 s)", ["t17"]),
                  ("nous seuls (hors zone 1er bake)", ["ooz"]),
                  ("nous seuls (heure du 3e tir)", ["tir3"]),
                  ("marché + nos tirs <17 s", ["autres17", "t17"]),
                  ("marché + tirs <17 + hors zone", ["autres17", "t17", "ooz"])]:
    R2, b = r2(y, [[r[c] for r in ok] for c in cols])
    print(f"  R² {R2:5.2f}  {lab:36s} coefs " + " ".join(f"{c}={v:+.2f}" for c, v in zip(cols, b[1:])))
# Ce qui fait nos tirs <17 s
print("\nce qui fait NOS tirs <17 s :")
for lab, cols in [("hors zone du 1er bake", ["ooz"]), ("groupe 1 prêt", ["g1"]),
                  ("signal→bake", ["bake"])]:
    sub = [r for r in ok if all(r[c] is not None for c in cols)]
    R2, b = r2([r["t17"] for r in sub], [[r[c] for r in sub] for c in cols])
    print(f"  R² {R2:5.2f}  {lab:24s} coef {b[1]:+.2f}")
