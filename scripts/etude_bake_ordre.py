#!/usr/bin/env python3
"""Pourquoi le gros groupe arrive trop tard — et où sont les 7,3 s d'arrivée.

RÉSULTAT (21/08, 1 097 bakes + 620 entrées admises) : dans un lot de 8 prompts
(128 séquences vLLM), les groupes sortent par ordre CROISSANT de longueur. Le
plus volumineux sort en position 6-8 dans 87 % des cas, prêt à 6,8 s contre
2,8 s pour le premier. Il arrive donc un round drand plus tard et paie un péage
sur son bucket `tokens // (rounds × 50)`.

⚠️ CE PÉAGE N'EST PAS UNE ANNULATION. Ma première lecture, fondée sur des
arrivées ESTIMÉES depuis la position dans le bake, concluait à un bucket
identique aux deux extrémités. Vérifié sur les entrées RÉELLEMENT admises :
+116 % de volume donnent +56 % de bucket. Le round en reprend la moitié.
Toujours vérifier sur les arrivées mesurées, jamais sur des arrivées déduites.

LEVIER QUI EN DÉCOULE : nos meilleures entrées sont à bucket 28, juste sous le
seuil de 29 (au-delà, 81 % des fenêtres paient). Un round gagné les porte à 40.
Leur lenteur vient du partage du GPU — donc `RELIQUARY_BAKE_BATCH_SIZE`.
Magnitude NON mesurée : A/B sur la box requis.

Deux modes :
    --box     lit /workspace/miner.log SUR LA BOX (ordre de sortie des bakes)
    (défaut)  lit data/*.jsonl en local (volume → arrivée → bucket)
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics as st
import sys
from pathlib import Path

DATA = Path("/root/subnet81/data")

DISTANT = r'''
import re, json, collections
GN = re.compile(r"groupe (\d+)/(\d+) pr.t . ([0-9.]+)s.*prompt=(\d+)")
FIN = re.compile(r"bake termin")
cyc = []; cur = None
for l in open("/workspace/miner.log", errors="replace"):
    g = GN.search(l)
    if g:
        if cur is None: cur = []
        cur.append((int(g.group(1)), float(g.group(3)), int(g.group(4))))
        continue
    if FIN.search(l) and cur:
        cyc.append(cur); cur = None
vol = {}
for l in open("/workspace/samples_v4.jsonl", errors="replace"):
    try: r = json.loads(l)
    except Exception: continue
    cl = r.get("completion_lens")
    if cl and r.get("prompt_idx") is not None:
        vol.setdefault(r["prompt_idx"], []).append(sum(cl))
out = []
for c in cyc:
    pts = [(i, t, sum(vol[p]) / len(vol[p])) for i, t, p in c if p in vol]
    if len(pts) >= 5: out.append(pts)
print(json.dumps({"n": len(out),
                  "data": [[[i, t, round(v)] for i, t, v in p] for p in out[-400:]]}))
'''


def spearman(xs, ys) -> float:
    def rk(v):
        s = sorted(range(len(v)), key=lambda i: v[i])
        o = [0] * len(v)
        for i, j in enumerate(s):
            o[j] = i
        return o
    a, c = rk(xs), rk(ys)
    n = len(a)
    return (1 - 6 * sum((a[i] - c[i]) ** 2 for i in range(n)) / (n * (n * n - 1))
            if n > 2 else 0.0)


def local() -> int:
    """Volume → arrivée → bucket, sur les entrées RÉELLEMENT admises."""
    G = {}
    for line in open(DATA / "samples_v4.jsonl", errors="replace"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("window_n") is not None:
            G[(r["window_n"], r.get("prompt_idx"))] = r
    rows = []
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
        # rounds drand ecoules : proxy verifie a 84 % contre arrival_drand_round
        rd = max(1, int(off // 3) + 1)
        rows.append({"tot": tot, "off": off, "rd": rd, "bucket": tot // (rd * 50)})
    if len(rows) < 100:
        print("corpus insuffisant")
        return 1
    rows.sort(key=lambda r: r["tot"])
    q = len(rows) // 5
    print(f"── volume → arrivée → bucket, sur {len(rows)} entrées ADMISES ──\n")
    print("  quintile | volume moy | arrivée méd | round | BUCKET")
    for i in range(5):
        g = rows[i * q:(i + 1) * q] if i < 4 else rows[4 * q:]
        print("     Q%d    |  %5.0f tok |   %+5.1f s   |  %.1f  |  %3.0f" % (
            i + 1, st.mean(x["tot"] for x in g), st.median(x["off"] for x in g),
            st.median(x["rd"] for x in g), st.median(x["bucket"] for x in g)))
    v = [r["tot"] for r in rows]
    print("\n  Spearman(volume, arrivée) = %+.3f  (plus gros = plus tard)"
          % spearman(v, [r["off"] for r in rows]))
    print("  Spearman(volume, bucket)  = %+.3f  (le volume paie, net du péage)"
          % spearman(v, [r["bucket"] for r in rows]))
    return 0


def box(port: str, host: str) -> int:
    import subprocess
    out = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=20", "-o", "StrictHostKeyChecking=no",
         "-p", port, host, "/workspace/venv/bin/python -"],
        input=DISTANT, capture_output=True, text=True, timeout=300).stdout
    try:
        d = json.loads(out.strip().splitlines()[-1])
    except Exception:
        print("box injoignable ou journal illisible")
        return 1
    par, tps, rangs = collections.defaultdict(list), collections.defaultdict(list), []
    for b in d["data"]:
        for i, t, v in b:
            par[i].append(v); tps[i].append(t)
        rangs.append(max(b, key=lambda x: x[2])[0])
    print(f"── ordre de sortie dans le bake ({len(d['data'])} bakes) ──\n")
    print("  position | groupes | volume moyen | prêt à")
    for i in sorted(par):
        if len(par[i]) < 30:
            continue
        print("     %-2d    |  %4d   |   %5.0f tok  |  %.1f s" % (
            i, len(par[i]), st.mean(par[i]), st.median(tps[i])))
    c = collections.Counter(rangs)
    tot = sum(c.values())
    tard = sum(n for i, n in c.items() if i >= 6)
    print("\n  le groupe le PLUS VOLUMINEUX sort en position 6-8 dans %.0f %% des bakes"
          % (100 * tard / tot))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", action="store_true", help="lire le journal sur la box")
    ap.add_argument("--host", default="root@38.255.28.21")
    ap.add_argument("--port", default="20098")
    a = ap.parse_args()
    return box(a.port, a.host) if a.box else local()


if __name__ == "__main__":
    sys.exit(main())
