#!/usr/bin/env python3
"""Teste si la TRAÎNE d'un groupe est exploitable — verdict : NON (21/08).

La traîne = `max(completion_lens) / moyenne`. L'effet sur le paiement est réel,
mais aucun des trois leviers envisagés ne le capture. Ce script rejoue les
trois mesures qui l'ont établi, pour qu'on n'ait pas à re-supposer.

  1. TEST APPARIÉ intra-fenêtre — la seule mesure honnête. Deux entrées de la
     MÊME fenêtre, à volume proche (±10 %), écart de ratio > 0,3. Sans
     appariement on mesure surtout le volume, qui est le vrai moteur du rang.
  2. PLAFOND D'ENVOI — le tri de la file d'envoi n'a d'effet que si le plafond
     de 32 soumissions/fenêtre mord. Il mord dans 1,4 % des fenêtres : trier ne
     change jamais qui part.
  3. TRI CANONIQUE PAR HACHAGE — `batch_selection._prompt_canonical_key` est
     `sha256(prompt_idx)` et ordonne les prompts dans un round drand. Semblait
     un levier gratuit ; mesuré nul.

Usage : python3 scripts/test_traine_leviers.py
"""
from __future__ import annotations

import collections
import hashlib
import json
import statistics as st
import sys
from pathlib import Path

DATA = Path("/root/subnet81/data")
VOL_TOL = 0.10      # volume apparié à +-10 %
RATIO_MIN = 0.30    # ecart de ratio minimal pour que la paire soit informative
ARR_TOL = 1.5       # un round drand fait 3 s : +-1,5 s reste dans le meme round


def _charger() -> dict[int, list[dict]]:
    """Joint verdicts -> submits -> samples, groupé par fenêtre."""
    V = {}
    for line in open(DATA / "verdicts_v4.jsonl", errors="replace"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        # `rewarded` est None tant que la fenêtre n'est pas scellée : les
        # inclure ferait compter des non-décidés comme des non-payés.
        if r.get("merkle_root") and r.get("rewarded") is not None:
            V[r["merkle_root"]] = r
    G = {}
    for line in open(DATA / "samples_v4.jsonl", errors="replace"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("window_n") is not None:
            G[(r["window_n"], r.get("prompt_idx"))] = r
    par_fenetre = collections.defaultdict(list)
    for line in open(DATA / "submits_v4.jsonl", errors="replace"):
        try:
            s = json.loads(line)
        except Exception:
            continue
        v = V.get(s.get("merkle_root"))
        if not v or not isinstance(v.get("canonical_rank"), int):
            continue
        g = G.get((s.get("window_n"), s.get("prompt_idx")))
        if not g:
            continue
        cl = g.get("completion_lens")
        # min < 32 : le gate CHALLENGE_K les rend ineligibles, les inclure
        # melangerait deux populations.
        if not cl or len(cl) < 8 or min(cl) < 32 or s.get("flip_offset_s") is None:
            continue
        par_fenetre[s["window_n"]].append({
            "rank": v["canonical_rank"], "paid": bool(v["rewarded"]),
            "tot": sum(cl), "ratio": max(cl) / (sum(cl) / len(cl)),
            "off": s["flip_offset_s"], "pidx": s.get("prompt_idx"),
        })
    return par_fenetre


def _paires(par_fenetre, arr_tol=None):
    for entrees in par_fenetre.values():
        for i in range(len(entrees)):
            for j in range(i + 1, len(entrees)):
                a, b = entrees[i], entrees[j]
                if abs(a["tot"] - b["tot"]) / max(a["tot"], b["tot"]) > VOL_TOL:
                    continue
                u, t = (a, b) if a["ratio"] < b["ratio"] else (b, a)
                if t["ratio"] - u["ratio"] < RATIO_MIN:
                    continue
                if arr_tol is not None and abs(t["off"] - u["off"]) > arr_tol:
                    continue
                yield u, t


def _bilan(titre, paires):
    paires = list(paires)
    if not paires:
        print(f"  {titre} : aucune paire")
        return
    gagne = sum(1 for u, t in paires if u["rank"] < t["rank"])
    tranchees = sum(1 for u, t in paires if u["rank"] != t["rank"])
    n = len(paires)
    marge = 100 * 1.96 * (0.25 / max(tranchees, 1)) ** 0.5
    print(f"  {titre}")
    print(f"    paires {n:3d} | uniforme devant {100*gagne/max(tranchees,1):.0f}% "
          f"(hasard 50 %, +-{marge:.0f}) | ecart median "
          f"{st.median(t['rank'] - u['rank'] for u, t in paires):+.0f} rangs")
    print(f"    payees : uniformes {100*sum(u['paid'] for u,_ in paires)/n:.0f}% "
          f"| a traine {100*sum(t['paid'] for _,t in paires)/n:.0f}%")


def main() -> int:
    pf = _charger()
    n = sum(len(v) for v in pf.values())
    print(f"entrees jointes avec verdict FINAL : {n} sur {len(pf)} fenetres\n")
    if n < 100:
        print("corpus trop maigre")
        return 1

    print("1) L'EFFET EST-IL REEL ? (test apparie intra-fenetre)")
    _bilan("volume controle (+-10 %)", _paires(pf))
    _bilan("volume ET arrivee controles (+-1,5 s)", _paires(pf, ARR_TOL))

    print("\n2) LE TRI DE LA FILE D'ENVOI PEUT-IL AGIR ?")
    par = collections.Counter()
    for line in open(DATA / "submits_v4.jsonl", errors="replace"):
        try:
            w = json.loads(line).get("window_n")
        except Exception:
            continue
        if w:
            par[w] += 1
    v = sorted(par.values())
    mord = sum(1 for x in v if x >= 32)
    print(f"  envois/fenetre : med {v[len(v)//2]} | p90 {v[int(len(v)*.9)]} "
          f"| max {v[-1]}")
    print(f"  fenetres ou le plafond de 32 mord : {mord}/{len(v)} "
          f"({100*mord/len(v):.1f} %) -> trier la file ne change RIEN")

    print("\n3) LE TRI CANONIQUE sha256(prompt_idx) EST-IL EXPLOITABLE ?")
    rows = [e for es in pf.values() for e in es]
    for e in rows:
        e["h"] = hashlib.sha256(int(e["pidx"]).to_bytes(8, "big")).digest()[0]
    rows.sort(key=lambda e: e["h"])
    q = len(rows) // 4
    for i in range(4):
        g = rows[i * q:(i + 1) * q]
        print(f"  hash Q{i+1} n={len(g):3d} paye {100*sum(x['paid'] for x in g)/len(g):5.1f}% "
              f"| rang med {st.median(x['rank'] for x in g):.0f}")
    print("  -> plat : le hachage n'est pas un levier")
    return 0


if __name__ == "__main__":
    sys.exit(main())
