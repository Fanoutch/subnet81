#!/usr/bin/env python3
"""Rapport de comportement du mineur v4 — lit les 4 writers + le log.

Usage (box) : python3 ops/rapport_v4.py [--fenetres N]  (défaut : tout)
Sert aussi de tick au moniteur (ops/monitor_v4.sh l'appelle en boucle).
"""
import json
import re
import sys
import time
from collections import Counter, defaultdict


def load(p):
    try:
        return [json.loads(l) for l in open(p)]
    except FileNotFoundError:
        return []


def main():
    S = load("/workspace/samples_v4.jsonl")
    W = load("/workspace/windows_v4.jsonl")
    U = load("/workspace/submits_v4.jsonl")
    V = load("/workspace/verdicts_v4.jsonl")
    now = time.time()

    live = [r for r in S if r.get("source") != "probe_ck0" and r.get("window_n")]
    h1 = [r for r in live if now - r.get("ts", 0) < 3600]
    print("═══ PRODUCTION (groupes gradés, hors probe) ═══")
    print(f"  total: {len(live)} | dernière heure: {len(h1)} ({len(h1)/60:.1f}/min)")
    if live:
        ks = Counter(r["k"] for r in live)
        print(f"  distribution k: {dict(sorted(ks.items()))}")
        pay = sum(1 for r in live if r.get("sigma", 0) >= 0.24)
        ved = sum(1 for r in live if r.get("score", 0) >= 0.23)
        tr = sum(1 for r in live if r.get("n_truncated", 0) > 0)
        print(f"  payables σ≥0.24: {pay/len(live):.0%} | vedettes score≥0.23: "
              f"{ved} ({ved/len(live):.0%}) | tronqués: {tr}")
        src = Counter(str(r.get("source")) for r in live)
        print(f"  sources des picks: {dict(src)}")
        lens = sorted(r["completion_lens"][len(r["completion_lens"])//2]
                      for r in live if r.get("completion_lens"))
        if lens:
            print(f"  longueur médiane: {lens[len(lens)//2]} tok | "
                  f"cap-hits: {sum(r.get('n_truncated',0) for r in live)}")

    # ═══ A/B PRIOR v5 : picks classés (prior) vs explore (contrôle randomisé,
    # mêmes fenêtres) — LE juge du gain réel. Verdict stable à ~40-50 fenêtres.
    ranked = [r for r in live if r.get("source") in ("ranked", "memo")]
    explo = [r for r in live if r.get("source") == "explore"]
    if ranked and explo:
        def vr(rows):
            return sum(1 for r in rows if r.get("score", 0) >= 0.23) / len(rows)
        print("═══ A/B PRIOR (ranked+memo vs explore, mêmes fenêtres) ═══")
        print(f"  picks prior/mémo: {len(ranked)} → vedettes {vr(ranked):.1%}")
        print(f"  picks explore   : {len(explo)} → vedettes {vr(explo):.1%}  "
              f"(gain ×{vr(ranked)/max(vr(explo),1e-9):.2f})")

    print("═══ FENÊTRES (writer B2) ═══")
    if W:
        cd = [w.get("cooldown_len", 0) for w in W]
        mh = [w.get("memo_hits") or 0 for w in W]
        print(f"  fenêtres tracées: {len(W)} | cooldown: {cd[0]}→{cd[-1]} | "
              f"memo_hits/tranche: méd {sorted(mh)[len(mh)//2]} max {max(mh)}")

    print("═══ SOUMISSIONS (writer B5) ═══")
    if U:
        acc = sum(1 for u in U if u.get("accepted"))
        rs = Counter(u.get("reason") for u in U)
        print(f"  envoyées: {len(U)} | acceptées à l'admission: {acc} "
              f"({acc/len(U):.0%}) | raisons: {dict(rs)}")
        perw = Counter(u["window_n"] for u in U)
        if perw:
            vals = sorted(perw.values())
            print(f"  par fenêtre: méd {vals[len(vals)//2]} max {max(vals)} "
                  f"(quota 32) sur {len(perw)} fenêtres")
        msrc = Counter(str(u.get("source")) for u in U)
        print(f"  sources soumises: {dict(msrc)}")

    print("═══ VERDICTS (writer B3 — la vérité du validateur) ═══")
    if V:
        rs = Counter(v.get("reason") for v in V)
        enr = [v for v in V if v.get("canonical_rank") is not None]
        rew = [v for v in enr if v.get("rewarded")]
        sel = [v for v in enr if v.get("selected_for_batch")]
        print(f"  verdicts: {len(V)} | raisons: {dict(rs)}")
        print(f"  enrichis (post-seal): {len(enr)} | selected: {len(sel)} | "
              f"REWARDED: {len(rew)} 💰" if enr else "  enrichis: 0 (seal pas encore publié)")
        if enr:
            rk = sorted(v["canonical_rank"] for v in enr)
            print(f"  rangs: {rk[:12]}")
    else:
        print("  (aucun)")

    # bilan par fenêtre du log (WindowTally)
    try:
        tail = open("/workspace/miner.log", errors="replace").read()[-400000:]
        tally = re.findall(r"réalisé fenêtre \d+.*", tail)
        if tally:
            print("═══ WINDOWTALLY (log, dernières fenêtres) ═══")
            for t in tally[-3:]:
                print("  " + t[:120])
        drops = Counter(re.findall(r'_record_drop.*reason="(\w+)"', tail) or
                        re.findall(r"pre_bake\[(\w+)\]", tail))
        if drops:
            print(f"  drops locaux (tail): {dict(drops)}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
