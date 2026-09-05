#!/usr/bin/env python3
"""A/B du retrait du gate anti-rollout-court — mesure sur 30 fenêtres.

MÉTRIQUE PRINCIPALE : **entrées admises par fenêtre**. C'est ce que le gate
change directement — il rend ~1,5 entrée par fenêtre (17,3 % de la production
qu'il jetait). Le rang, lui, n'est pas censé bouger : testé le 21/08, les
fenêtres où le gate jette le plus ne sont ni plus lentes ni moins bien classées
(corrélations −0,09).

⛔ NE PAS trancher sur les payées/fenêtre : écart-type ≈ espérance, il faudrait
des centaines de fenêtres. Et **compter PAR HEURE** en secondaire, car le
validateur a raccourci son cycle le 21/08 (+30 % de fenêtres/heure) : tout ce
qui se compte « par fenêtre » baisse mécaniquement.

SIGNAL D'ABANDON : le moindre `logprob_mismatch`. Avant PR #188 la vérification
échouait par construction sous 32 tokens ; si elle échoue encore chez nous, le
rejet retombe au stage `logprob` **qui est un stage à dette** — 2 dans une
fenêtre et le reste est refusé.

MATURITÉ STRICTE : une fenêtre ne compte que si TOUTES ses admises ont un
verdict décidé. Les verdicts mettent jusqu'à 28 min ; compter les indécis comme
non payés fabrique une baisse. Verdicts dédupliqués par `merkle_root` (le dump
ré-écrit la même ligne à chaque interrogation).

Usage :
    python3 scripts/ab_gate.py --avant 30175      # référence, fenêtres < 30175
    python3 scripts/ab_gate.py --apres 30175      # bras B, fenêtres >= 30175
    python3 scripts/ab_gate.py --comparer 30175   # les deux + verdict
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import statistics as st
import subprocess
import sys

BOX, PORT = "root@38.255.28.21", "20098"

DISTANT = r'''
import json, collections, statistics as st, sys
COUPE = int(sys.argv[1])
G = {}
for l in open("/workspace/samples_v4.jsonl", errors="replace"):
    try: r = json.loads(l)
    except Exception: continue
    cl = r.get("completion_lens")
    if cl and r.get("window_n") is not None:
        G[(r["window_n"], r.get("prompt_idx"))] = (sum(cl), min(cl))
V = collections.defaultdict(list)
for l in open("/workspace/verdicts_v4.jsonl", errors="replace"):
    try: r = json.loads(l)
    except Exception: continue
    if r.get("merkle_root"): V[r["merkle_root"]].append(r)
DEBT = {"code_grader_crash","grail","termination","force_span","logprob",
        "distribution","boxed_answer","token_authenticity",
        "all_token_authenticity","code_semantic_auth","forced_seed"}
S = collections.defaultdict(list)
for l in open("/workspace/submits_v4.jsonl", errors="replace"):
    try: r = json.loads(l)
    except Exception: continue
    if r.get("window_n") and r.get("ts"): S[r["window_n"]].append(r)
out = []
for w, es in S.items():
    acc = [e for e in es if e.get("accepted")]
    if not acc: continue
    dec = pay = courts = courts_ok = lpm = dette = 0
    rk = []
    for e in acc:
        rs = V.get(e.get("merkle_root"), [])
        court = G.get((w, e.get("prompt_idx")), (0, 999))[1] < 32
        if court: courts += 1
        if any(x.get("rewarded") is not None for x in rs):
            dec += 1
            if any(x.get("rewarded") for x in rs):
                pay += 1
                if court: courts_ok += 1
        for x in rs:
            if isinstance(x.get("canonical_rank"), int): rk.append(x["canonical_rank"])
    # rejets a dette + logprob, sur TOUS les envois (pas que les admis)
    for e in es:
        for x in V.get(e.get("merkle_root"), []):
            rr = x.get("reject_reason") or x.get("reason") or ""
            if rr == "logprob_mismatch": lpm += 1
            if (x.get("reject_stage") or "") in DEBT: dette += 1
    if dec < len(acc): continue          # MATURITE STRICTE
    offs = [e["flip_offset_s"] for e in acc if e.get("flip_offset_s") is not None]
    vols = [G.get((w, e.get("prompt_idx")), (0,0))[0] for e in acc]
    vols = [v for v in vols if v]
    out.append({"w": w, "t": min(e["ts"] for e in es), "acc": len(acc),
                "env": len(es), "pay": pay, "courts": courts,
                "courts_ok": courts_ok, "lpm": lpm, "dette": dette,
                "off": min(offs) if offs else None,
                "vol": st.mean(vols) if vols else None,
                "best": min(rk) if rk else None})
print(json.dumps({"coupe": COUPE, "fenetres": out}))
'''


def charger(coupe: int) -> list[dict]:
    out = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=20", "-o", "StrictHostKeyChecking=no",
         "-p", PORT, BOX, f"/workspace/venv/bin/python - {coupe}"],
        input=DISTANT, capture_output=True, text=True, timeout=300).stdout
    return json.loads(out.strip().splitlines()[-1])["fenetres"]


def bilan(g: list[dict], nom: str) -> dict | None:
    if len(g) < 5:
        print(f"  {nom:10s} : {len(g)} fenêtres mûres — insuffisant")
        return None
    dur = (max(x["t"] for x in g) - min(x["t"] for x in g)) / 3600 or 0.01
    r = {"n": len(g), "acc": st.mean(x["acc"] for x in g),
         "s_acc": st.pstdev(x["acc"] for x in g),
         "pay_h": sum(x["pay"] for x in g) / dur,
         "pay_f": st.mean(x["pay"] for x in g),
         "courts": sum(x["courts"] for x in g),
         "courts_ok": sum(x["courts_ok"] for x in g),
         "lpm": sum(x["lpm"] for x in g),
         "dette2": sum(1 for x in g if x["dette"] >= 2),
         "off": st.median(x["off"] for x in g if x["off"] is not None),
         "best": st.median(x["best"] for x in g if x["best"] is not None)}
    print(f"  {nom:10s} : n={r['n']:3d} | admises/fen {r['acc']:.2f} | "
          f"{r['pay_h']:5.1f} payées/h ({r['pay_f']:.2f}/fen) | "
          f"1re {r['off']:+.1f}s | meilleur rang {r['best']:.0f}")
    print(f"             entrées courtes admises {r['courts']} "
          f"(dont {r['courts_ok']} payées) | logprob_mismatch {r['lpm']} | "
          f"fen à dette≥2 : {r['dette2']}")
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparer", type=int, metavar="FENETRE")
    ap.add_argument("--depuis", type=int, metavar="FENETRE",
                    help="1re fenetre du bras B hors RODAGE. Le moteur met "
                         "~15 min a retrouver son regime apres un restart : "
                         "juger avant, c'est le piege qui a fait retirer puis "
                         "remettre la cascade et juger sprint 8 bon puis "
                         "mauvais. Sans cette option, tout le bras B compte.")
    ap.add_argument("--avant", type=int)
    ap.add_argument("--apres", type=int)
    a = ap.parse_args()
    coupe = a.comparer or a.avant or a.apres
    if coupe is None:
        print("préciser --comparer / --avant / --apres <numéro de fenêtre>")
        return 1
    fen = charger(coupe)
    av = sorted([x for x in fen if x["w"] < coupe], key=lambda x: x["w"])[-30:]
    seuil_b = a.depuis or coupe
    ap_ = sorted([x for x in fen if x["w"] >= seuil_b], key=lambda x: x["w"])
    if a.depuis:
        ecartees = sum(1 for x in fen if coupe <= x["w"] < a.depuis)
        if ecartees:
            print(f"  ({ecartees} fenêtre(s) de rodage écartée(s), "
                  f"le bras B commence à {a.depuis})\n")

    if a.avant:
        bilan(av, "référence")
        return 0
    if a.apres:
        bilan(ap_, "bras B")
        return 0

    print(f"── A/B du retrait du gate, coupure à la fenêtre {coupe} ──\n")
    A = bilan(av, "référence")
    B = bilan(ap_, "bras B")
    if not A or not B:
        print("\n  → pas encore assez de fenêtres mûres dans le bras B.")
        return 0
    if B["lpm"] > 0:
        print(f"\n  🛑 ABANDON : {B['lpm']} logprob_mismatch. "
              "Remettre RELIQUARY_MIN_ROLLOUT_LEN=32 et redémarrer.")
        return 0
    se = math.sqrt(A["s_acc"]**2 / A["n"] + B["s_acc"]**2 / B["n"])
    d = B["acc"] - A["acc"]
    print(f"\n  admises/fenêtre : {d:+.2f} (erreur-type {se:.2f}, "
          f"IC95 [{d-1.96*se:+.2f} ; {d+1.96*se:+.2f}])")
    if abs(d) < 1.96 * se:
        besoin = math.ceil(2 * ((1.96 + 0.84) * max(A["s_acc"], B["s_acc"])
                                / max(abs(d), 0.1))**2)
        print(f"  → NON CONCLUANT. Il faudrait ~{besoin} fenêtres par bras "
              "pour trancher un écart de cette taille.")
    else:
        print("  → écart SIGNIFICATIF au seuil de 5 %.")
    print(f"  entrées courtes admises dans le bras B : {B['courts']} "
          f"(attendu ~1,5/fenêtre, soit ~{1.5*B['n']:.0f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
