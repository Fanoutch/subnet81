#!/usr/bin/env python3
"""Rejoue le classement des prompts AVEC et SANS le bonus de volume.

QUESTION TRANCHÉE : μ améliore-t-il le volume des PREMIÈRES entrées — les
seules qui comptent, puisque 66 % de notre production part après la fermeture
du batch et n'est jamais admise ?

MÉTHODE — contrefactuel intra-fenêtre. Pour chaque fenêtre on reprend les
groupes RÉELLEMENT bakés (donc de vraies longueurs mesurées), on recalcule
leur score `prior − λ·risque + μ·volume`, on re-classe, et on regarde le
volume des k premiers du nouveau classement.

⚠️ CE QUE ÇA NE MESURE PAS. Le vrai tri s'applique à ~300 candidats de la
tranche, dont on n'observe QUE les 8 retenus : les prompts non bakés n'ont pas
de longueur mesurée, donc ils sont hors de portée de toute simulation honnête.
Ce test est donc une BORNE BASSE — il mesure le pouvoir de tri du modèle sur
un échantillon déjà filtré par le prior, ce qui est le cas le plus défavorable.

⚠️ Les groupes `source != "ranked"` (mémo, exploration) contournent le tri :
ils sont exclus du re-classement mais gardés dans le décompte, comme en vol.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, "/root/subnet81/.worktrees/miner-priv-port-v4-dapo")
from reliquary.miner import prompt_predictor as pp  # noqa: E402

DATA = Path("/root/subnet81/data")


def charger(n_max: int) -> dict:
    par = collections.defaultdict(list)
    for line in open(DATA / "samples_v4.jsonl", errors="replace"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        cl = r.get("completion_lens")
        if not r.get("window_n") or not cl or r.get("score") is None:
            continue
        # min < 32 : le gate CHALLENGE_K les jette, ils ne concourent pas
        if min(cl) < 32:
            continue
        par[r["window_n"]].append(r)
    ws = sorted(par)[-n_max:]
    return {w: par[w] for w in ws}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modele", default=str(DATA / "volume_v1.json"))
    ap.add_argument("--fenetres", type=int, default=200)
    ap.add_argument("--top", type=int, default=3,
                    help="entrées qui arrivent avant la fermeture du batch")
    a = ap.parse_args()

    mod = json.load(open(a.modele))
    par = charger(a.fenetres)
    par = {w: v for w, v in par.items() if len(v) >= a.top + 2}
    print(f"fenêtres exploitables : {len(par)} "
          f"(>= {a.top + 2} groupes bakés, rollout min >= 32 tok)")
    if len(par) < 30:
        print("échantillon insuffisant")
        return 1

    # score de volume, une fois par prompt
    vs = {}
    for v in par.values():
        for r in v:
            if r["prompt_idx"] not in vs:
                vs[r["prompt_idx"]] = pp.volume_score(mod, r["prompt"])

    print(f"\n  {'μ':>6} | {'vol. du top-3':>13} | {'meilleur du top-3':>17} "
          f"| {'part >=6000 tok':>15} | {'ordre changé':>12}")
    base = None
    for mu in (0.0, 0.005, 0.01, 0.02, 0.05, 0.10):
        tops, bests, gros, bouge, n = [], [], 0, 0, 0
        for w, v in par.items():
            classables = [r for r in v if r.get("source") == "ranked"]
            autres = [r for r in v if r.get("source") != "ranked"]
            if len(classables) < 2:
                continue
            av = [r["prompt_idx"] for r in
                  sorted(classables, key=lambda r: -r["score"])]
            ordre = sorted(classables,
                           key=lambda r: -(r["score"] + mu * vs[r["prompt_idx"]]))
            ap_ = [r["prompt_idx"] for r in ordre]
            if av[:a.top] != ap_[:a.top]:
                bouge += 1
            top = (ordre + autres)[:a.top]
            vols = [sum(r["completion_lens"]) for r in top]
            tops.append(st.mean(vols)); bests.append(max(vols))
            gros += sum(1 for x in vols if x >= 6000); n += len(vols)
        m = st.mean(tops)
        if base is None:
            base = m
        print(f"  {mu:6.3f} | {m:8.0f} tok  | {st.mean(bests):12.0f} tok  "
              f"| {100*gros/n:13.0f} % | {100*bouge/len(par):10.0f} %"
              + ("   <- actuel" if mu == 0 else f"   {100*(m/base-1):+.1f} %"))

    print("\n  Le tri ne peut agir que s'il CHANGE l'ordre : la dernière")
    print("  colonne borne tout gain possible. Un μ qui ne bouge rien ne peut")
    print("  rien rapporter, et un μ qui bouge tout a noyé le prior.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
