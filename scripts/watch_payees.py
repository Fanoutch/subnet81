#!/usr/bin/env python3
"""Payées par fenêtre — comparaison à la référence d'avant la coupure du 20/08.

Lit les données SUR LA BOX (le rapatriement local a une demi-heure de retard).

Trois précautions, chacune payée d'une erreur ce soir :

1. Ne compter QUE les fenêtres TRANCHÉES. Un verdict arrive en deux temps :
   d'abord `reason: accepted` avec rewarded=None (reçu, seal pas encore
   décidé), puis rewarded=True/False. Compter un rewarded=None comme « non
   payée » sous-estime systématiquement — c'est ce qui a produit des fenêtres
   annoncées à zéro qui payaient 2 ou 3 fois un quart d'heure plus tard, et
   trois recommandations contradictoires de ma part.

2. Isoler les fenêtres tuées par LEUR correcteur. Deux `worker_dropped` au
   stade `code_grader` épuisent le quota d'échecs coûteux du validateur
   (2 par hotkey et par fenêtre) et font refuser TOUT le reste — y compris des
   entrées classées dans le top-16. Vu sur la 29913 : rangs 8 et 11 non
   sélectionnés. Ces fenêtres ne mesurent pas notre performance.

3. Afficher l'ÂGE DU MOTEUR. La règle d'or du projet : ne jamais juger une
   configuration sans savoir depuis combien de temps le moteur tourne.
"""
import json
import statistics as st
import subprocess
import sys
import time
from collections import defaultdict

BOX = "root@38.255.28.21"
PORT = "20098"

# Référence mesurée le 20/08 entre 17h51 et 20h30, sur 14 fenêtres TRANCHÉES,
# en egress direct, avant le déploiement des 2 correctifs de latence.
# Les fichiers bruts qui l'ont produite ont été détruits à 21h45 par le
# rapatriement (rsync --append-verify sur une box réinitialisée) : on la garde
# donc en dur, c'est le seul point de comparaison solide qui subsiste.
REF = {"acc": 3.2, "payees": 1.14, "part_payantes": 61, "prem": 8.5}

DISTANT = r'''
import json, time
from collections import defaultdict
sub = [json.loads(l) for l in open("/workspace/submits_v4.jsonl")]
ver = {}
for l in open("/workspace/verdicts_v4.jsonl"):
    v = json.loads(l)
    if v.get("merkle_root"):
        ver[v["merkle_root"]] = v
w = defaultdict(lambda: {"acc":0,"dec":0,"paid":0,"off":[],"drop":0})
for u in sub:
    n, r = u.get("window_n"), u.get("reason")
    if n is None or r != "submitted":
        continue
    w[n]["acc"] += 1
    if u.get("flip_offset_s") is not None:
        w[n]["off"].append(u["flip_offset_s"])
    v = ver.get(u.get("merkle_root"))
    if v is not None and v.get("rewarded") is not None:
        w[n]["dec"] += 1
        if v["rewarded"]:
            w[n]["paid"] += 1
for v in ver.values():
    if v.get("reject_reason") == "worker_dropped":
        w[v.get("window_n")]["drop"] += 1
t0 = None
for l in open("/workspace/miner.log", errors="replace"):
    try:
        t0 = time.mktime(time.strptime(l[:19], "%Y-%m-%d %H:%M:%S")); break
    except Exception:
        continue
print(json.dumps({"w": {str(k): v for k, v in w.items()},
                  "age_min": (time.time()-t0)/60 if t0 else None}))
'''


def main():
    out = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=20", "-o", "StrictHostKeyChecking=no",
         "-p", PORT, BOX, "/workspace/venv/bin/python - "],
        input=DISTANT, capture_output=True, text=True, timeout=90).stdout
    try:
        d = json.loads(out.strip().splitlines()[-1])
    except Exception:
        print("payées: box injoignable ou données illisibles")
        return
    wins = {int(k): v for k, v in d["w"].items()}
    age = d.get("age_min")

    # tranchée = tous les envois acceptés ont un verdict DÉCIDÉ
    mures = {k: v for k, v in wins.items() if v["acc"] and v["dec"] == v["acc"]}
    # fenêtres où LEUR correcteur a épuisé notre quota d'échecs coûteux
    leur_faute = {k for k, v in mures.items() if v["drop"] >= 2 and v["paid"] == 0}
    net = {k: v for k, v in mures.items() if k not in leur_faute}

    if not net:
        print(f"payées: 0 fenêtre tranchée exploitable "
              f"({len(wins)} avec envois, moteur {age:.0f} min)"
              if age else "payées: aucune fenêtre tranchée")
        return

    n = len(net)
    acc = sum(v["acc"] for v in net.values()) / n
    pay = sum(v["paid"] for v in net.values()) / n
    part = 100 * sum(1 for v in net.values() if v["paid"]) / n
    prem = [min(v["off"]) for v in net.values() if v["off"]]
    p = st.median(prem) if prem else 0

    def d_(x, r):
        return f"{x:+.0f}%" if r else ""

    fleche = "▲" if pay >= REF["payees"] else "▼"
    print(f"payées {fleche} {pay:.2f}/fen (réf {REF['payees']:.2f}) | "
          f"{acc:.1f} acc/fen (réf {REF['acc']:.1f}) | "
          f"{part:.0f}% payantes (réf {REF['part_payantes']}%) | "
          f"1re +{p:.1f}s (réf +{REF['prem']}s) | "
          f"n={n} tranchées, moteur {age:.0f} min"
          + (f" | {len(leur_faute)} fen exclues (leur correcteur)"
             if leur_faute else ""))
    if n < 10:
        print(f"   ⚠️ {n} fenêtres seulement — NE PAS conclure avant ~15")


if __name__ == "__main__":
    main()
