#!/usr/bin/env python3
"""Le mineur a-t-il RÉELLEMENT manqué des fenêtres ?

⚠️ LA VÉRIFICATION QUI MANQUAIT le 21/08 au matin. J'avais mesuré des trous
dans `bake terminé` et conclu à 48,8 min perdues par nuit sur les
téléchargements de checkpoint — analyse entièrement fausse. Les numéros de
fenêtre étaient CONSÉCUTIFS : 128 vues de 29904 à 30031, ZÉRO manquée.

Les longs trous de génération sont les phases TRAINING → PUBLISHING → READY du
VALIDATEUR, pendant lesquelles il n'ouvre aucune fenêtre. Il n'y a rien à miner,
donc rien à récupérer. Le téléchargement du checkpoint, lui, dure ~7 secondes.

RÈGLE : un trou dans NOTRE activité n'est une perte que si le validateur était
OUVERT pendant ce temps. Lancer ce script AVANT toute conclusion sur des
« fenêtres perdues ».
"""
import json
import subprocess

BOX, PORT = "root@38.255.28.21", "20098"
DISTANT = r'''
import json
ws = set()
for l in open("/workspace/windows_v4.jsonl"):
    try: r = json.loads(l)
    except Exception: continue
    if r.get("window_n"): ws.add(r["window_n"])
ws = sorted(ws)
trous = [(ws[i-1], ws[i]) for i in range(1, len(ws)) if ws[i] - ws[i-1] > 1]
print(json.dumps({"n": len(ws), "lo": ws[0], "hi": ws[-1], "trous": trous[:10]}))
'''
out = subprocess.run(
    ["ssh", "-o", "ConnectTimeout=20", "-o", "StrictHostKeyChecking=no",
     "-p", PORT, BOX, "/workspace/venv/bin/python -"],
    input=DISTANT, capture_output=True, text=True, timeout=90).stdout.strip()
d = json.loads(out.splitlines()[-1])
attendu = d["hi"] - d["lo"] + 1
manquees = attendu - d["n"]
print(f"  {d['n']} fenêtres vues, de {d['lo']} à {d['hi']} (attendues {attendu})")
if manquees == 0:
    print("  ✅ AUCUNE fenêtre manquée — les trous de génération sont les phases "
          "d'entraînement du validateur, pas une perte de notre fait")
else:
    print(f"  ⚠️ {manquees} fenêtres MANQUÉES — trous : {d['trous']}")
