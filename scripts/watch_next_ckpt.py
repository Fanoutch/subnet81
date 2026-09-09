#!/usr/bin/env python3
"""Guette le PROCHAIN rechargement de checkpoint et rapporte son coût réel.

Trois choses à vérifier, et une seule est déjà corrigée :
  1. la garde VRAM a-t-elle TOLÉRÉ le pic (correctif déployé le 21/08 07:10)
     au lieu de redémarrer le mineur ? 3 rechargements sur 7 déclenchaient un
     restart inutile, coûtant 2-3 fenêtres EN PLUS des 8 min ;
  2. combien de temps a duré le TÉLÉCHARGEMENT (~7 min attendues tant que
     HF_XET_HIGH_PERFORMANCE n'est pas déployé ; ~10 s ensuite) ;
  3. combien de temps a duré le CHARGEMENT (~1 min, déjà optimisé).
"""
import subprocess
import sys
import time

BOX, PORT = "root@38.255.28.21", "20098"
DISTANT = r'''
import re, time
loads, ends, bakes = [], [], []
for l in open("/workspace/miner.log", errors="replace"):
    try: t = time.mktime(time.strptime(l[:19], "%Y-%m-%d %H:%M:%S"))
    except Exception: continue
    if "Loading checkpoint from" in l: loads.append(t)
    elif "warmup: moteur chaud" in l: ends.append(t)
    elif "bake termin" in l: bakes.append(t)
adv = []
for l in open("/workspace/miner.log", errors="replace"):
    m = re.search(r"checkpoint ([0-9]+) -> ([0-9]+)", l)
    if m: adv.append((l[:19], m.group(1), m.group(2)))
wd = []
try:
    for l in open("/workspace/watchdog.log", errors="replace"):
        if "VRAM" in l or "FUITE" in l: wd.append(l.strip())
except Exception: pass
print(repr({"loads": loads[-3:], "ends": ends[-3:], "bakes": bakes[-1:] ,
            "adv": adv[-2:], "wd": wd[-3:],
            "n_adv": len(adv)}))
'''


def snap():
    out = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=20", "-o", "StrictHostKeyChecking=no",
         "-p", PORT, BOX, "/workspace/venv/bin/python -"],
        input=DISTANT, capture_output=True, text=True, timeout=90).stdout.strip()
    return eval(out.splitlines()[-1]) if out else None


base = snap()
if base is None:
    print("ckpt-watch: box injoignable"); sys.exit(1)
n0 = base["n_adv"]
# ⚠️ PIÈGE (21/08 07h20) : `restart_miner.sh` tronque miner.log via `tee`.
# Un guetteur lancé AVANT un redémarrage garde une référence trop haute et ne
# se déclenche JAMAIS. On relit donc la référence à chaque démarrage — et on
# l affiche pour que ce soit vérifiable.
print(f"ckpt-watch: en veille — reference {n0} avancees dans le journal courant")

while True:
    time.sleep(120)
    try:
        d = snap()
    except Exception:
        continue
    if d is None or d["n_adv"] <= n0:
        continue
    # nouvelle avancée détectée
    horo, a, b = d["adv"][-1]
    L = d["loads"][-1] if d["loads"] else None
    E = min([e for e in d["ends"] if L and e >= L], default=None)
    prev_bake = None
    dl = lo = None
    if L and E:
        lo = (E - L) / 60
    print(f"═══ RECHARGEMENT DÉTECTÉ : checkpoint {a} → {b} à {horo} ═══")
    if lo is not None:
        print(f"   chargement : {lo:.1f} min")
    vram_ok = any("pic normal" in w for w in d["wd"])
    vram_ko = any("FUITE_VRAM" in w for w in d["wd"][-1:])
    if vram_ok:
        print("   ✅ garde VRAM : pic TOLÉRÉ, aucun redémarrage — le correctif marche")
    elif vram_ko:
        print("   ⛔ garde VRAM : a QUAND MÊME redémarré — le correctif ne suffit pas")
    else:
        print("   ○ garde VRAM : pas de pic au-dessus du seuil cette fois")
    for w in d["wd"]:
        print("     journal :", w[:120])
    n0 = d["n_adv"]
