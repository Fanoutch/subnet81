#!/usr/bin/env python3
"""Vigie concurrence — 3 signaux + alertes de durcissement."""
import json, os, sys, time
from collections import defaultdict
S = sys.argv[1]
sub = [json.loads(l) for l in open("/root/subnet81/data/submits_v4.jsonl")]
ver = [json.loads(l) for l in open("/root/subnet81/data/verdicts_v4.jsonl")]
m2 = {u["merkle_root"]: u for u in sub if u.get("merkle_root")}
last = {}
for v in ver:
    if v.get("merkle_root") in m2:
        last[v["merkle_root"]] = v
now = time.time()

# 1) rang de NOTRE première entrée par fenêtre (dernière heure)
best = {}
for mk, v in last.items():
    u = m2[mk]
    if u.get("ts", 0) < now - 3600 or v.get("canonical_rank") is None:
        continue
    w = u["window_n"]
    off = u.get("flip_offset_s") or 999
    if w not in best or off < best[w][0]:
        best[w] = (off, v["canonical_rank"])
ranks = sorted(r for _, r in best.values())
offs = sorted(o for o, _ in best.values())

# 2) heure de fermeture du batch (1er batch_filled par fenêtre)
seals = []
byw = defaultdict(list)
for u in sub:
    if u.get("ts", 0) > now - 3600 and u.get("flip_offset_s") is not None:
        byw[u["window_n"]].append(u)
for w, rows in byw.items():
    bf = [r["flip_offset_s"] for r in rows if r["reason"] == "batch_filled"]
    if bf:
        seals.append(min(bf))
seals.sort()

def med(xs):
    return xs[len(xs) // 2] if xs else None

# 3) peloton (dashboard)
lag = None
try:
    m = json.load(open(os.path.join(S, "market_last.json")))
    cw = m.get("current_window") or {}
    lg = cw.get("upload_lag_ms") or {}
    lag = round((lg.get("p50") or 0) / 1000, 1)
except Exception:
    pass

parts = []
if ranks:
    parts.append(f"rang 1re entrée méd {med(ranks)} (offset méd +{med(offs):.0f}s, n={len(ranks)} fen)")
if seals:
    parts.append(f"fermeture batch méd +{med(seals):.0f}s")
if lag:
    parts.append(f"lag p50 gagnants {lag}s")
line = "concurrence: " + " | ".join(parts) if parts else "concurrence: pas de données"

# ALERTES — recalibrées le 20/08 au soir.
#
# Le seuil absolu « rang > 20 » criait à CHAQUE cycle depuis des heures : le
# rang médian est durablement à 28-33, donc l'alerte signalait un NIVEAU, pas
# un CHANGEMENT — c'est-à-dire rien d'actionnable. Et le critère « batch ferme
# tôt » visait à l'envers : mesuré ce soir, une fermeture TARDIVE (+26 s) est
# favorable, elle laisse entrer nos entrées ; ce qui se durcit alors, c'est le
# classement à l'intérieur du batch, pas le portillon.
#
# On alerte donc sur une DÉGRADATION relative à la ligne de base glissante des
# 12 derniers relevés, conservée dans un fichier d'état.
import json as _json
import os as _os

ETAT = "/root/subnet81/data/.competition_baseline.json"
hist = []
if _os.path.exists(ETAT):
    try:
        hist = _json.load(open(ETAT))
    except Exception:
        hist = []
courant = {"rang": med(ranks) if ranks else None,
           "ferm": med(seals) if seals else None,
           "lag": lag or None}
base = {}
for k in ("rang", "ferm", "lag"):
    vals = [h[k] for h in hist if h.get(k) is not None]
    if len(vals) >= 4:
        vals = sorted(vals)
        base[k] = vals[len(vals) // 2]
hist = (hist + [courant])[-12:]
try:
    _json.dump(hist, open(ETAT, "w"))
except Exception:
    pass

alerts = []
if base.get("rang") and courant["rang"] and courant["rang"] > base["rang"] * 1.3:
    alerts.append(f"rang 1re entrée: {courant['rang']:.0f} vs {base['rang']:.0f} de référence")
if base.get("ferm") and courant["ferm"] and courant["ferm"] < base["ferm"] * 0.6:
    alerts.append(f"batch ferme plus tôt: +{courant['ferm']:.0f}s vs +{base['ferm']:.0f}s")
if base.get("lag") and courant["lag"] and courant["lag"] < base["lag"] * 0.6:
    alerts.append(f"peloton accéléré: {courant['lag']:.0f}s vs {base['lag']:.0f}s")
if alerts:
    line += "  ⚠️ DURCISSEMENT: " + " ; ".join(alerts)
print(line)
