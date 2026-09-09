#!/usr/bin/env python3
"""Surveille le TÉLÉCHARGEMENT des checkpoints — le drapeau pose-t-il problème ?

CONTEXTE — `HF_XET_HIGH_PERFORMANCE=1` a été déployé le 21/08 à 07h16 sur une
prémisse FAUSSE : je croyais que le téléchargement coûtait 7 min par
rechargement. Il en coûte **7 secondes** ; les longs trous de génération sont
les phases d'entraînement du VALIDATEUR, pendant lesquelles aucune fenêtre n'est
ouverte (vérifié : 129 fenêtres attendues, 129 vues, zéro manquée).

Le drapeau est donc INERTE. Ce script existe pour vérifier qu'il ne NUIT pas,
sur trois signaux mesurables :
  1. la DURÉE réelle du transfert (début du journal → « File reconstruction
     completed successfully »). Référence sans le drapeau : quelques secondes.
  2. la CONCURRENCE maximale atteinte. Elle est ADAPTATIVE : Xet l'ajuste au
     temps de réponse. Mesurée à 59-64 AVANT le drapeau et 27 APRÈS — le
     drapeau ne l'augmente donc pas. Une montée durable au-delà de 64 serait
     le signe d'une contention réseau ou CPU.
  3. les vraies ERREURS (niveau ERROR/WARN). ⚠️ NE PAS compter les lignes du
     module `retry_wrapper` : il journalise les SUCCÈS (« Request Success »),
     pas des reprises. Mon premier compteur s'y était trompé.

Affiche une ligne par téléchargement, du plus ancien au plus récent.
"""
import json
import subprocess
import sys

BOX, PORT = "root@38.255.28.21", "20098"

DISTANT = r'''
import json, glob, os, datetime as dt
res = []
for f in sorted(glob.glob("/workspace/hf/xet/logs/*.log"), key=os.path.getmtime):
    # Un journal xet couvre TOUTE la vie du processus, pas un seul
    # telechargement : partir de sa premiere ligne donnait des durees de
    # plusieurs HEURES. On isole la RAFALE contigue qui precede chaque
    # « File reconstruction completed » — lignes espacees de moins de 60 s.
    ev = []
    for line in open(f, errors="replace"):
        try: d = json.loads(line)
        except Exception: continue
        ts = d.get("timestamp")
        if not ts: continue
        try: t = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception: continue
        msg = str(d.get("fields", {}).get("message", ""))
        lvl = d.get("level", "")
        vraie_err = (lvl in ("ERROR", "WARN")
                     and "retry_wrapper" not in str(d.get("filename", "")))
        c = 0
        if "Increased concurrency" in msg:
            try: c = int(msg.rsplit(" to ", 1)[1].split(";")[0])
            except Exception: c = 0
        ev.append((t, "File reconstruction completed" in msg, vraie_err, c))
    for i, (t, done, _, _) in enumerate(ev):
        if not done: continue
        j = i
        while j > 0 and (ev[j][0] - ev[j-1][0]).total_seconds() < 60:
            j -= 1
        rafale = ev[j:i+1]
        res.append({"h": t.strftime("%m-%d %H:%M"),
                    "s": round((t - rafale[0][0]).total_seconds(), 1),
                    "conc": max((x[3] for x in rafale), default=0),
                    "err": sum(1 for x in rafale if x[2]),
                    "ts": t.timestamp()})
print(json.dumps(res[-8:]))
'''


def main() -> int:
    out = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=20", "-o", "StrictHostKeyChecking=no",
         "-p", PORT, BOX, "/workspace/venv/bin/python -"],
        input=DISTANT, capture_output=True, text=True, timeout=90).stdout.strip()
    try:
        rows = json.loads(out.splitlines()[-1])
    except Exception:
        print("dl-watch: box injoignable ou journaux illisibles")
        return 1
    if not rows:
        print("dl-watch: aucun téléchargement journalisé")
        return 0
    print("  %-12s %9s %7s %8s  %s" % ("fin", "durée", "conc.", "erreurs", "verdict"))
    for r in rows:
        # 21/08 07:16 UTC = pose du drapeau (comparaison sur horodatage, pas
        # sur du texte : "21:26" >= "07:16" est vrai lexicalement mais faux)
        drapeau = "avec drapeau" if r.get("ts", 0) >= 1787296560 else "sans"
        souci = []
        if r["s"] > 60: souci.append("LENT")
        if r["conc"] > 70: souci.append("CONCURRENCE HAUTE")
        if r["err"] > 0: souci.append("%d ERREURS" % r["err"])
        print("  %-12s %8.1fs %7d %8d  %-13s %s"
              % (r["h"], r["s"], r["conc"], r["err"], drapeau,
                 " ".join(souci) if souci else "ok"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
