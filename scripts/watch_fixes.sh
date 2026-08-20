#!/bin/bash
# Surveillance des 2 fixes du 20/08 : fuite de processus de grading + malus anti-court.
while true; do
  sleep 600
  ssh -o ConnectTimeout=20 -p 20098 root@38.255.28.21 'python3 - <<PYEOF
import re, subprocess, time
# 1) fantômes de grading (le fix P0 doit les maintenir à ~0)
n = int(subprocess.run(["bash","-c","pgrep -fc code_grader_driver || echo 0"],
                       capture_output=True, text=True).stdout.strip() or 0)
# 2) vitesse de génération vs âge du moteur
rows=[]
for l in open("/workspace/miner.log", errors="replace"):
    m=re.search(r"groupe 1/8 prêt à ([0-9.]+)s", l)
    if m:
        try: t=time.mktime(time.strptime(l[:19], "%Y-%m-%d %H:%M:%S"))
        except Exception: continue
        rows.append((t, float(m.group(1))))
# AGE DU MOTEUR - fiabilise le 20/08 22h. Ecart premier<->dernier lot sous-estime
# (ignore le chargement) et a fait croire a un redemarrage fantome. On prend
# ici lecart entre le premier horodatage du log (recree a chaque restart par
# tee) et maintenant. NE PAS utiliser ps -o etimes : valeurs aberrantes sur
# cette box (4118139328 s mesures = 130 ans).
# NB: pas dapostrophe dans ce bloc, il vit dans une chaine shell entre quotes.
_t0 = None
for l in open("/workspace/miner.log", errors="replace"):
    try:
        _t0 = time.mktime(time.strptime(l[:19], "%Y-%m-%d %H:%M:%S")); break
    except Exception:
        continue
age = (time.time()-_t0)/60 if _t0 else ((rows[-1][0]-rows[0][0])/60 if len(rows)>1 else 0)
recent = sorted(v for _,v in rows[-12:])
med = recent[len(recent)//2] if recent else 0
# 3) taux de rollouts courts (le malus doit le faire baisser)
tot = sum(1 for l in open("/workspace/miner.log", errors="replace") if "prêt à" in l)
sh  = sum(1 for l in open("/workspace/miner.log", errors="replace") if "short_rollout" in l)
print(f"age={age:.0f}min gen1={med:.1f}s fantomes={n} courts={sh}/{tot} ({100*sh/max(tot,1):.0f}%)")
PYEOF' 2>/dev/null
done
