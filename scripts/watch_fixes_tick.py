import os, re, subprocess, time
# 1) fantômes de grading (le fix P0 doit les maintenir à ~0)
n = int(subprocess.run(["bash","-c","pgrep -fc code_grader_driver || echo 0"],
                       capture_output=True, text=True).stdout.strip() or 0)
# 2) vitesse de génération vs âge du moteur
# On suit TOUT lescalier, pas seulement le premier groupe : une famine CPU
# frappe surtout la FIN du lot. Pendant la fuite du 19/08, le groupe 1 restait
# a ~3 s pendant que le groupe 8 passait de 9 s a 21 s. gen8 est donc le
# signal precoce, gen1 le signal tardif.
# PASSE UNIQUE (21/08) : le journal fait des dizaines de Mo apres quelques
# heures ; le lire une fois par indicateur faisait deborder le delai SSH.
# Tout est extrait en un seul parcours.
rows=[]; last8=[]; bakes=[]; loads=[]; ends=[]; advs=[]; sh=0; tot=0
_re8 = re.compile(r"groupe ([0-9])/8 pr.t . ([0-9.]+)s")
for l in open("/workspace/miner.log", errors="replace"):
    if "pr.t " in l or "pret " in l or "prêt " in l:
        tot += 1
    if "short_rollout" in l:
        sh += 1
    m=_re8.search(l)
    if m:
        g, val = int(m.group(1)), float(m.group(2))
        if g == 1:
            try: rows.append((time.mktime(time.strptime(l[:19], "%Y-%m-%d %H:%M:%S")), val))
            except Exception: pass
        elif g == 8:
            last8.append(val)
        continue
    # ⚠️ La ligne « checkpoint N -> M » doit être dans CETTE condition, sinon
    # elle n atteint jamais les elif et `advs` reste vide — le bloc de
    # rechargement ne s affiche alors JAMAIS. (Défaut introduit puis corrigé
    # le 21/08 : la correction précédente plaçait le test hors de portée.)
    if ("bake termin" in l or "Loading checkpoint from" in l
            or "warmup: moteur chaud" in l
            or ("checkpoint " in l and " -> " in l)):
        try: t=time.mktime(time.strptime(l[:19], "%Y-%m-%d %H:%M:%S"))
        except Exception: continue
        if "bake termin" in l: bakes.append(t)
        elif "Loading checkpoint from" in l: loads.append(t)
        elif "warmup: moteur chaud" in l: ends.append(t)
        else: advs.append(t)
# AGE DU MOTEUR - fiabilise le 20/08 22h. Ecart premier<->dernier lot sous-estime
# (ignore le chargement) et a fait croire a un redemarrage fantome. On prend
# ici lecart entre le premier horodatage du log (recree a chaque restart par
# tee) et maintenant. NE PAS utiliser ps -o etimes : valeurs aberrantes sur
# cette box (4118139328 s mesures = 130 ans).
# NB: pas dapostrophe dans ce bloc, il vit dans une chaine shell entre quotes.
# AGE REEL DU PROCESSUS. TROIS sources de temps cassees sur cette box :
#  - `ps -o etimes` : 4.12e9 s (~130 ans), bug conteneur connu ;
#  - /proc/uptime : incoherent avec lhorloge du conteneur, donne un age NEGATIF ;
#  - le 1er horodatage de miner.log : le watchdog relance SANS tronquer le log,
#    donc il remontait a 21h52 alors que le moteur datait de 03h20 (440 min
#    annonces au lieu de 115).
# `ps -o lstart` est la seule qui marche : date absolue de demarrage.
age = 0
try:
    _out = subprocess.run(
        ["bash","-c","ps -o lstart= -p $(pgrep -f 'cli.main mine' | head -1)"],
        capture_output=True, text=True).stdout.strip()
    if _out:
        age = (time.time() - time.mktime(time.strptime(_out))) / 60.0
except Exception:
    age = (rows[-1][0]-rows[0][0])/60 if len(rows)>1 else 0
recent = sorted(v for _,v in rows[-12:])
med = recent[len(recent)//2] if recent else 0
# 3) taux de rollouts courts (le malus doit le faire baisser)
r8 = sorted(last8[-12:])
med8 = r8[len(r8)//2] if r8 else 0
# reference mesuree ce soir sur moteur sain : gen1 ~3.0s, gen8 ~7.6s
# pendant la fuite du 19/08 : gen8 montait a 21s en 45 min
# Un gen8 eleve a DEUX causes opposees :
#   - famine CPU (la fuite du 19/08) : gen8 monte ET gen1 se degrade ET les
#     fantomes saccumulent ;
#   - gros lot (bonne nouvelle) : gen8 monte SEUL, parce que les groupes
#     produisent plus de tokens. Mesure : gen8 18,7 s pour 9600 tokens contre
#     7,5 s pour 4400 - la generation est plus longue, pas ralentie.
# On nalerte donc que sur la SIGNATURE de la famine, pas sur gen8 seul.
alerte = "  FAMINE ?" if (med8 > 14 and (med > 5 or n > 5)) else ""

# RECHARGEMENT DE CHECKPOINT (21/08) : c est le poste de perte n1, 48,8 min sur
# une nuit, dont 85 % en TELECHARGEMENT (7,1 min de moyenne) et 15 % en
# chargement (deja optimise : re-warm, kill des EngineCore zombies).
# On mesure les deux separement pour verifier l effet de HF_XET_HIGH_PERFORMANCE :
# le telechargement doit passer sous 3 min. On affiche aussi le delai depuis le
# dernier rechargement, pour savoir si le chiffre est frais ou vieux.
ck = ""
try:
    # Une avancee REELLE de checkpoint, pas le chargement du DEMARRAGE.
    # Le mineur journalise « Loading checkpoint from » dans les deux cas :
    # les confondre faisait annoncer « ckpt il y a 4 min, dl=11,5 min »
    # alors quaucun rechargement navait eu lieu (la mesure partait du
    # dernier lot davant le redemarrage). On ne retient donc quun
    # chargement proche dune ligne « checkpoint N -> M ».
    L = None
    for _a in advs:
        _c = [x for x in loads if abs(x - _a) <= 120]
        if _c: L = max(_c)
    if L and ends:

        E = min([e for e in ends if e >= L], default=None)
        avant = max([b for b in bakes if b < L], default=None)
        if E and avant:
            dl, lo = (L - avant) / 60.0, (E - L) / 60.0
            age_ck = (time.time() - E) / 60.0
            flag = "" if dl <= 3.0 else "  TELECHARGEMENT LENT"
            ck = " | ckpt il y a %.0fmin: dl=%.1fmin load=%.1fmin%s" % (age_ck, dl, lo, flag)
except Exception:
    ck = ""
print(f"age={age:.0f}min gen1={med:.1f}s gen8={med8:.1f}s (sain ~7.6s) fantomes={n} courts={sh}/{tot} ({100*sh/max(tot,1):.0f}%){alerte}{ck}")
