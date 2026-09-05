#!/usr/bin/env python3
"""Usine a candidats HORAIRE — entraine toujours, ne promeut que sur victoire.

Lecon payee 3 fois (v6.0a, v6.0b, vol2-maitre) : un candidat frais ne vaut
rien tant qu'il n'a pas battu le deploye sur du STRICTEMENT posterieur — et
meme la, la simulation a menti une fois (vol2 : +0,47 en sim, -36 % de sigma=0
en vol). Donc : (1) eval posterieure obligatoire, (2) STAGING seulement apres
DEUX victoires consecutives, (3) jamais de restart automatique — le staging
depose le fichier, l'activation reste une decision humaine.
Cadence : cron horaire (les fenetres avancent ~40/h, ~2 400 groupes/h)."""
import json, math, os, random, re, statistics as st, subprocess, sys, time, glob

BOX = os.environ.get("RELIQUARY_BOX", "root@157.10.162.245")
PORT = os.environ.get("RELIQUARY_BOX_PORT", "20301")
D = "/root/subnet81/data"
ETAT = D + "/retrain_continu_etat.json"
sys.path.insert(0, "/root/subnet81/.worktrees/miner-priv-port-v4-dapo")
from reliquary.miner import prompt_predictor as pp

def log(m): print("%s | %s" % (time.strftime("%H:%M:%S"), m), flush=True)

# ── 1. donnees fraiches de la box (6 dernieres heures ≈ 240 fenetres) ──────
r = subprocess.run(["ssh", "-o", "ConnectTimeout=25", "-p", PORT, BOX,
    "tail -20000 /workspace/samples_v4.jsonl"], capture_output=True, text=True, timeout=300)
rows = {}
for l in r.stdout.splitlines():
    try: rec = json.loads(l)
    except Exception: continue
    if rec.get("prompt") and rec.get("rewards") and rec.get("completion_lens"):
        rows.setdefault((rec["window_n"], rec["prompt_idx"]), rec)
R = sorted(rows.values(), key=lambda x: x["window_n"])
if len(R) < 1500: log("trop peu de frais (%d) — abandon" % len(R)); sys.exit(0)
ws = sorted({x["window_n"] for x in R})
cut = ws[int(len(ws) * 0.6)]
log("frais %d groupes, fenetres %d-%d, coupe %d" % (len(R), ws[0], ws[-1], cut))

# ── 2. candidat : diff-prior re-entraine corpus complet + frais(train) ─────
corpus = []
for path in (D + "/samples_v4.jsonl",):
    for l in open(path, errors="replace"):
        try: rec = json.loads(l)
        except Exception: continue
        if rec.get("prompt") and rec.get("score") is not None and (rec.get("window_n") or 0) >= 37050:
            corpus.append({"prompt": rec["prompt"], "target": float(rec["score"])})
eval_txt = {x["prompt"] for x in R if x["window_n"] > cut}
train = [c for c in corpus if c["prompt"] not in eval_txt] + [
    {"prompt": x["prompt"],
     "target": st.pstdev([float(v) for v in x["rewards"]]) * (1 - st.mean(float(v) for v in x["rewards"]))}
    for x in R if x["window_n"] <= cut]
cand = pp.train_word_priors(train)
log("candidat entraine sur %d lignes" % len(train))

# ── 3. bras d'eval sur le POSTERIEUR uniquement ────────────────────────────
V59 = json.load(open(sorted(glob.glob(D + "/predictor_v5.9_*.json"))[-1]))
M = json.load(open(D + "/risk_sigma0_v1_2026-08-31.json"))
TOK = M["tokens"]; WRE = re.compile(r"\b\w\w+\b")
def psz(t):
    wsq = WRE.findall(t.lower()); g = {}
    for i, w in enumerate(wsq):
        g[w] = g.get(w, 0) + 1
        if i + 1 < len(wsq): b = w + " " + wsq[i+1]; g[b] = g.get(b, 0) + 1
    f = {k: (1 + math.log(c)) * TOK[k][0] for k, c in g.items() if k in TOK}
    n = math.sqrt(sum(v * v for v in f.values())) or 1.0
    z0, z16 = M["intercept_k0"], M["intercept_k16"]
    for k, v in f.items(): z0 += v / n * TOK[k][1]; z16 += v / n * TOK[k][2]
    return 1/(1+math.exp(-z0)) + 1/(1+math.exp(-z16))
H = []
for x in R:
    if x["window_n"] <= cut: continue
    k = sum(1 for v in x["rewards"] if v and v > 0.999)
    H.append({"k": k, "vol": min(sum(x["completion_lens"]), 8192*16),
              "s59": pp.score_prompt(V59, x["prompt"]),
              "scand": pp.score_prompt(cand, x["prompt"]),
              "surv": max(1e-9, 1 - psz(x["prompt"]))})
ARR = [9, 11, 13, 14.5, 16, 17, 18, 19]
def mesure(keyf, iters=500):
    rnd = random.Random(81); iz = []; paid = []
    for _ in range(iters):
        v = rnd.sample(H, min(22, len(H))); v.sort(key=keyf)
        iz.append(st.mean(1 if 1 <= h["k"] <= 15 else 0 for h in v[:10]))
        paid.append(sum(1 for s, h in enumerate(v[:8])
            if 1 <= h["k"] <= 15 and h["vol"] // ((int(ARR[s]//3)+1)*50) >= 48))
    return st.mean(iz), st.mean(paid), st.stdev(paid)/math.sqrt(iters)
bras = {
  "deploye_v59": lambda h: -h["s59"],
  "candidat":    lambda h: -h["scand"],
  "unifie_a8":   lambda h: -(h["surv"]**8) * max(h["s59"], 1e-9),
}
res = {}
for nom, f in bras.items():
    res[nom] = mesure(f)
    log("%-12s in_zone@10 %.1f%% | payees %.2f ± %.2f" % (nom, 100*res[nom][0], res[nom][1], res[nom][2]))

# ── 4. le portier : staging apres DEUX victoires consecutives ──────────────
base = res["deploye_v59"]
etat = {}
try: etat = json.load(open(ETAT))
except Exception: pass
for nom in ("candidat", "unifie_a8"):
    gagne = res[nom][1] - base[1] > 2 * (res[nom][2] + base[2])
    serie = etat.get(nom, 0) + 1 if gagne else 0
    etat[nom] = serie
    if serie >= 2:
        out = D + "/staging_%s_%s.json" % (nom, time.strftime("%m%d_%H%M"))
        json.dump(cand if nom == "candidat" else {"note": "unifie=a8, table a regenerer"}, open(out, "w"))
        log("🏆 %s : 2e victoire consecutive — STAGE %s (activation = decision humaine)" % (nom, out))
        etat[nom] = 0
    elif gagne:
        log("%s : victoire 1/2 — confirmation au prochain tour" % nom)
json.dump(etat, open(ETAT, "w"))
