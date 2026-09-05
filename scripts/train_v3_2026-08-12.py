"""Prédicteur v3 — ré-entraînement sur l'ère cap-8192 (box H100, 2026-08-11→12).

Données :
  • 8192-era : data/samples_code_8192_2026-08-12.jsonl — TOUT le fichier est
    au cap 8192 (créé 08-11 20:35, après la bascule). Labels natifs.
  • historique : data/samples_code.jsonl (cumul, caps mêlés). Les lignes
    PROPRES (all-EOS) y sont cap-invariantes : un rollout terminé par EOS sous
    2600 génère exactement les mêmes tokens à 8192 (le forced-seed ne dépend
    pas du cap). Les labels de troncature historiques, eux, NE transfèrent PAS.

Cibles (mêmes recettes que train_v2) :
  A_full   : in_zone sur lignes propres, historique propre + 8192 propre
  A_8192   : in_zone sur lignes propres, 8192 uniquement (l'historique aide-t-il ?)
  C_8192   : payable&propre sur lignes taguées 8192 uniquement (labels natifs)
Baselines : v2_A prod (candidate A du 08-11) + random.

Éval : protocole mineur (1 pick parmi 20), held-out TEMPOREL = derniers 30 %
du fichier 8192 ; les prompts du test sont exclus de tous les trains.
"""
import json, hashlib, random, sys, os
sys.path.insert(0, "/root/subnet81/reliquary-miner-priv")
from reliquary.miner import prompt_predictor as pp

def load(p): return [json.loads(l) for l in open(p) if l.strip()]

era8192 = load("/root/subnet81/data/samples_code_8192_2026-08-12.jsonl")
hist = load("/root/subnet81/data/samples_code.jsonl")

def ph(r): return hashlib.md5(r["prompt"].encode()).hexdigest()

n_test = int(len(era8192) * 0.30)
test = era8192[-n_test:]
test_hashes = {ph(r) for r in test}
train_8192 = [r for r in era8192[:-n_test] if ph(r) not in test_hashes]

# historique : dédoublonné par prompt, prompts du test exclus
seen = set(test_hashes)
hist_rows = []
for r in hist:
    h = ph(r)
    if h in seen: continue
    seen.add(h); hist_rows.append(r)

def clean(r): return "n_truncated" in r and not r["n_truncated"]
def payable_clean(r): return bool(r.get("in_zone")) and clean(r)

def auction(rew):
    m = sum(rew)/len(rew); v = sum((x-m)**2 for x in rew)/len(rew)
    return (v**0.5) * (1-m)

clean_full = [r for r in hist_rows if clean(r)] + [r for r in train_8192 if clean(r)]
clean_8192 = [r for r in train_8192 if clean(r)]
tagged_8192 = [r for r in train_8192 if "n_truncated" in r]

recA_full = [{"prompt": r["prompt"], "target": 1.0 if r.get("in_zone") else 0.0} for r in clean_full]
recA_8192 = [{"prompt": r["prompt"], "target": 1.0 if r.get("in_zone") else 0.0} for r in clean_8192]
recC_8192 = [{"prompt": r["prompt"], "target": 1.0 if payable_clean(r) else 0.0} for r in tagged_8192]

models = {
    "A_full":  pp.train_word_priors(recA_full, k=10.0),
    "A_8192":  pp.train_word_priors(recA_8192, k=10.0),
    "C_8192":  pp.train_word_priors(recC_8192, k=10.0),
    "v2A_prod": pp.load_model("/root/subnet81/reliquary-miner-priv/ops/prod_backup_2026-08-12/predictor_v2_A_inzone_clean.json"),
}

rng = random.Random(42)
N_TRIALS = 4000
pool = test
print(f"train: hist_clean+8192_clean={len(clean_full)} | 8192_clean={len(clean_8192)} | 8192_tagged={len(tagged_8192)}")
print(f"test (8192-era, temporel): {len(pool)} rows | base: payable&propre "
      f"{100*sum(payable_clean(r) for r in pool)/len(pool):.1f}%, tronqués "
      f"{100*sum(1 for r in pool if r.get('n_truncated'))/len(pool):.1f}%")

cache = {name: {} for name in models}
def score(name, r):
    h = id(r); c = cache[name]
    if h not in c: c[h] = pp.score_prompt(models[name], r["prompt"])
    return c[h]

results = {name: {"pay": 0, "trunc": 0, "auct": 0.0} for name in list(models) + ["random"]}
for _ in range(N_TRIALS):
    cand = rng.sample(pool, 20)
    for name in models:
        pick = max(cand, key=lambda r: score(name, r))
        results[name]["pay"] += payable_clean(pick)
        results[name]["trunc"] += bool(pick.get("n_truncated"))
        results[name]["auct"] += auction(pick["rewards"]) if clean(pick) else 0.0
    pick = rng.choice(cand)
    results["random"]["pay"] += payable_clean(pick)
    results["random"]["trunc"] += bool(pick.get("n_truncated"))
    results["random"]["auct"] += auction(pick["rewards"]) if clean(pick) else 0.0

print(f"\n{'modèle':12s} {'payable&propre':>14s} {'tronqué':>8s} {'auction moy':>12s}")
for name, r in results.items():
    print(f"{name:12s} {100*r['pay']/N_TRIALS:13.1f}% {100*r['trunc']/N_TRIALS:7.1f}% {r['auct']/N_TRIALS:12.4f}")

best = max(models, key=lambda n: results[n]["pay"])
print(f"\nmeilleur: {best}")
if best != "v2A_prod":
    out = f"/root/subnet81/data/predictor_v3_{best}_2026-08-12.json"
    pp.save_model(models[best], out)
    print("sauvé:", out)
