import json, hashlib, random, sys
sys.path.insert(0, "/root/subnet81/reliquary-miner-priv")
from reliquary.miner import prompt_predictor as pp

def load(p): return [json.loads(l) for l in open(p) if l.strip()]
old = load("/root/subnet81/reliquary-miner-priv/ops/prod_backup_2026-08-07/samples_code.jsonl")
new = load(__import__("os").path.dirname(__file__) + "/samples_code_2026-08-10.jsonl")

# dedup global par prompt (garde la 1re occurrence), ordre chronologique
seen, rows = set(), []
for r in old + new:
    h = hashlib.md5(r["prompt"].encode()).hexdigest()
    if h in seen: continue
    seen.add(h); rows.append(r)

# split temporel : test = derniers 30% d'aujourd'hui
n_test = int(len(new) * 0.30)
test = rows[-n_test:]
trainrows = rows[:-n_test]

def clean(r): return "n_truncated" in r and not r["n_truncated"]
def payable_clean(r): return bool(r.get("in_zone")) and clean(r)

def auction(rew):
    m = sum(rew)/len(rew); v = sum((x-m)**2 for x in rew)/len(rew)
    return (v**0.5) * (1-m)

# cibles — A et B sur lignes propres uniquement ; C sur toutes les lignes taguées
train_clean = [r for r in trainrows if clean(r)]
train_tagged = [r for r in trainrows if "n_truncated" in r]
recA = [{"prompt": r["prompt"], "target": 1.0 if r.get("in_zone") else 0.0} for r in train_clean]
recB = [{"prompt": r["prompt"], "target": auction(r["rewards"])} for r in train_clean]
recC = [{"prompt": r["prompt"], "target": 1.0 if payable_clean(r) else 0.0} for r in train_tagged]

models = {
    "A_inzone_clean": pp.train_word_priors(recA, k=10.0),
    "B_auction_clean": pp.train_word_priors(recB, k=10.0),
    "C_payable_noTrunc": pp.train_word_priors(recC, k=10.0),
    "v11_prod": pp.load_model("/root/subnet81/reliquary-miner-priv/ops/prod_backup_2026-08-07/prompt_predictor_v11.json"),
}

# éval : protocole mineur = choisir 1 parmi 20 candidats tirés du held-out
rng = random.Random(42)
N_TRIALS = 4000
pool = test
print(f"train: {len(trainrows)} rows ({len(train_clean)} propres, {len(train_tagged)} taguées) | test: {len(pool)} rows "
      f"| base test: in_zone_clean {100*sum(payable_clean(r) for r in pool)/len(pool):.1f}%, tronqués {100*sum(1 for r in pool if r.get('n_truncated'))/len(pool):.1f}%")

cache = {name: {} for name in models}
def score(name, r):
    h = id(r)
    c = cache[name]
    if h not in c: c[h] = pp.score_prompt(models[name], r["prompt"])
    return c[h]

results = {name: {"pay": 0, "trunc": 0, "auct": 0.0} for name in models}
results["random"] = {"pay": 0, "trunc": 0, "auct": 0.0}
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

print(f"\n{'modèle':20s} {'payable&propre':>14s} {'tronqué':>8s} {'auction moy':>12s}")
for name, r in results.items():
    print(f"{name:20s} {100*r['pay']/N_TRIALS:13.1f}% {100*r['trunc']/N_TRIALS:7.1f}% {r['auct']/N_TRIALS:12.4f}")

# sauvegarde des candidats
import os
for name in ("A_inzone_clean", "B_auction_clean", "C_payable_noTrunc"):
    pp.save_model(models[name], os.path.dirname(__file__) + f"/predictor_v2_{name}.json")
print("\nmodèles sauvés dans le scratchpad (PAS déployés)")
