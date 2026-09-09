"""Prédicteur v4 — cible = NUMÉRATEUR attendu (Σ tokens réalisés si soumissible).

Constat 2026-08-13 : sous #173 le rang = Σ tokens du groupe / rounds. v3
(binaire « in-zone ») sélectionne des groupes soumissibles mais COURTS
(médiane 7 200 tokens) → rangs 20-60 malgré une arrivée à 27 s. La cible v4 =
Σ completion_lens si (in-zone ∧ all-EOS) sinon 0 — le modèle apprend à viser
les prompts qui rapportent des GROS numérateurs.

Données : l'ère cap-8192 uniquement (les Σ des époques cap-2600 sont censurés).
Éval : 1 pick parmi 20 (protocole mineur), held-out temporel. Métriques :
  • numérateur moyen du pick (la cible réelle du classement)
  • taux payable&propre (il ne doit pas s'effondrer vs v3)
"""
import json, hashlib, random, sys
sys.path.insert(0, "/root/subnet81/reliquary-miner-priv")
from reliquary.miner import prompt_predictor as pp

def load(p): return [json.loads(l) for l in open(p) if l.strip()]

era = load("/root/subnet81/data/samples_code_8192plus_2026-08-13.jsonl")
era = [r for r in era if "n_truncated" in r and r.get("completion_lens")]

def ph(r): return hashlib.md5(r["prompt"].encode()).hexdigest()
def clean(r): return not r["n_truncated"]
def payable_clean(r): return bool(r.get("in_zone")) and clean(r)
def numer(r): return sum(r["completion_lens"]) if payable_clean(r) else 0.0

n_test = int(len(era) * 0.30)
test = era[-n_test:]
test_hashes = {ph(r) for r in test}
train = [r for r in era[:-n_test] if ph(r) not in test_hashes]

# cible normalisée (les word-priors apprennent mieux sur [0,1])
cap_group = 8 * 15616.0
recV4 = [{"prompt": r["prompt"], "target": min(numer(r), cap_group) / cap_group} for r in train]

models = {
    "v4_numerateur": pp.train_word_priors(recV4, k=10.0),
    "v3_prod": pp.load_model("/root/subnet81/data/predictor_v3_A_8192_2026-08-12.json"),
}

rng = random.Random(42)
N = 4000
print(f"train {len(train)} | test {len(test)} | base payable&propre "
      f"{100*sum(payable_clean(r) for r in test)/len(test):.1f}% | "
      f"numérateur moyen (toutes lignes test) {sum(numer(r) for r in test)/len(test):.0f}")

cache = {n: {} for n in models}
def score(n, r):
    c = cache[n]
    if id(r) not in c: c[id(r)] = pp.score_prompt(models[n], r["prompt"])
    return c[id(r)]

res = {n: {"num": 0.0, "pay": 0, "trunc": 0} for n in list(models) + ["random"]}
for _ in range(N):
    cand = rng.sample(test, 20)
    for n in models:
        pick = max(cand, key=lambda r: score(n, r))
        res[n]["num"] += numer(pick)
        res[n]["pay"] += payable_clean(pick)
        res[n]["trunc"] += bool(pick["n_truncated"])
    pick = rng.choice(cand)
    res["random"]["num"] += numer(pick)
    res["random"]["pay"] += payable_clean(pick)
    res["random"]["trunc"] += bool(pick["n_truncated"])

print(f"\n{'modèle':16s} {'numérateur moyen':>16s} {'payable&propre':>14s} {'tronqué':>8s}")
for n, r in res.items():
    print(f"{n:16s} {r['num']/N:16.0f} {100*r['pay']/N:13.1f}% {100*r['trunc']/N:7.1f}%")

if res["v4_numerateur"]["num"] > res["v3_prod"]["num"]:
    out = "/root/subnet81/data/predictor_v4_numerateur_2026-08-13.json"
    pp.save_model(models["v4_numerateur"], out)
    print("\nsauvé:", out)
