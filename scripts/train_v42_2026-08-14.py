"""Prédicteur v4.2 — cible bucket, labels DÉCENSURÉS.

Problème des versions précédentes : tout groupe tronqué à un ancien cap
(2600, 8192) portait la cible 0 alors qu'il aurait pu terminer au cap 16384 →
le modèle apprenait à FUIR les prompts longs (nos soumissions plafonnaient à
Σ≈6-10k, bucket ~28, rangs 16-64 sauf fenêtres creuses).

Décensure : on EXCLUT de l'entraînement les lignes tronquées à un cap < 16384
(labels faux) ; on garde les tronquées À 16384 (vrais négatifs) et toutes les
lignes propres (labels cap-invariants). L'ère 16384 (miner du 08-13) apporte
les premiers positifs longs : 48 gros porteurs Σ>20k in-zone propres.

Cible : bucket réalisé Σ/(F+max), F=1916 (modèle de livraison mesuré H200).
"""
import json, hashlib, random, sys
sys.path.insert(0, "/root/subnet81/reliquary-miner-priv")
from reliquary.miner import prompt_predictor as pp

def load(p): return [json.loads(l) for l in open(p) if l.strip()]

rows = [r for r in load("/root/subnet81/data/samples_code.jsonl")
        if "n_truncated" in r and r.get("completion_lens")]

def censored(r):
    """Tronqué à un ancien cap (2600 ou 8192) → label faux, à exclure."""
    if not r["n_truncated"]: return False
    mx = max(r["completion_lens"])
    return mx < 16000  # tronqué avant le cap actuel = censure d'époque

rows = [r for r in rows if not censored(r)]

def ph(r): return hashlib.md5(r["prompt"].encode()).hexdigest()
def pc(r): return bool(r.get("in_zone")) and not r["n_truncated"]
F = 1916.0
def bucket(r):
    if not pc(r): return 0.0
    return min(sum(r["completion_lens"]), 124928.0) / (F + max(r["completion_lens"]))

# dédup par prompt (garde la DERNIÈRE occurrence = label le plus récent/décensuré)
by_h = {}
for r in rows: by_h[ph(r)] = r
rows = list(by_h.values())

rng = random.Random(42)
rng.shuffle(rows)
n_test = int(len(rows) * 0.25)
test, train = rows[:n_test], rows[n_test:]

bmax = max(bucket(r) for r in train) or 1.0
rec = [{"prompt": r["prompt"], "target": bucket(r) / bmax} for r in train]

models = {
    "v42": pp.train_word_priors(rec, k=10.0),
    "v41_prod": pp.load_model("/root/subnet81/data/predictor_v41_bucket_2026-08-13.json"),
}

N = 4000
longs = sum(1 for r in train if max(r["completion_lens"]) > 8192)
print(f"train {len(train)} (dont {longs} à rollouts >8192) | test {len(test)} | "
      f"positifs gros (Σ>20k) train: {sum(1 for r in train if bucket(r)>0 and sum(r['completion_lens'])>20000)}")

cache = {n: {} for n in models}
def sc(n, r):
    c = cache[n]
    if id(r) not in c: c[id(r)] = pp.score_prompt(models[n], r["prompt"])
    return c[id(r)]

res = {n: {"bkt": 0.0, "pay": 0, "sum": 0.0} for n in list(models) + ["random"]}
for _ in range(N):
    cand = rng.sample(test, 20)
    for n in models:
        p = max(cand, key=lambda r: sc(n, r))
        res[n]["bkt"] += bucket(p); res[n]["pay"] += pc(p)
        res[n]["sum"] += sum(p["completion_lens"]) if pc(p) else 0
    p = rng.choice(cand)
    res["random"]["bkt"] += bucket(p); res["random"]["pay"] += pc(p)
    res["random"]["sum"] += sum(p["completion_lens"]) if pc(p) else 0

print(f"\n{'modèle':10s} {'bucket':>8s} {'payable':>8s} {'Σ moyen (payables)':>19s}")
for n, r in res.items():
    mean_sum = r["sum"] / max(1, r["pay"])
    print(f"{n:10s} {r['bkt']/N:8.2f} {100*r['pay']/N:7.1f}% {mean_sum:19.0f}")

pp.save_model(models["v42"], "/root/subnet81/data/predictor_v42_decensure_2026-08-14.json")
print("\nsauvé: /root/subnet81/data/predictor_v42_decensure_2026-08-14.json")
