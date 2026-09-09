#!/usr/bin/env python3
"""Prior v5.0 — premier prédicteur du monde v4, entraîné from scratch.

Corpus : probe ck0 + collecte live (samples_v4.jsonl), UNIQUEMENT des données
v4 (protocol_version==4). Cible = score d'enchère observé std·(1-mean)
(décision runbook §Prior v5). Anti-fuite DOUBLE (leçon des 7 duels invalidés) :
split TEMPOREL (dernier quart par ts = éval) ET prompts d'éval jamais vus à
l'entraînement. Métriques = Spearman + lift top-N + précision vedette.
Usage (dev box) : PYTHONPATH=/root/subnet81/.worktrees/miner-priv-port-v4-dapo \
  python3 scripts/train_prior_v50.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/subnet81/.worktrees/miner-priv-port-v4-dapo")
from reliquary.miner import prompt_predictor as pp  # noqa: E402

DATA = Path("/root/subnet81/data")
OUT = DATA / f"predictor_v50_{time.strftime('%Y-%m-%d_%H%M')}.json"


def load(p):
    rows = []
    try:
        for line in open(p):
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    except FileNotFoundError:
        pass
    return rows


def usable(r):
    return (r.get("prompt") and r.get("score") is not None
            and int(r.get("n_truncated", 0) or 0) == 0
            and int(r.get("protocol_version", 4) or 4) >= 4)


def main():
    rows = [r for r in load(DATA / "probe_ck0_opencodeinstruct.jsonl") if usable(r)]
    seen = {(r["prompt_idx"], r.get("ts")) for r in rows}
    for r in load(DATA / "samples_v4.jsonl"):
        if usable(r) and (r["prompt_idx"], r.get("ts")) not in seen:
            rows.append(r)
    rows.sort(key=lambda r: r.get("ts") or 0)
    n = len(rows)
    cut = int(n * 0.75)
    train, ev = rows[:cut], rows[cut:]
    train_idx = {r["prompt_idx"] for r in train}
    ev = [r for r in ev if r["prompt_idx"] not in train_idx]  # jamais vus
    print(f"corpus {n} | train {len(train)} | éval (postérieurs ET jamais vus) {len(ev)}")
    if len(ev) < 60:
        print("⚠️ éval trop petite — attendre plus de collecte"); return

    model = pp.train_word_priors(
        [{"prompt": r["prompt"], "target": float(r["score"])} for r in train])
    scored = [(pp.score_prompt(model, r["prompt"]), float(r["score"])) for r in ev]

    # Spearman
    def ranks(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        rk = [0.0] * len(xs)
        for pos, i in enumerate(order):
            rk[i] = pos
        return rk
    ra, rb = ranks([s for s, _ in scored]), ranks([t for _, t in scored])
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((a - ma) * (b - mb) for a, b in zip(ra, rb))
    va = sum((a - ma) ** 2 for a in ra) ** 0.5
    vb = sum((b - mb) ** 2 for b in rb) ** 0.5
    spear = cov / (va * vb) if va and vb else float("nan")

    # lift top-N (le picker prend le meilleur parmi N candidats)
    by_pred = sorted(scored, reverse=True)
    base = sum(t for _, t in scored) / len(scored)
    top10 = by_pred[:max(1, len(scored) // 10)]
    lift = (sum(t for _, t in top10) / len(top10)) / base if base else float("nan")
    # précision vedette
    base_v = sum(1 for _, t in scored if t >= 0.23) / len(scored)
    top20 = by_pred[:max(1, len(scored) // 5)]
    prec_v = sum(1 for _, t in top20 if t >= 0.23) / len(top20)

    print(f"Spearman(pred, score) = {spear:.3f}")
    print(f"lift top-10% (score moyen vs base {base:.3f}) = ×{lift:.2f}")
    print(f"P(vedette | top-20% pred) = {prec_v:.1%}  (base {base_v:.1%})")

    model["meta"] = {"trained": time.strftime("%F %T"), "corpus": n,
                     "target": "auction_score_v4", "spearman_holdout": round(spear, 3),
                     "lift_top10": round(lift, 2), "protocol": 4}
    json.dump(model, open(OUT, "w"))
    print(f"modèle → {OUT}")


if __name__ == "__main__":
    main()
