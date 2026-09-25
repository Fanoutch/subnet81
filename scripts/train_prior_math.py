#!/usr/bin/env python3
"""Prior MATH (24/09) — P(groupe en zone) à partir du texte de l'énoncé.

Sous V1 (fill-closed, paiement forfaitaire par groupe sélectionné) la seule
chose qui compte au tirage est que le groupe tombe EN ZONE (k ∈ [1,15] sur
16). Tirage uniforme mesuré le 24/09 : ~40 % en zone ; par source OMI :
augmented_math 49 %, augmented_gsm8k 18 %, gsm8k 17 %, math 24 %.

Étiquettes = NOS groupes notés sur la box math :
  - positifs : ``samples_math.jsonl`` (in_zone=True, avec k) ;
  - négatifs : ``ooz_math.jsonl`` (rewards 16/16 ou 0/16).
(Les groupes abandonnés à la terminaison ne sont pas étiquetés.)

Anti-fuite DOUBLE (leçon des 7 duels invalidés) : split TEMPOREL (dernier
quart par ts = éval) ET prompts d'éval jamais vus à l'entraînement.
Métriques : AUC, taux en zone du top-10 % / top-2 % prédit contre la base, et
contre le filtre « augmented_math seul » (si --sources fourni).

Usage : python3 scripts/train_prior_math.py --data DIR [--out FICHIER]
        (DIR contient samples_math.jsonl et ooz_math.jsonl, copiés de la box)
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from reliquary.miner import prompt_predictor as pp  # noqa: E402


def load_labels(data: Path) -> list[dict]:
    rows = {}
    for name, positive in (("samples_math.jsonl", True), ("ooz_math.jsonl", False)):
        p = data / name
        if not p.exists():
            continue
        for line in open(p):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("env", "openmathinstruct") != "openmathinstruct" or not r.get("prompt"):
                continue
            if positive:
                if not r.get("in_zone") or int(r.get("n_truncated", 0) or 0):
                    continue
                y = 1.0
            else:
                y = 0.0
            key = (int(r["prompt_idx"]), r.get("window_n"))
            rows[key] = {"prompt": r["prompt"], "idx": int(r["prompt_idx"]),
                         "ts": float(r.get("ts") or 0.0), "y": y}
    return sorted(rows.values(), key=lambda r: r["ts"])


def load_r2_positives(dirs, exclude) -> list[str]:
    """Énoncés math SÉLECTIONNÉS par le validateur (donc en zone pour la
    fenêtre qui les a vus). Positifs seulement : les mineurs jettent eux-mêmes
    leurs groupes hors zone, qui n'arrivent jamais dans l'archive."""
    import glob
    import gzip
    seen = set()
    for d in dirs:
        for f in sorted(glob.glob(str(Path(d) / "*.json.gz"))):
            try:
                arc = json.load(gzip.open(f))
            except Exception:
                continue
            for e in arc.get("batch") or []:
                if e.get("env_name") != "openmathinstruct":
                    continue
                t = e.get("prompt")
                if t and t not in exclude:
                    seen.add(t)
    return sorted(seen)


def auc(scored: list[tuple[float, float]]) -> float:
    pos = [s for s, y in scored if y > 0.5]
    neg = [s for s, y in scored if y <= 0.5]
    if not pos or not neg:
        return float("nan")
    order = sorted(scored, key=lambda t: t[0])
    rank_sum, i = 0.0, 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and order[j + 1][0] == order[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        rank_sum += avg * sum(1 for t in order[i:j + 1] if t[1] > 0.5)
        i = j + 1
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def top_rate(scored, frac):
    by = sorted(scored, key=lambda t: -t[0])
    top = by[:max(1, int(len(by) * frac))]
    return sum(y for _, y in top) / len(top), len(top)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--k", type=float, default=10.0, help="lissage des word priors")
    ap.add_argument("--min-eval", type=int, default=150)
    ap.add_argument("--r2", nargs="*", default=[],
                    help="caches R2 (window-N.json.gz) : groupes math sélectionnés "
                         "versés comme POSITIFS à l'entraînement seulement")
    ap.add_argument("--r2-weight", type=float, default=1.0,
                    help="part des positifs R2 gardés (sous-échantillonnage)")
    a = ap.parse_args()

    rows = load_labels(Path(a.data))
    n = len(rows)
    cut = int(n * 0.75)
    train, ev = rows[:cut], rows[cut:]
    seen = {r["idx"] for r in train}
    ev = [r for r in ev if r["idx"] not in seen]
    base_all = sum(r["y"] for r in rows) / max(1, n)
    print(f"corpus {n} (en zone {base_all:.1%}) | train {len(train)} | "
          f"éval postérieure ET jamais vue {len(ev)}")
    if len(ev) < a.min_eval:
        print(f"⚠️ éval trop petite (< {a.min_eval}) — attendre plus de collecte")
        return 2

    r2 = load_r2_positives(a.r2, exclude={r["prompt"] for r in ev})
    if a.r2_weight < 1.0:
        import random
        r2 = random.Random(81).sample(r2, int(len(r2) * a.r2_weight))
    if r2:
        print(f"+ {len(r2)} positifs R2 (énoncés distincts, hors éval) à l'entraînement")
    model = pp.train_word_priors(
        [{"prompt": r["prompt"], "target": r["y"]} for r in train]
        + [{"prompt": t, "target": 1.0} for t in r2], k=a.k)
    scored = [(pp.score_prompt(model, r["prompt"]), r["y"]) for r in ev]
    base = sum(y for _, y in scored) / len(scored)
    a_ = auc(scored)
    t10, n10 = top_rate(scored, 0.10)
    t25, n25 = top_rate(scored, 0.25)
    print(f"AUC = {a_:.3f}")
    print(f"en zone : base {base:.1%} | top-25 % {t25:.1%} (n={n25}) | "
          f"top-10 % {t10:.1%} (n={n10}) → ×{t10 / base:.2f}")

    model["meta"] = {"trained": time.strftime("%F %T"), "corpus": n,
                     "env": "openmathinstruct", "target": "in_zone_v1",
                     "auc_holdout": round(a_, 3), "base_holdout": round(base, 3),
                     "top10_holdout": round(t10, 3), "protocol": 6}
    if a.out:
        # modèle final entraîné sur TOUT le corpus (l'éval ci-dessus le juge)
        final = pp.train_word_priors(
            [{"prompt": r["prompt"], "target": r["y"]} for r in rows]
            + [{"prompt": t, "target": 1.0}
               for t in load_r2_positives(a.r2, exclude=set())], k=a.k)
        final["meta"] = dict(model["meta"], corpus_final=n)
        pp.save_model(final, a.out)
        print(f"modèle → {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
