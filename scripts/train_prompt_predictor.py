#!/usr/bin/env python3
"""Train + evaluate the word-prior difficulty predictor from probe labels.

Reads a difficulty-probe ``labeled.jsonl`` (rows with ``prompt``, ``rewards``,
``in_zone``, and a ``split`` tag), trains the empirical-Bayes word-prior model on
the train split, reports held-out AUC + the practical selection lift, and
persists the model as JSON (CPU-only, no sklearn/GPU at inference).

Usage:
  python scripts/train_prompt_predictor.py --in labeled_code.jsonl \
         --out prompt_predictor.json --k 10
"""
from __future__ import annotations

import argparse
import json
import random

from reliquary.miner import prompt_predictor as pp


def _split(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    train = [r for r in rows if r.get("split") == "train"]
    test = [r for r in rows if r.get("split") == "test"]
    if not train or not test:  # no tags → random 80/20
        random.Random(0).shuffle(rows)
        cut = max(1, int(len(rows) * 0.2))
        test, train = rows[:cut], rows[cut:]
        print("[train] no split tags → random 80/20")
    return train, test


def _lift(model: dict, test: list[dict], frac: float = 0.1) -> None:
    """Practical payoff: in-zone rate among the top-fraction the model would
    pick, vs the base rate. This is the number that maps to revenue."""
    ranked = sorted(
        test,
        key=lambda r: pp.selection_score(pp.score_prompt(model, r["prompt"])),
        reverse=True,
    )
    k = max(1, int(len(ranked) * frac))
    top = ranked[:k]
    base = sum(1 for r in test if r.get("in_zone")) / len(test)
    picked = sum(1 for r in top if r.get("in_zone")) / len(top)
    lift = (picked / base) if base > 0 else float("inf")
    print(f"\nselection lift (test): base in-zone {100*base:.2f}%  →  "
          f"top-{int(frac*100)}% picked {100*picked:.2f}%  (x{lift:.1f})")


def _interpret(model: dict, n: int = 12) -> None:
    """Words most associated with the uncertain (payable) band vs the certain
    extremes — a sanity check that the model learned something meaningful."""
    priors = model["word_priors"]
    idf = model["idf"]
    # only reasonably-distinctive tokens (idf > 0) to avoid stopword noise
    toks = [t for t in priors if idf.get(t, 0.0) > 0.5]
    by_uncertainty = sorted(toks, key=lambda t: abs(priors[t] - 0.5))
    print("\nmost 'uncertain' tokens (prior ~0.5, payable-leaning):")
    for t in by_uncertainty[:n]:
        print(f"  {t:24s} prior={priors[t]:.3f}")
    print("most 'decided' tokens (prior ~0 or ~1, skip-leaning):")
    for t in by_uncertainty[-n:]:
        print(f"  {t:24s} prior={priors[t]:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", default="prompt_predictor.json")
    ap.add_argument("--k", type=float, default=10.0, help="shrinkage strength")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.in_path) if l.strip()]
    rows = [r for r in rows if "rewards" in r and "in_zone" in r]
    train, test = _split(rows)

    ytr = sum(1 for r in train if r.get("in_zone")) / len(train)
    yte = sum(1 for r in test if r.get("in_zone")) / len(test)
    print(f"train={len(train)} (in-zone {100*ytr:.2f}%)  "
          f"test={len(test)} (in-zone {100*yte:.2f}%)  k={args.k}")

    model, test_auc = pp.train_and_evaluate(train, test, k=args.k)
    print(f"\nAUC test (held-out) = {test_auc:.3f}")
    _lift(model, test)
    _interpret(model)

    pp.save_model(model, args.out)
    print(f"\nmodel saved → {args.out}  ({len(model['word_priors'])} tokens)")

    print("\n=== VERDICT ===")
    if test_auc >= 0.60:
        print(f"  AUC {test_auc:.3f} >= 0.60 → SIGNAL: wire into selector.next(eligible=).")
    elif test_auc >= 0.55:
        print(f"  AUC {test_auc:.3f} weak → more labels or richer features before wiring.")
    else:
        print(f"  AUC {test_auc:.3f} ~ 0.5 → no text signal; predictor won't help.")


if __name__ == "__main__":
    main()
