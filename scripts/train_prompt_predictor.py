#!/usr/bin/env python3
"""Train + evaluate the TF-IDF word-prior predictor from probe labels.

Target = auction score std·(1-mean) (peaks at k=2). Reports held-out Spearman,
the top-fraction realised-value lift (the decision metric), and the word-impact
report (which words push toward the payable band). Persists the model as JSON
(CPU-only, stdlib, no sklearn/GPU at inference).

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


def _impact(model: dict, n: int = 15, min_df: int = 5) -> None:
    rep = pp.word_impact_report(model, min_df=min_df)
    print(f"\ntop payable-signal words (prior high → pull toward k=2, df>={min_df}):")
    for tok, prior, df, idf in rep["payable"][:n]:
        print(f"  {tok:24s} prior={prior:.3f}  df={df}  idf={idf:.2f}")
    print("top unanimous words (prior low → solve-all/fail-all, no value):")
    for tok, prior, df, idf in rep["unanimous"][:n]:
        print(f"  {tok:24s} prior={prior:.3f}  df={df}  idf={idf:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", default="prompt_predictor.json")
    ap.add_argument("--k", type=float, default=10.0, help="shrinkage strength")
    ap.add_argument("--top-frac", type=float, default=0.1)
    ap.add_argument("--min-df", type=int, default=5)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.in_path) if l.strip()]
    rows = [r for r in rows if "rewards" in r]
    train, test = _split(rows)

    def _rate(rs):
        z = sum(1 for r in rs if r.get("in_zone"))
        return 100.0 * z / len(rs) if rs else 0.0

    print(f"train={len(train)} (in-zone {_rate(train):.2f}%)  "
          f"test={len(test)} (in-zone {_rate(test):.2f}%)  k={args.k}")

    model, m = pp.train_and_evaluate(train, test, k=args.k, top_frac=args.top_frac)
    lift = (m["top_value"] / m["base_value"]) if m["base_value"] > 0 else float("inf")
    print(f"\nSpearman(pred, true auction) test = {m['spearman']:.3f}")
    print(f"top-{int(args.top_frac*100)}% realised auction = {m['top_value']:.4f}  "
          f"vs base {m['base_value']:.4f}  (x{lift:.2f})")
    print(f"top-{int(args.top_frac*100)}% in-zone rate = {100*m['top_payable_rate']:.1f}%")
    _impact(model, min_df=args.min_df)

    pp.save_model(model, args.out)
    print(f"\nmodel saved → {args.out}  ({len(model['word_priors'])} tokens)")

    print("\n=== VERDICT ===")
    if m["base_value"] > 0 and lift >= 1.3:
        print(f"  top-N value x{lift:.2f} >= 1.3 → SIGNAL: wire into selector.next(eligible=).")
    elif m["base_value"] > 0 and lift >= 1.1:
        print(f"  top-N value x{lift:.2f} weak → more labels before wiring.")
    else:
        print(f"  top-N value x{lift:.2f} ~ 1 → no text signal; predictor won't help.")


if __name__ == "__main__":
    main()
