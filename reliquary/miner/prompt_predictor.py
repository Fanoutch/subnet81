"""Word-prior difficulty predictor — stdlib-only, CPU, no GPU, no sklearn.

Trained offline on difficulty-probe labels (prompt text + mean reward). At
runtime the miner loads the persisted JSON model and scores the current window's
prompts from their text, prioritising those predicted to land in the payable
sigma-zone (mean reward near 0.5). See difficulty-probe design notes.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

_WORD_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase unigrams + adjacent bigrams of a prompt's text."""
    words = _WORD_RE.findall((text or "").lower())
    bigrams = [f"{a} {b}" for a, b in zip(words, words[1:])]
    return words + bigrams


def train_word_priors(records: list[dict], k: float = 10.0) -> dict:
    """Empirical-Bayes word priors of the per-prompt target (mean reward).

    ``records`` = ``[{"prompt": str, "target": float}, ...]``. For each token
    (unigram/bigram) the prior is its target mean shrunk toward the global mean:

        prior[w] = (Σ target_over_docs_with_w + k·global_mean) / (df[w] + k)

    ``k`` is the shrinkage strength: a token seen far fewer than ``k`` times is
    pulled hard to ``global_mean`` (protects rare tokens from overfitting).
    """
    targets = [float(r["target"]) for r in records]
    global_mean = sum(targets) / len(targets) if targets else 0.5

    sums: dict[str, float] = {}
    df: dict[str, int] = {}
    for r in records:
        target = float(r["target"])
        for tok in set(tokenize(r["prompt"])):
            sums[tok] = sums.get(tok, 0.0) + target
            df[tok] = df.get(tok, 0) + 1

    n_docs = len(records)
    word_priors = {
        tok: (sums[tok] + k * global_mean) / (df[tok] + k) for tok in sums
    }
    idf = {tok: math.log(n_docs / df[tok]) for tok in df}
    return {"global_mean": global_mean, "word_priors": word_priors, "idf": idf, "df": df}


def score_prompt(model: dict, text: str) -> float:
    """Predicted mean reward = idf-weighted mean of known-token priors.

    Tokens absent from the model (or with zero idf) contribute nothing. If no
    token carries weight, fall back to the global mean (an uninformative guess).
    """
    priors = model["word_priors"]
    idf = model["idf"]
    num = 0.0
    den = 0.0
    for tok in tokenize(text):
        w = idf.get(tok, 0.0)
        if w > 0.0 and tok in priors:
            num += w * priors[tok]
            den += w
    return num / den if den > 0.0 else model["global_mean"]


def select_top(
    model: dict, candidates: list[tuple[int, str]], top_n: int
) -> list[int]:
    """Top-``top_n`` prompt indices by PREDICTED auction score, descending.

    ``candidates`` = ``[(prompt_idx, prompt_text), ...]`` for the current window
    slice. Higher predicted ``std·(1-mean)`` = more payable → baked first. Feed
    the result to ``selector.next(eligible=set(...))``.
    """
    ranked = sorted(
        candidates, key=lambda c: score_prompt(model, c[1]), reverse=True
    )
    return [idx for idx, _text in ranked[:top_n]]


def word_impact_report(model: dict, min_df: int = 3) -> dict:
    """Rank tokens by their learned auction-prior, restricted to ``df >= min_df``.

    Returns ``{"payable": [...], "unanimous": [...]}`` where each entry is
    ``(token, prior, df, idf)``:
      - ``payable``   = highest priors — words that pull a prompt toward the k=2
                        payable band (the high-impact words we want to detect).
      - ``unanimous`` = lowest priors — words tied to solve-all / fail-all groups
                        (no auction value).
    ``df`` gates out rare tokens whose prior is unreliable. This is the empirical
    answer to "which words carry payability signal on the checkpoint".
    """
    priors = model["word_priors"]
    idf = model["idf"]
    df = model["df"]
    kept = [
        (tok, priors[tok], df[tok], idf.get(tok, 0.0))
        for tok in priors
        if df.get(tok, 0) >= min_df
    ]
    payable = sorted(kept, key=lambda e: e[1], reverse=True)
    unanimous = sorted(kept, key=lambda e: e[1])
    return {"payable": payable, "unanimous": unanimous}


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation (ties get averaged ranks). 0 = no monotonic
    association, ±1 = perfectly concordant/discordant. Returns 0.0 if either
    input is constant (undefined correlation)."""
    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for m in range(i, j + 1):
                r[order[m]] = avg
            i = j + 1
        return r

    n = len(xs)
    if n < 2:
        return 0.0
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    vx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    vy = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    if vx == 0.0 or vy == 0.0:
        return 0.0
    return cov / (vx * vy)


def train_and_evaluate(
    train_rows: list[dict], test_rows: list[dict], k: float = 10.0,
    top_frac: float = 0.1,
) -> tuple[dict, dict]:
    """Train word priors on the AUCTION target; report held-out metrics.

    Per-row target = ``auction_score(row["rewards"])``. Returns
    ``(model, metrics)`` with:
      - ``spearman``         : Spearman(predicted score, true auction) on test.
      - ``top_value``        : mean true auction of the top-``top_frac`` test rows
                               ranked by predicted score (the decision metric).
      - ``base_value``       : mean true auction over all test rows
                               (= expected value of a random pick).
      - ``top_payable_rate`` : fraction of that top with ``in_zone`` True.
    ``top_value`` beating ``base_value`` by a clear margin is the deployment gate.
    """
    records = [
        {"prompt": r["prompt"], "target": auction_score(r["rewards"])}
        for r in train_rows
    ]
    model = train_word_priors(records, k=k)
    preds = [score_prompt(model, r["prompt"]) for r in test_rows]
    truth = [auction_score(r["rewards"]) for r in test_rows]
    order = sorted(range(len(test_rows)), key=lambda i: preds[i], reverse=True)
    n_top = max(1, int(len(order) * top_frac))
    top = order[:n_top]
    metrics = {
        "spearman": spearman(preds, truth),
        "top_value": sum(truth[i] for i in top) / n_top,
        "base_value": (sum(truth) / len(truth)) if truth else 0.0,
        "top_payable_rate": sum(
            1 for i in top if test_rows[i].get("in_zone")
        ) / n_top,
    }
    return model, metrics


def save_model(model: dict, path) -> None:
    """Persist the model as JSON (no sklearn/numpy — plain dicts and floats)."""
    Path(path).write_text(json.dumps(model))


def load_model(path) -> dict:
    """Load a JSON model persisted by :func:`save_model`."""
    return json.loads(Path(path).read_text())


def auction_score(rewards: list[float]) -> float:
    """Auction value of a rollout group = ``std·(1-mean)``.

    Peaks at k=2 (mean 0.25), collapses to 0 at k=0 (all-fail) and k=8
    (all-pass). Population std (÷n), mirroring ``validator.verifier.rewards_std``
    — kept inline so this module stays stdlib-only. This is the training target:
    ranking prompts by predicted auction_score surfaces the payable band.
    """
    n = len(rewards)
    if n == 0:
        return 0.0
    mean = sum(rewards) / n
    std = (sum((r - mean) ** 2 for r in rewards) / n) ** 0.5 if n >= 2 else 0.0
    return std * (1.0 - mean)
