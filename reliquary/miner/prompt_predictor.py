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
    return {"global_mean": global_mean, "word_priors": word_priors, "idf": idf}


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


def selection_score(predicted_mean: float, target: float = 0.5) -> float:
    """Ranking score: higher = closer to the payable band centre (mean ~0.5).

    ``-|pred - target|`` so a prompt predicted near ``target`` ranks above one
    the model predicts it will solve-all or fail-all.
    """
    return -abs(predicted_mean - target)


def select_eligible(
    model: dict, candidates: list[tuple[int, str]], top_n: int
) -> list[int]:
    """Top-``top_n`` prompt indices ranked by uncertainty (most promising first).

    ``candidates`` = ``[(prompt_idx, prompt_text), ...]`` for the current window
    slice. Feed the result to ``selector.next(eligible=set(...))`` so the miner
    bakes the prompts most likely to land in the payable band first.
    """
    ranked = sorted(
        candidates,
        key=lambda c: selection_score(score_prompt(model, c[1])),
        reverse=True,
    )
    return [idx for idx, _text in ranked[:top_n]]


def auc(scores: list[float], labels: list[int]) -> float:
    """ROC AUC via the Mann-Whitney statistic (ties count as half).

    = probability a random positive outranks a random negative. 0.5 = no
    signal, 1.0 = perfect ranking. Used as the deployment gate: rank the
    held-out prompts by ``selection_score`` and check they surface the real
    in-zone (payable) prompts.
    """
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def evaluate(model: dict, rows: list[dict], label_key: str = "in_zone") -> float:
    """Held-out AUC of the uncertainty ranking against the true payable label.

    ``rows`` carry a prompt and a boolean ``in_zone`` (payable). Returns the AUC
    of ``selection_score(score_prompt(...))`` vs that label — the deployment
    gate (>= 0.60 means the text-only ranking surfaces payable prompts).
    """
    scores = [selection_score(score_prompt(model, r["prompt"])) for r in rows]
    labels = [1 if r.get(label_key) else 0 for r in rows]
    return auc(scores, labels)


def train_and_evaluate(
    train_rows: list[dict], test_rows: list[dict], k: float = 10.0,
) -> tuple[dict, float]:
    """Full pipeline from probe-labelled rows: target = mean of the reward
    vector; train word priors on ``train_rows``; report held-out AUC on
    ``test_rows``. Returns ``(model, test_auc)``."""
    records = [
        {"prompt": r["prompt"], "target": _mean(r["rewards"])} for r in train_rows
    ]
    model = train_word_priors(records, k=k)
    return model, evaluate(model, test_rows)


def save_model(model: dict, path) -> None:
    """Persist the model as JSON (no sklearn/numpy — plain dicts and floats)."""
    Path(path).write_text(json.dumps(model))


def load_model(path) -> dict:
    """Load a JSON model persisted by :func:`save_model`."""
    return json.loads(Path(path).read_text())
