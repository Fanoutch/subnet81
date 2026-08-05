# Prédicteur TF-IDF ciblé k=2 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-cibler le prédicteur de difficulté (prior TF-IDF par mot) sur le score d'auction `std·(1-mean)` (pique à k=2), ajouter un rapport d'impact des mots, et aligner le probe sur la config prod (cap 2600) — le tout testé CPU, prêt pour un dump 4B.

**Architecture:** On garde le module stdlib `reliquary/miner/prompt_predictor.py` (prior empirical-Bayes + idf) mais on remplace la cible « mean reward » par « score d'auction », la sélection devient un tri décroissant du score prédit, et on ajoute `word_impact_report` + métriques d'eval (Spearman, valeur réelle du top-N). Le probe enregistre la terminaison par rollout et tourne au cap prod.

**Tech Stack:** Python stdlib (aucun sklearn/numpy dans le module d'inférence), pytest. Probe = vLLM + transformers (GPU, hors périmètre code d'aujourd'hui).

## Global Constraints

Valeurs verbatim de la spec — s'appliquent à toutes les tâches :
- **Module `prompt_predictor.py` = stdlib-only** (pas de sklearn/numpy/torch ; JSON + dicts + floats).
- **Cible d'apprentissage = `std·(1-mean)`** du vecteur de rewards (std population, ÷n). Pique à k=2 (mean 0.25), = 0 à k=0 et k=8.
- **Config probe = config prod EXACTE** : `RELIQUARY_MAX_NEW_TOKENS=2600`, `RELIQUARY_MAX_TRUNCATED_CODE=0`, sampler protocole `T=0.6 / top_p=0.95 / top_k=20`, env code, checkpoint 4B `ReliquaryForge/qwen3.5-4b-reliquary-v4`.
- **Seul filtre de drop = non-terminaison (pas d'EOS)** ; pas de seuil de longueur. Au cap 2600, un prompt dont la solution dépasse 2600 → tronqué → k=0 → évité par construction.
- **Ne PAS** prédire la moyenne puis trier vers 0.25 (débordement mesuré vers k=0).
- Test runner : `cd /root/subnet81/reliquary-miner-priv && PYTHONPATH=. pytest <file>::<test> -v`.
- Pas de commit auto sans que l'utilisateur ait vu le diff (règle projet). Les étapes « Commit » ci-dessous sont à exécuter seulement si l'utilisateur le demande ; sinon s'arrêter à « tests verts ».

---

### Task 1: `auction_score` — la nouvelle cible

**Files:**
- Modify: `reliquary/miner/prompt_predictor.py` (ajouter la fonction)
- Test: `tests/test_prompt_predictor.py` (ajouter le test)

**Interfaces:**
- Produces: `auction_score(rewards: list[float]) -> float` — `std·(1-mean)`, std population.

- [ ] **Step 1: Write the failing test**

Ajouter dans `tests/test_prompt_predictor.py` :

```python
def test_auction_score_peaks_at_k2_and_zero_at_unanimous():
    # 8 rollouts binaires. std population = sqrt(mean·(1-mean)) ; ×(1-mean).
    assert pp.auction_score([0, 0, 0, 0, 0, 0, 0, 0]) == 0.0          # k=0
    assert pp.auction_score([1, 1, 1, 1, 1, 1, 1, 1]) == 0.0          # k=8
    k2 = pp.auction_score([1, 1, 0, 0, 0, 0, 0, 0])                    # k=2
    k3 = pp.auction_score([1, 1, 1, 0, 0, 0, 0, 0])                    # k=3
    k4 = pp.auction_score([1, 1, 1, 1, 0, 0, 0, 0])                    # k=4
    assert k2 > k3 > k4          # pique à k=2, décroît ensuite
    assert abs(k2 - 0.3247595) < 1e-6
    assert pp.auction_score([]) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_prompt_predictor.py::test_auction_score_peaks_at_k2_and_zero_at_unanimous -v`
Expected: FAIL avec `AttributeError: module ... has no attribute 'auction_score'`

- [ ] **Step 3: Write minimal implementation**

Ajouter dans `reliquary/miner/prompt_predictor.py` :

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_prompt_predictor.py::test_auction_score_peaks_at_k2_and_zero_at_unanimous -v`
Expected: PASS

- [ ] **Step 5: Commit** (seulement si demandé)

```bash
git add reliquary/miner/prompt_predictor.py tests/test_prompt_predictor.py
git commit -m "feat(predictor): auction_score target (std·(1-mean), peaks at k=2)"
```

---

### Task 2: `select_top` + retrait de la sélection mean-target

**Files:**
- Modify: `reliquary/miner/prompt_predictor.py` (ajouter `select_top` ; RETIRER `selection_score`, `select_eligible`, `evaluate`, `auc`)
- Test: `tests/test_prompt_predictor.py` (ajouter le test `select_top` ; RETIRER `test_selection_score_peaks_at_half_and_is_symmetric`, `test_auc_ranks_positives_above_negatives`, `test_select_eligible_returns_top_n_by_uncertainty_ranked`, `test_evaluate_computes_auc_of_selection_score_vs_in_zone`)

**Interfaces:**
- Consumes: `score_prompt(model, text)` (inchangé) — prédit désormais le score d'auction.
- Produces: `select_top(model, candidates: list[tuple[int, str]], top_n: int) -> list[int]` — indices triés par score prédit DÉCROISSANT.

- [ ] **Step 1: Write the failing test**

Dans `tests/test_prompt_predictor.py`, RETIRER les 4 tests mean-target listés ci-dessus, puis ajouter :

```python
def test_select_top_returns_highest_predicted_auction_first():
    # priors = score d'auction appris ; plus haut = plus payable.
    model = {
        "global_mean": 0.2,
        "word_priors": {"hard": 0.32, "mid": 0.20, "easy": 0.02},
        "idf": {"hard": 1.0, "mid": 1.0, "easy": 1.0},
    }
    candidates = [(10, "easy"), (20, "hard"), (30, "mid")]
    # scores prédits : hard 0.32 > mid 0.20 > easy 0.02 → top-2 = [20, 30]
    assert pp.select_top(model, candidates, top_n=2) == [20, 30]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_prompt_predictor.py::test_select_top_returns_highest_predicted_auction_first -v`
Expected: FAIL avec `AttributeError: ... 'select_top'`

- [ ] **Step 3: Write minimal implementation**

Dans `reliquary/miner/prompt_predictor.py` : SUPPRIMER `selection_score`, `select_eligible`, `evaluate`, `auc` (leur logique mean-target/AUC est remplacée). Ajouter :

```python
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
```

- [ ] **Step 4: Run the full module test file**

Run: `PYTHONPATH=. pytest tests/test_prompt_predictor.py -v`
Expected: PASS pour tous les tests restants (les 4 retirés ne doivent plus apparaître ; aucun `ImportError`/`AttributeError` résiduel — `train_and_evaluate` est réécrit en Task 4, garder son ancien test rouge est TOLÉRÉ jusque-là : le lancer isolément n'est pas requis à cette étape).

Note : `test_train_and_evaluate_learns_word_difficulty_and_ranks_holdout` va casser ici (il attend l'ancienne API AUC/in_zone). Le RÉÉCRIRE est la Task 4. À cette étape, le retirer temporairement du fichier pour garder la suite verte, puis le réintroduire en Task 4.

- [ ] **Step 5: Commit** (seulement si demandé)

```bash
git add reliquary/miner/prompt_predictor.py tests/test_prompt_predictor.py
git commit -m "refactor(predictor): replace mean-target selection with descending auction rank"
```

---

### Task 3: `word_impact_report` (+ persistance de `df` dans le modèle)

**Files:**
- Modify: `reliquary/miner/prompt_predictor.py` (ajouter `"df"` au dict modèle dans `train_word_priors` ; ajouter `word_impact_report`)
- Test: `tests/test_prompt_predictor.py`

**Interfaces:**
- Consumes: modèle avec `word_priors`, `idf`, et désormais `df`.
- Produces: `word_impact_report(model, min_df=3) -> dict` = `{"payable": [(tok, prior, df, idf), ...], "unanimous": [...]}` triés.

- [ ] **Step 1: Write the failing test**

```python
def test_train_word_priors_persists_document_frequency():
    records = [
        {"prompt": "the rare", "target": 0.3},
        {"prompt": "the cat", "target": 0.3},
    ]
    model = pp.train_word_priors(records, k=10.0)
    assert model["df"]["the"] == 2
    assert model["df"]["rare"] == 1


def test_word_impact_report_ranks_payable_words_over_unanimous():
    # "recursion" appris payable (prior haut), "loop" unanime (prior bas),
    # "x" trop rare (df=1 < min_df=2) → exclu des deux listes.
    model = {
        "global_mean": 0.15,
        "word_priors": {"recursion": 0.31, "loop": 0.02, "x": 0.31},
        "idf": {"recursion": 0.7, "loop": 0.7, "x": 2.0},
        "df": {"recursion": 40, "loop": 40, "x": 1},
    }
    rep = pp.word_impact_report(model, min_df=2)
    payable_tokens = [t for t, *_ in rep["payable"]]
    unanimous_tokens = [t for t, *_ in rep["unanimous"]]
    assert payable_tokens[0] == "recursion"      # plus haut prior en tête
    assert unanimous_tokens[0] == "loop"          # plus bas prior en tête
    assert "x" not in payable_tokens and "x" not in unanimous_tokens  # df filtré
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_prompt_predictor.py::test_train_word_priors_persists_document_frequency tests/test_prompt_predictor.py::test_word_impact_report_ranks_payable_words_over_unanimous -v`
Expected: FAIL (`KeyError: 'df'` puis `AttributeError: 'word_impact_report'`)

- [ ] **Step 3: Write minimal implementation**

Dans `train_word_priors`, changer le `return` pour inclure `df` :

```python
    return {"global_mean": global_mean, "word_priors": word_priors,
            "idf": idf, "df": df}
```

Ajouter :

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/test_prompt_predictor.py -v`
Expected: PASS (y compris `test_save_load_round_trips_the_model` — le round-trip JSON tolère la clé `df` supplémentaire).

- [ ] **Step 5: Commit** (seulement si demandé)

```bash
git add reliquary/miner/prompt_predictor.py tests/test_prompt_predictor.py
git commit -m "feat(predictor): word_impact_report + persist df in model"
```

---

### Task 4: `spearman` + `train_and_evaluate` (cible auction + métriques)

**Files:**
- Modify: `reliquary/miner/prompt_predictor.py` (ajouter `spearman` ; réécrire `train_and_evaluate`)
- Test: `tests/test_prompt_predictor.py`

**Interfaces:**
- Consumes: `auction_score` (Task 1), `train_word_priors` (Task 3), `score_prompt`.
- Produces:
  - `spearman(xs: list[float], ys: list[float]) -> float`
  - `train_and_evaluate(train_rows, test_rows, k=10.0, top_frac=0.1) -> tuple[dict, dict]` — `metrics = {"spearman", "top_value", "base_value", "top_payable_rate"}`. `train_rows`/`test_rows` portent `rewards` (liste) ; `test_rows` portent aussi `in_zone` (bool).

- [ ] **Step 1: Write the failing test**

```python
def test_spearman_is_one_for_monotone_and_zero_for_flat():
    assert abs(pp.spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-9
    assert abs(pp.spearman([1, 2, 3, 4], [40, 30, 20, 10]) + 1.0) < 1e-9
    assert pp.spearman([1, 1, 1], [5, 6, 7]) == 0.0   # variance nulle → 0


def test_train_and_evaluate_targets_auction_and_reports_topN_lift():
    # "recursion" → groupes k=2 (auction haut) ; "loop" → k=8 (auction 0).
    hard = [1, 1, 0, 0, 0, 0, 0, 0]
    easy = [1, 1, 1, 1, 1, 1, 1, 1]
    train_rows = [
        {"prompt": "use recursion", "rewards": hard, "in_zone": True},
        {"prompt": "use recursion", "rewards": hard, "in_zone": True},
        {"prompt": "use loop", "rewards": easy, "in_zone": False},
        {"prompt": "use loop", "rewards": easy, "in_zone": False},
    ]
    test_rows = [
        {"prompt": "use recursion", "rewards": hard, "in_zone": True},
        {"prompt": "use loop", "rewards": easy, "in_zone": False},
    ]
    model, metrics = pp.train_and_evaluate(train_rows, test_rows, k=1.0, top_frac=0.5)
    # "recursion" a un prior d'auction > "loop"
    assert model["word_priors"]["recursion"] > model["word_priors"]["loop"]
    # le top-50% (1 ligne) est bien le prompt payable → valeur > base
    assert metrics["top_value"] > metrics["base_value"]
    assert metrics["top_payable_rate"] == 1.0
    assert metrics["spearman"] > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_prompt_predictor.py::test_spearman_is_one_for_monotone_and_zero_for_flat tests/test_prompt_predictor.py::test_train_and_evaluate_targets_auction_and_reports_topN_lift -v`
Expected: FAIL (`AttributeError: 'spearman'`, puis mismatch de signature sur `train_and_evaluate`)

- [ ] **Step 3: Write minimal implementation**

Ajouter `spearman` et REMPLACER l'ancien `train_and_evaluate` :

```python
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
```

Réintroduire (réécrit) le test retiré en Task 2 n'est pas nécessaire — `test_train_and_evaluate_targets_auction_and_reports_topN_lift` le remplace.

- [ ] **Step 4: Run the full module test file**

Run: `PYTHONPATH=. pytest tests/test_prompt_predictor.py -v`
Expected: PASS (toute la suite).

- [ ] **Step 5: Commit** (seulement si demandé)

```bash
git add reliquary/miner/prompt_predictor.py tests/test_prompt_predictor.py
git commit -m "feat(predictor): spearman + auction-target train_and_evaluate with top-N lift"
```

---

### Task 5: Réécrire `scripts/train_prompt_predictor.py`

**Files:**
- Modify: `scripts/train_prompt_predictor.py`
- Test: `tests/test_train_prompt_predictor_cli.py` (créer)

**Interfaces:**
- Consumes: `pp.train_and_evaluate`, `pp.word_impact_report`, `pp.save_model`.

- [ ] **Step 1: Write the failing test**

Créer `tests/test_train_prompt_predictor_cli.py` :

```python
"""Smoke test the training CLI end-to-end on a tiny synthetic probe file."""
from __future__ import annotations

import json
import os
import subprocess
import sys


def test_cli_trains_and_writes_model(tmp_path):
    hard = [1, 1, 0, 0, 0, 0, 0, 0]
    easy = [1, 1, 1, 1, 1, 1, 1, 1]
    rows = []
    for _ in range(8):
        rows.append({"prompt": "use recursion here", "rewards": hard,
                     "in_zone": True, "split": "train"})
        rows.append({"prompt": "use a simple loop", "rewards": easy,
                     "in_zone": False, "split": "train"})
    rows.append({"prompt": "use recursion here", "rewards": hard,
                 "in_zone": True, "split": "test"})
    rows.append({"prompt": "use a simple loop", "rewards": easy,
                 "in_zone": False, "split": "test"})
    in_path = tmp_path / "labeled.jsonl"
    in_path.write_text("\n".join(json.dumps(r) for r in rows))
    out_path = tmp_path / "model.json"

    res = subprocess.run(
        [sys.executable, "scripts/train_prompt_predictor.py",
         "--in", str(in_path), "--out", str(out_path), "--k", "1"],
        capture_output=True, text=True, cwd=".",
        env={**os.environ, "PYTHONPATH": "."},  # so `import reliquary` resolves
    )
    assert res.returncode == 0, res.stderr
    model = json.loads(out_path.read_text())
    assert "word_priors" in model and "df" in model
    # payable word learned above the unanimous one
    assert model["word_priors"]["recursion"] > model["word_priors"]["loop"]
    assert "top-" in res.stdout  # the lift line printed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_train_prompt_predictor_cli.py -v`
Expected: FAIL (l'ancien script appelle `train_and_evaluate` avec l'ancienne signature → `test_auc` unpack / `_lift` utilise `selection_score` supprimé → `AttributeError`).

- [ ] **Step 3: Rewrite the script**

Remplacer INTÉGRALEMENT le corps de `scripts/train_prompt_predictor.py` par :

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_train_prompt_predictor_cli.py -v`
Expected: PASS

- [ ] **Step 5: Commit** (seulement si demandé)

```bash
git add scripts/train_prompt_predictor.py tests/test_train_prompt_predictor_cli.py
git commit -m "feat(predictor): training CLI on auction target + word-impact report"
```

---

### Task 6: Probe — enregistrer la terminaison + aligner le cap défaut

**Files:**
- Modify: `scripts/difficulty_probe.py` (helper `count_truncated` ; l'appeler dans `stage_generate_code` et `stage_generate_code_hf` ; passer le défaut `--max-tokens` à 2600)
- Test: `tests/test_difficulty_probe_truncation.py` (créer)

**Interfaces:**
- Produces: `count_truncated(rollouts: list[list[int]], eos_ids) -> int` — nb de rollouts sans EOS (tronqués → jetés par le mineur).

- [ ] **Step 1: Write the failing test**

Créer `tests/test_difficulty_probe_truncation.py` :

```python
"""The probe must record, per group, how many rollouts truncated (no EOS) —
those are the ones the miner drops at the 2600 cap. Diagnostic only; the label
already reflects them as failures."""
from __future__ import annotations

import importlib.util
import pathlib
import sys

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "difficulty_probe.py"


def _load_probe():
    # scripts/ is not a package (no __init__.py) → load by path, matching
    # tests/test_difficulty_probe_bft.py.
    spec = importlib.util.spec_from_file_location("difficulty_probe", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["difficulty_probe"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_count_truncated_counts_rollouts_without_eos():
    probe = _load_probe()
    eos = [99]
    rollouts = [
        [1, 2, 99],        # terminated (EOS present)
        [1, 2, 3, 4],      # truncated (no EOS)
        [99],              # terminated
        [5, 6, 7],         # truncated
    ]
    assert probe.count_truncated(rollouts, eos) == 2
    assert probe.count_truncated([], eos) == 0
    assert probe.count_truncated([[1, 2, 3]], []) == 0   # no eos ids → n/a
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_difficulty_probe_truncation.py -v`
Expected: FAIL (`AttributeError: ... 'count_truncated'`)

- [ ] **Step 3: Implement + wire**

Dans `scripts/difficulty_probe.py`, ajouter (près de `first_eos_index`) :

```python
def count_truncated(rollouts, eos_ids) -> int:
    """Rollouts with NO EOS = hit the generation cap → dropped by the miner
    (terminating_rollouts / too_many_truncated). Diagnostic: high counts at the
    prod cap mean the prompt's solution overflows 2600 tokens."""
    eos = set(int(e) for e in (eos_ids or ()))
    if not eos:
        return 0
    return sum(
        1 for toks in rollouts if not any(int(t) in eos for t in toks)
    )
```

Dans `stage_generate_code`, après avoir calculé `rewards` pour un groupe, ajouter la clé au row écrit :

```python
            r["n_truncated"] = count_truncated(rollouts, eos_ids)
```

(placer juste avant `r["rewards"], r["m"], r["sigma"], r["in_zone"] = ...` — même bloc de boucle ; `rollouts` = la variable itérée `for r, problem, rollouts in zip(...)`).

Faire le même ajout dans `stage_generate_code_hf` (variable de rollouts équivalente de cette fonction).

Enfin, aligner le défaut du CLI sur la prod — remplacer :

```python
    g.add_argument("--max-tokens", type=int, default=3500,
                   help="code envs only; math ignores it (protocol BFT budgets "
                        "2048 think + 512 answer)")
```

par :

```python
    g.add_argument("--max-tokens", type=int, default=2600,
                   help="code envs only (PROD cap = 2600, ops/launch_miner.sh); "
                        "math ignores it (protocol BFT budgets)")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_difficulty_probe_truncation.py -v`
Expected: PASS

- [ ] **Step 5: Commit** (seulement si demandé)

```bash
git add scripts/difficulty_probe.py tests/test_difficulty_probe_truncation.py
git commit -m "feat(probe): record per-group truncation + align default cap to prod 2600"
```

---

### Task 7: Runbook GPU (à exécuter quand l'H100 est prête — PAS de code)

**Files:** aucun (documentation opérationnelle ; s'assure que l'exécuteur a les commandes exactes).

Non testable en CPU — c'est la seule étape GPU. Séquence, dans un tmux sur l'H100 :

- [ ] **Step 1: Vérifier la suite CPU verte avant tout**

Run: `PYTHONPATH=. pytest tests/test_prompt_predictor.py tests/test_train_prompt_predictor_cli.py tests/test_difficulty_probe_truncation.py -v`
Expected: PASS (toutes).

- [ ] **Step 2: Échantillonner la tranche de prompts code**

```bash
PYTHONPATH=. python scripts/difficulty_probe.py sample --env code \
  --train-n 3000 --test-n 600 --out sample_code.jsonl
```

- [ ] **Step 3: Générer + grader au cap PROD (2600), MAX_TRUNCATED=0**

```bash
RELIQUARY_MAX_TRUNCATED_CODE=0 \
PYTHONPATH=. python scripts/difficulty_probe.py generate --env code \
  --backend vllm --in sample_code.jsonl --out labeled_code.jsonl \
  --model ReliquaryForge/qwen3.5-4b-reliquary-v4 \
  --m 8 --temperature 0.6 --max-tokens 2600 --max-model-len 8192
```

(Si vLLM sature la VRAM au cap : baisser `--batch`. `--max-tokens 2600` est le point non négociable — il DOIT égaler le cap mineur.)

- [ ] **Step 4: Entraîner + lire le verdict et le rapport d'impact des mots**

```bash
PYTHONPATH=. python scripts/train_prompt_predictor.py \
  --in labeled_code.jsonl --out prompt_predictor.json --k 10
```

Lire : Spearman, `top-N value xN` (gate ≥ 1.3), et surtout la liste **top payable-signal words** — c'est la réponse empirique à « les mots portent-ils un signal sur le 4B ». Sauvegarder `labeled_code.jsonl` (données réutilisables, hors rsync).

- [ ] **Step 5: Décision de câblage**

Si `top-N value` bat nettement la base (x≥1.3) → câbler `select_top` dans `selector.next(eligible=)` derrière un flag off-par-défaut (tâche de suivi, hors de ce plan). Sinon → résultat négatif documenté ; ne pas câbler.

---

## Notes de sûreté / rappels

- Les prompts qui tronquent à 2600 sortent en k=0 dans les labels → `select_top` les évite tout seul. Ne PAS ajouter de modélisation de longueur (YAGNI).
- Caveat forced-seed (probe = sampling libre proxy ; forced-seed non reproductible offline) : biais uniforme sur le std, à re-valider un jour sur prod. Non bloquant.
- `prompt_predictor.py` reste stdlib-only : aucune importation de sklearn/numpy/torch/validator dans le module d'inférence.
