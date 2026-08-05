"""TDD for the word-prior difficulty predictor (stdlib-only, CPU, no GPU).

The predictor scores a prompt's TEXT to estimate its mean reward under the
subnet model, so the miner can prioritise prompts likely to land in the payable
sigma-zone (mean reward near 0.5). Trained offline on difficulty-probe labels.
"""
from __future__ import annotations

from reliquary.miner import prompt_predictor as pp


def test_tokenize_lowercases_and_emits_unigrams_and_bigrams():
    toks = pp.tokenize("Find the Target")
    # unigrams (lowercased) + adjacent bigrams, order-independent
    assert set(toks) == {
        "find", "the", "target",
        "find the", "the target",
    }


def test_train_shrinks_rare_word_priors_toward_global_mean():
    # Two prompts, one all-fail (0.0) one all-pass (1.0) → global_mean 0.5.
    # Each word appears once, so with k=10 its prior is pulled hard to 0.5:
    #   prior = (sum_target + k*global_mean) / (count + k)
    records = [
        {"prompt": "alpha", "target": 0.0},
        {"prompt": "beta", "target": 1.0},
    ]
    model = pp.train_word_priors(records, k=10.0)
    assert model["global_mean"] == 0.5
    assert model["word_priors"]["alpha"] == (0.0 + 10 * 0.5) / (1 + 10)
    assert model["word_priors"]["beta"] == (1.0 + 10 * 0.5) / (1 + 10)


def test_train_produces_idf_downweighting_common_tokens():
    import math
    # "the" appears in all 4 docs → idf 0 (ignored); "rare" in 1 → idf log(4).
    records = [
        {"prompt": "the rare", "target": 0.5},
        {"prompt": "the cat", "target": 0.5},
        {"prompt": "the dog", "target": 0.5},
        {"prompt": "the fish", "target": 0.5},
    ]
    model = pp.train_word_priors(records, k=10.0)
    assert model["idf"]["the"] == math.log(4 / 4)
    assert model["idf"]["rare"] == math.log(4 / 1)


def test_score_prompt_is_idf_weighted_mean_of_known_priors():
    model = {
        "global_mean": 0.5,
        "word_priors": {"easy": 0.9, "hard": 0.1},
        "idf": {"easy": 2.0, "hard": 1.0},
    }
    # "easy hard" bigram is unknown → skipped; only the two unigrams count.
    pred = pp.score_prompt(model, "easy hard")
    assert pred == (2.0 * 0.9 + 1.0 * 0.1) / (2.0 + 1.0)


def test_score_prompt_falls_back_to_global_mean_when_all_unknown():
    model = {"global_mean": 0.42, "word_priors": {}, "idf": {}}
    assert pp.score_prompt(model, "totally novel words") == 0.42


def test_selection_score_peaks_at_half_and_is_symmetric():
    assert pp.selection_score(0.5) == 0.0
    assert pp.selection_score(0.9) == pp.selection_score(0.1)
    # a prompt predicted near 0.5 ranks above one predicted near-certain
    assert pp.selection_score(0.5) > pp.selection_score(0.95)


def test_save_load_round_trips_the_model(tmp_path):
    records = [
        {"prompt": "sort the list", "target": 0.3},
        {"prompt": "parse the string", "target": 0.7},
    ]
    model = pp.train_word_priors(records, k=8.0)
    path = tmp_path / "model.json"
    pp.save_model(model, path)
    loaded = pp.load_model(path)
    assert loaded == model
    # and scoring is identical through the round-trip
    assert pp.score_prompt(loaded, "sort the string") == pp.score_prompt(
        model, "sort the string"
    )


def test_auc_ranks_positives_above_negatives():
    # perfect separation → 1.0
    assert pp.auc([3.0, 1.0, 2.0], [1, 0, 0]) == 1.0
    # reversed → 0.0
    assert pp.auc([1.0, 3.0, 2.0], [1, 0, 0]) == 0.0
    # positive between the two negatives → 0.5
    assert pp.auc([2.0, 3.0, 1.0], [1, 0, 0]) == 0.5
    # ties count as half
    assert pp.auc([1.0, 1.0], [1, 0]) == 0.5


def test_select_eligible_returns_top_n_by_uncertainty_ranked():
    model = {
        "global_mean": 0.5,
        "word_priors": {"mid": 0.5, "easy": 1.0, "hard": 0.2},
        "idf": {"mid": 1.0, "easy": 1.0, "hard": 1.0},
    }
    candidates = [(10, "mid"), (20, "easy"), (30, "hard")]
    # sel scores: mid 0.0 (best), hard -0.3, easy -0.5 → top-2 = [10, 30]
    top = pp.select_eligible(model, candidates, top_n=2)
    assert top == [10, 30]


def test_evaluate_computes_auc_of_selection_score_vs_in_zone():
    model = {
        "global_mean": 0.5,
        "word_priors": {"mid": 0.5, "easy": 1.0},
        "idf": {"mid": 1.0, "easy": 1.0},
    }
    rows = [
        {"prompt": "mid", "in_zone": True},   # sel 0.0  (high)
        {"prompt": "easy", "in_zone": False},  # sel -0.5 (low)
    ]
    assert pp.evaluate(model, rows) == 1.0


def test_train_and_evaluate_learns_word_difficulty_and_ranks_holdout():
    train_rows = [
        {"prompt": "use loop", "rewards": [1, 1, 1, 1, 1, 1, 1, 1], "in_zone": False},
        {"prompt": "use loop", "rewards": [1, 1, 1, 1, 1, 1, 1, 1], "in_zone": False},
        {"prompt": "use recursion", "rewards": [1, 1, 1, 1, 0, 0, 0, 0], "in_zone": True},
        {"prompt": "use recursion", "rewards": [1, 1, 1, 0, 0, 0, 0, 0], "in_zone": True},
    ]
    test_rows = [
        {"prompt": "use recursion", "in_zone": True},
        {"prompt": "use loop", "in_zone": False},
    ]
    model, test_auc = pp.train_and_evaluate(train_rows, test_rows, k=1.0)
    # "recursion" learned as uncertain (mean ~0.44), "loop" as solved (mean 1.0)
    assert test_auc == 1.0
    assert model["word_priors"]["recursion"] < model["word_priors"]["loop"]


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
