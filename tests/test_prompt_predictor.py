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
