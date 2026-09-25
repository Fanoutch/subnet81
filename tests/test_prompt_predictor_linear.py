"""Prior math (24/09) : modèle LINÉAIRE (régression logistique exportée)
évalué par ``score_prompt`` sans dépendance. score = biais + Σ poids des
tokens (unigrammes + bigrammes de ``tokenize``) présents, chacun compté une
fois. Les modèles word-priors historiques restent inchangés."""
from reliquary.miner import prompt_predictor as pp


def test_linear_model_scores_bias_plus_present_token_weights():
    m = {"type": "linear", "bias": -0.5,
         "weights": {"triangle": 1.0, "dollars": -2.0, "the triangle": 0.25}}
    # « triangle » deux fois : compté UNE fois (présence binaire)
    assert pp.score_prompt(m, "The triangle, the triangle") == -0.5 + 1.0 + 0.25
    assert pp.score_prompt(m, "She earns dollars") == -0.5 - 2.0
    assert pp.score_prompt(m, "nothing known") == -0.5


def test_word_priors_models_unchanged():
    m = pp.train_word_priors([{"prompt": "a b", "target": 1.0},
                              {"prompt": "a c", "target": 0.0}], k=1.0)
    assert "type" not in m
    assert isinstance(pp.score_prompt(m, "a b"), float)
