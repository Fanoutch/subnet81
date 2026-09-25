"""Prior math v2 (25/09) : mots + métadonnées du dataset + indice de marge.

Les marqueurs de métadonnées (source OMI, type/longueur de la réponse
attendue, longueur de l'énoncé) sont produits par UNE fonction partagée entre
l'entraînement et le mineur — un écart d'un caractère et les poids appris ne
s'appliquent plus. ``score_problem`` note un problème complet ; sans modèle
« meta », il retombe exactement sur ``score_prompt`` (priors historiques
inchangés).
"""
from reliquary.miner import prompt_predictor as pp

PROMPT = ("Solve the following math problem step by step.\n\n"
          "Find x if 2x + 3 = 11.\n\nPut your final answer within \\boxed{}.")


def test_meta_tokens_answer_types():
    t = pp.math_meta_tokens
    assert "__ans_int" in t(PROMPT, "augmented_math", "4")
    assert "__ans_neg" in t(PROMPT, "augmented_math", "-12")
    assert "__ans_mag2" in t(PROMPT, "augmented_math", "-12")
    assert "__ans_dec" in t(PROMPT, "math", "0.25")
    assert "__ans_frac" in t(PROMPT, "math", "\\frac{3}{20}")
    assert "__ans_frac" in t(PROMPT, "math", "3/20")
    assert "__ans_radpi" in t(PROMPT, "math", "2\\sqrt{3}")
    assert "__ans_tuple" in t(PROMPT, "math", "(1, 2)")
    assert "__ans_text" in t(PROMPT, "math", "x^2+1")


def test_meta_tokens_source_and_lengths():
    t = pp.math_meta_tokens(PROMPT, "augmented_gsm8k", "4")
    assert "__src_augmented_gsm8k" in t
    assert "__anslen0" in t
    # longueur de l'énoncé en mots, après le préambule du template
    assert "__plen0" in t


def test_meta_tokens_never_collide_with_text_tokens():
    # tokenize ne produit que [a-z0-9]+ : aucun jeton « __ » possible
    assert all(not tok.startswith("__") for tok in pp.tokenize(PROMPT + " __src_math"))


def test_score_problem_adds_meta_weights():
    m = {"type": "linear", "meta": "math_v1", "bias": 0.0,
         "weights": {"find": 1.0, "__src_augmented_math": 2.0, "__ans_int": -0.5}}
    prob = {"prompt": PROMPT, "source": "augmented_math", "ground_truth": "4"}
    assert pp.score_problem(m, prob) == 1.0 + 2.0 - 0.5


def test_score_problem_without_meta_is_score_prompt():
    m = {"type": "linear", "bias": 0.1, "weights": {"find": 1.0, "__ans_int": 9.0}}
    prob = {"prompt": PROMPT, "source": "augmented_math", "ground_truth": "4"}
    assert pp.score_problem(m, prob) == pp.score_prompt(m, PROMPT)
    wp = pp.train_word_priors([{"prompt": "a b", "target": 1.0},
                               {"prompt": "a c", "target": 0.0}], k=1.0)
    assert pp.score_problem(wp, {"prompt": "a b"}) == pp.score_prompt(wp, "a b")


def test_score_problem_missing_source_is_tolerated():
    m = {"type": "linear", "meta": "math_v1", "bias": 0.0,
         "weights": {"find": 1.0, "__src_augmented_math": 2.0}}
    assert pp.score_problem(m, {"prompt": PROMPT}) == 1.0
