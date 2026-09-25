"""Miroir local de ``verifier.evaluate_boxed_answer_probability`` (24/09).

Le validateur rejette un groupe (``BOXED_ANSWER_TAMPERED``, stage à dette)
si un token de la DERNIÈRE ``\\boxed{...}`` a une probabilité choisie
< 1e-3 alors que l'argmax est ≥ 0,99. Sous forced-seed ça arrive
honnêtement de temps en temps : on jette le groupe AVANT l'envoi.
"""
import math

from reliquary.miner.engine import find_last_boxed_token_range, local_boxed_screen


class _Tok:
    """Tokenizer factice : un id = un fragment de texte."""

    def __init__(self, frags):
        self.frags = frags

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.frags[i] for i in ids)


FRAGS = ["So ", "x", " = ", "\\boxed{", "4", "2", "}", ".", " Also ", "\\fbox{", "7", "}"]
TOK = _Tok(FRAGS)


def _lp(ps):
    return [math.log(p) for p in ps]


def test_range_of_last_boxed_content():
    # tokens : So x = \boxed{ 4 2 } .
    toks = [0, 1, 2, 3, 4, 5, 6, 7]
    assert find_last_boxed_token_range(toks, TOK) == (4, 5)


def test_last_of_boxed_and_fbox():
    toks = [3, 4, 6, 8, 9, 10, 11]  # \boxed{4} Also \fbox{7}
    assert find_last_boxed_token_range(toks, TOK) == (5, 5)


def test_unclosed_or_absent_box_is_none():
    assert find_last_boxed_token_range([0, 1, 2], TOK) is None
    assert find_last_boxed_token_range([3, 4, 5], TOK) is None


def test_matches_full_forward_scan_on_long_prefix():
    # Un long préfixe avant la boîte : la recherche par la fin doit donner
    # les mêmes indices que le balayage complet du validateur.
    toks = [0, 1, 2] * 200 + [3, 4, 5, 6, 7]
    n = len(toks)
    assert find_last_boxed_token_range(toks, TOK) == (n - 4, n - 3)


def test_screen_flags_confident_swap_inside_box():
    toks = [0, 1, 2, 3, 4, 5, 6, 7]
    ps = [0.9] * 8
    amx = [0.95] * 8
    ps[5], amx[5] = 5e-4, 0.995  # « 2 » tiré alors que le modèle voulait autre chose
    assert local_boxed_screen(_lp(ps), amx, toks, TOK) == "local_boxed_answer"


def test_screen_ignores_low_prob_outside_box_or_unconfident_argmax():
    toks = [0, 1, 2, 3, 4, 5, 6, 7]
    ps = [0.9] * 8
    amx = [0.95] * 8
    ps[1], amx[1] = 1e-5, 0.999  # hors de la boîte : pas ce contrôle
    ps[4], amx[4] = 5e-4, 0.60   # dans la boîte mais argmax peu sûr
    assert local_boxed_screen(_lp(ps), amx, toks, TOK) is None


def test_screen_margin_catches_values_just_above_validator_threshold():
    # 1,5e-3 passe chez le validateur (seuil 1e-3) mais notre marge (2e-3)
    # le jette : nos probabilités ne sont pas exactement les siennes.
    toks = [0, 1, 2, 3, 4, 5, 6, 7]
    ps = [0.9] * 8
    amx = [0.95] * 8
    ps[4], amx[4] = 1.5e-3, 0.985
    assert local_boxed_screen(_lp(ps), amx, toks, TOK) == "local_boxed_answer"


def test_screen_disabled_by_env(monkeypatch):
    monkeypatch.setenv("RELIQUARY_BOXED_SCREEN", "0")
    toks = [0, 1, 2, 3, 4, 5, 6, 7]
    ps = [0.9] * 8
    amx = [0.95] * 8
    ps[5], amx[5] = 5e-4, 0.995
    assert local_boxed_screen(_lp(ps), amx, toks, TOK) is None


def test_screen_without_argmax_is_noop():
    toks = [0, 1, 2, 3, 4, 5, 6, 7]
    assert local_boxed_screen(_lp([1e-6] * 8), None, toks, TOK) is None
