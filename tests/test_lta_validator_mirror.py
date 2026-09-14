"""Miroir des contrôles d'authenticité RÉELLEMENT appliqués par le validateur V1.

Validateur 0a69244 (``verifier.evaluate_token_authenticity``, enforcé) :
  (1) tout token choisi < 1e-8 → TOKEN_TAMPERED ;
  (2) token NUMÉRIQUE (chiffre, ou ``- + . / −`` collé à un chiffre) choisi
      < 1e-6 alors que l'argmax ≥ 0,99 → TOKEN_TAMPERED.
Le contrôle « tous tokens » (p < 1e-5 et argmax ≥ 0,99) n'est PAS enforcé en
fill-closed (``ALL_TOKEN_AUTH_ENFORCE = PROTOCOL_VERSION != 5 and not
FILL_CLOSED_ENABLED``). Notre porte douce (p < 1e-4, argmax ≥ 0,985, tous
tokens) jetait pourtant 9 groupes sur 20 (fen 45885). Mode ``validator`` :
miroir de (1) et (2) avec marge ×10, porte douce en ombre seulement.
"""
from __future__ import annotations

import math

import pytest

from reliquary.miner import engine

DIG, MINUS, WORD = 5, 6, 7


class _Tok:
    _vocab = {"5": DIG, "-": MINUS, "hello": WORD, " 42": 8, " ": 9}

    def get_vocab(self):
        return dict(self._vocab)

    def decode(self, ids):
        inv = {v: k for k, v in self._vocab.items()}
        return "".join(inv[i] for i in ids)


def _lps(ps):
    return [math.log(p) for p in ps]


@pytest.fixture(autouse=True)
def _mode_validator(monkeypatch):
    monkeypatch.setenv("RELIQUARY_LTA_MODE", "validator")
    monkeypatch.setenv("RELIQUARY_LOCAL_TOKEN_AUTH", "1")
    for k in ("RELIQUARY_LTA_HARD_MIN", "RELIQUARY_LTA_CHOSEN_MAX",
              "RELIQUARY_LTA_ARGMAX_MIN", "RELIQUARY_LTA_NUMERIC_MAX",
              "RELIQUARY_LTA_NUMERIC_ARGMAX_MIN"):
        monkeypatch.delenv(k, raising=False)


def test_numeric_ids_comme_le_validateur():
    digits, signs = engine.numeric_token_ids(_Tok())
    assert digits == frozenset({DIG, 8})        # " 42".strip() → chiffres
    assert signs == frozenset({MINUS})


def test_porte_douce_non_numerique_ne_jette_plus():
    ids = engine.numeric_token_ids(_Tok())
    reason = engine.local_verif_screen(
        _lps([0.5, 5e-5]), [0.4, 0.995],
        completion_tokens=[WORD, WORD], numeric_ids=ids)
    assert reason is None
    assert engine._LTA_SHADOW["hit"] is True        # tracé en ombre


def test_token_numerique_edite_jete():
    ids = engine.numeric_token_ids(_Tok())
    reason = engine.local_verif_screen(
        _lps([0.5, 5e-6]), [0.4, 0.995],
        completion_tokens=[WORD, DIG], numeric_ids=ids)
    assert reason == "local_token_auth_numeric"


def test_signe_colle_a_un_chiffre_est_numerique():
    ids = engine.numeric_token_ids(_Tok())
    reason = engine.local_verif_screen(
        _lps([5e-6, 0.5]), [0.995, 0.4],
        completion_tokens=[MINUS, DIG], numeric_ids=ids)
    assert reason == "local_token_auth_numeric"


def test_signe_isole_non_numerique():
    ids = engine.numeric_token_ids(_Tok())
    reason = engine.local_verif_screen(
        _lps([5e-6, 0.5]), [0.995, 0.4],
        completion_tokens=[MINUS, WORD], numeric_ids=ids)
    assert reason is None


def test_numerique_argmax_faible_garde():
    ids = engine.numeric_token_ids(_Tok())
    reason = engine.local_verif_screen(
        _lps([0.5, 5e-6]), [0.4, 0.5],
        completion_tokens=[WORD, DIG], numeric_ids=ids)
    assert reason is None


def test_seuil_dur_toujours_applique():
    ids = engine.numeric_token_ids(_Tok())
    reason = engine.local_verif_screen(
        _lps([0.5, 5e-8]), [0.4, 0.2],
        completion_tokens=[WORD, WORD], numeric_ids=ids)
    assert reason == "local_token_auth_hard"


def test_mode_legacy_inchange(monkeypatch):
    monkeypatch.setenv("RELIQUARY_LTA_MODE", "soft")
    reason = engine.local_verif_screen(_lps([0.5, 5e-5]), [0.4, 0.995])
    assert reason == "local_token_auth"
