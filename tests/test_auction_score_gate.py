"""Filtrer sur le SCORE D'ENCHÈRE, pas sur sigma seul.

Le validateur classe les candidats par ``std(rewards) * (1 - mean(rewards))``
(difficulty_auction.difficulty_score) et ne paie que les 8 premiers. Or notre
filtre ne testait que ``sigma >= 0.43``, ce qui laisse passer les k>=4 :
variance maximale mais score MINIMAL.

Mesuré sur 1835 groupes réels (2026-08-05) :
    k=2  (moyenne 0.2-0.32) : score médian 0.325  ← maximum théorique
    k=3  (moyenne 0.32-0.45): score médian 0.303
    k>=4 (moyenne >0.45)    : score médian 0.182  ← 72% de nos soumissions
Nos 3 soumissions classées 39/40/49 étaient dans le dernier groupe.
"""
import pytest

from reliquary.miner.engine import auction_score, passes_auction_gate


def _sigma(v):
    m = sum(v) / len(v)
    return (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5


class TestAuctionScore:
    def test_matches_validator_formula(self):
        v = [1.0] * 2 + [0.0] * 6                      # k=2
        assert auction_score(v) == pytest.approx(_sigma(v) * (1 - 0.25))

    def test_k2_scores_higher_than_k4(self):
        k2 = [1.0] * 2 + [0.0] * 6
        k4 = [1.0] * 4 + [0.0] * 4
        assert auction_score(k2) > auction_score(k4)
        assert auction_score(k2) == pytest.approx(0.3248, abs=1e-3)
        assert auction_score(k4) == pytest.approx(0.2500, abs=1e-3)

    def test_empty_is_zero(self):
        assert auction_score([]) == 0.0


class TestGate:
    def test_k2_passes(self):
        assert passes_auction_gate([1.0] * 2 + [0.0] * 6, min_score=0.30)

    def test_k3_passes(self):
        assert passes_auction_gate([1.0] * 3 + [0.0] * 5, min_score=0.30)

    def test_k4_rejected(self):
        """k=4 : sigma maximal (0.5) mais score 0.25 → rang ~39, jamais payé."""
        assert not passes_auction_gate([1.0] * 4 + [0.0] * 4, min_score=0.30)

    def test_unanimous_rejected(self):
        assert not passes_auction_gate([0.0] * 8, min_score=0.30)

    def test_threshold_zero_lets_everything_through(self):
        """Échappatoire de diagnostic, comme RELIQUARY_ZONE_SIGMA_MIN."""
        assert passes_auction_gate([1.0] * 4 + [0.0] * 4, min_score=0.0)
