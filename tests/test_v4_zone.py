"""Task 5 du port v4 : zone gate σ 0.24/0.22 — tout k∈[1,15] payable sous v4.

Critère dynamic-sampling DAPO (upstream 8c38992) : σ(k=1|15, M=16) = 0.2421 >
0.24 → admis ; k=0/16 (σ=0) restent rejetés. v3 par défaut inchangé.
"""
import pytest

from reliquary.miner import zone
from tests.v4helpers import reload_constants


@pytest.fixture(autouse=True)
def _restore_constants(monkeypatch):
    yield
    reload_constants(monkeypatch)


def test_v4_thresholds(monkeypatch):
    monkeypatch.setattr("reliquary.constants.PROTOCOL_VERSION", 4)
    assert zone.active_thresholds() == (0.24, 0.22)


def test_v3_thresholds_unchanged(monkeypatch):
    monkeypatch.setattr("reliquary.constants.PROTOCOL_VERSION", 3)
    assert zone.active_thresholds() == (0.43, 0.33)


def test_v4_k1_of_16_in_zone(monkeypatch):
    monkeypatch.setattr("reliquary.constants.PROTOCOL_VERSION", 4)
    # σ(k=1, M=16) = √(1/16·15/16) = 0.2421 > 0.24 → payable sous v4
    rewards = [1.0] + [0.0] * 15
    assert zone.is_in_zone(rewards, bootstrap=False)
    assert zone.is_in_zone([1.0] * 15 + [0.0], bootstrap=False)


def test_v4_unanime_out_of_zone(monkeypatch):
    monkeypatch.setattr("reliquary.constants.PROTOCOL_VERSION", 4)
    assert not zone.is_in_zone([0.0] * 16, bootstrap=False)
    assert not zone.is_in_zone([1.0] * 16, bootstrap=False)


def test_v3_k1_of_8_still_rejected(monkeypatch):
    monkeypatch.setattr("reliquary.constants.PROTOCOL_VERSION", 3)
    rewards = [1.0] + [0.0] * 7  # σ = 0.331 < 0.43
    assert not zone.is_in_zone(rewards, bootstrap=False)


def test_v4_constants_sigma_min(monkeypatch):
    c = reload_constants(monkeypatch, 4)
    assert c.SIGMA_MIN == 0.24
    assert c.BOOTSTRAP_SIGMA_MIN == 0.22


def test_v3_constants_sigma_min_untouched(monkeypatch):
    # 0.33/0.33 = valeur historique de notre fork (désalignement cosmétique
    # documenté — le chemin mineur effectif passe par zone.py à 0.43). La
    # byte-identité v3 prime sur la « correction ».
    c = reload_constants(monkeypatch)
    assert c.SIGMA_MIN == 0.33
    assert c.BOOTSTRAP_SIGMA_MIN == 0.33
