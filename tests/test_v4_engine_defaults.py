"""Audit v4 items 2,3,4,5,8,10,12 : défauts moteur dérivés du protocole.

L'engine lit ces valeurs à l'import → tests par reload sous
RELIQUARY_PROTOCOL_VERSION. Défauts v3 verrouillés à l'identique.
"""
import importlib
import json

import pytest

from tests.v4helpers import reload_constants


def _reload_engine(monkeypatch, version=None):
    reload_constants(monkeypatch, version)
    import reliquary.miner.engine as e

    importlib.reload(e)
    return e


@pytest.fixture(autouse=True)
def _restore(monkeypatch):
    yield
    _reload_engine(monkeypatch)  # env nettoyé → état v3 module restauré


def test_v3_engine_defaults_unchanged(monkeypatch):
    e = _reload_engine(monkeypatch)
    assert (e.K_MIN, e.K_MAX) == (3, 5)
    assert e.AUCTION_MIN_SCORE == 0.32
    assert e.RANKING_TIME_BUDGET_S == 25
    assert e.P_STOP_LOCAL_MIN == 0.01
    assert e._VALIDATOR_STEADY_SIGMA_MIN == 0.43


def test_v4_engine_defaults(monkeypatch):
    e = _reload_engine(monkeypatch, 4)
    assert (e.K_MIN, e.K_MAX) == (1, 15)
    assert e.AUCTION_MIN_SCORE == 0.0
    assert e.RANKING_TIME_BUDGET_S == 12
    assert e.P_STOP_LOCAL_MIN == 0.001
    assert e._VALIDATOR_STEADY_SIGMA_MIN == 0.24


def test_v4_env_overrides_still_win(monkeypatch):
    monkeypatch.setenv("RELIQUARY_AUCTION_MIN_SCORE", "0.1")
    monkeypatch.setenv("RELIQUARY_K_MIN", "2")
    e = _reload_engine(monkeypatch, 4)
    assert e.AUCTION_MIN_SCORE == 0.1
    assert e.K_MIN == 2
    monkeypatch.delenv("RELIQUARY_AUCTION_MIN_SCORE")
    monkeypatch.delenv("RELIQUARY_K_MIN")


def test_v4_dump_label_uses_v4_sigma(monkeypatch, tmp_path):
    e = _reload_engine(monkeypatch, 4)
    out = tmp_path / "dump.jsonl"
    monkeypatch.setenv("RELIQUARY_SAMPLE_DUMP", str(out))
    # σ = 0.30 : hors zone v3 (0.43), EN zone v4 (0.24). k=8/16 → σ = 0.5 non ;
    # on construit un vecteur continu de σ ≈ 0.30 : 10×0.8 + 6×0.2.
    rewards = [0.8] * 10 + [0.2] * 6
    e.dump_group_sample(
        prompt="p", prompt_idx=1, rewards=rewards, env_name="opencodeinstruct",
    )
    row = json.loads(out.read_text().splitlines()[0])
    assert row["sigma_min"] == 0.24
    assert row["in_zone"] is (row["sigma"] >= 0.24)
    assert row["in_zone"] is True
    monkeypatch.delenv("RELIQUARY_SAMPLE_DUMP")


def test_v4_window_tally_m16(monkeypatch):
    e = _reload_engine(monkeypatch, 4)
    t = e.WindowTally()
    # k=1 sur 16, intact : σ=0.2421 ≥ 0.24 → payable sous v4
    t.add(1, [1.0] + [0.0] * 15, 0)
    assert t._n == 1 and t._payable == 1
    # un vecteur de 8 est ignoré sous v4 (M=16)
    t.add(1, [1.0] * 4 + [0.0] * 4, 0)
    assert t._n == 1
