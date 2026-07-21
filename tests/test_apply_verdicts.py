from reliquary.miner.engine import MiningEngine
from reliquary.miner.mix_controller import MixController
from reliquary.protocol.submission import VerdictsResponse


def _engine_with_mix(envs):
    e = object.__new__(MiningEngine)
    e.active_envs = list(envs)
    e._mix = MixController(envs, total_slots=8, slot_floor=1, alpha=1.0)
    e._submitted_env = {}
    return e


def test_apply_verdicts_records_rewarded_outcome_for_mapped_env():
    e = _engine_with_mix(["math", "code"])
    e._submitted_env = {"a" * 64: "code"}
    payload = {"verdicts": [{"merkle_root": "a" * 64, "accepted": True,
               "reason": "accepted", "ts": 9.0, "rewarded": True}]}
    e._apply_verdicts(VerdictsResponse.model_validate(payload))
    slots = e._mix.target_slots()
    assert slots["code"] >= slots["math"]   # code a payé → reçoit ≥


def test_apply_verdicts_ignores_unknown_merkle_root():
    e = _engine_with_mix(["math"])
    payload = {"verdicts": [{"merkle_root": "f" * 64, "accepted": True,
               "reason": "accepted", "ts": 1.0, "rewarded": True}]}
    # merkle inconnu → pas de crash, pas d'outcome
    assert e._apply_verdicts(VerdictsResponse.model_validate(payload)) == 1.0


def test_apply_verdicts_skips_when_rewarded_is_none():
    e = _engine_with_mix(["math"])
    e._submitted_env = {"b" * 64: "math"}
    payload = {"verdicts": [{"merkle_root": "b" * 64, "accepted": False,
               "reason": "grail_fail", "ts": 2.0}]}  # rewarded absent → None
    e._apply_verdicts(VerdictsResponse.model_validate(payload))  # ne lève pas


def test_apply_verdicts_returns_max_ts_for_cursor():
    e = _engine_with_mix(["math"])
    payload = {"verdicts": [
        {"merkle_root": "c" * 64, "accepted": True, "reason": "accepted", "ts": 3.0},
        {"merkle_root": "d" * 64, "accepted": True, "reason": "accepted", "ts": 7.5},
    ]}
    assert e._apply_verdicts(VerdictsResponse.model_validate(payload)) == 7.5
