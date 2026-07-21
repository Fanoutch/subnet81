import asyncio
from reliquary.miner.engine import MiningEngine
from reliquary.miner.mix_controller import MixController
from reliquary.protocol.submission import VerdictsResponse


def _engine(envs):
    e = object.__new__(MiningEngine)
    e.active_envs = list(envs)
    e._mix = MixController(envs, total_slots=8, slot_floor=1, alpha=1.0)
    e._submitted_env = {"a" * 64: "code"}
    e._verdicts_since = 0.0

    class _W:  # minimal wallet stub
        class hotkey:
            ss58_address = "5HotkeyStub"
    e.wallet = _W()
    return e


def test_verdicts_loop_one_tick_records_and_advances_cursor(monkeypatch):
    e = _engine(["math", "code"])
    payload = {"verdicts": [{"merkle_root": "a" * 64, "accepted": True,
               "reason": "accepted", "ts": 4.0, "rewarded": True}]}
    resp = VerdictsResponse.model_validate(payload)

    async def fake_fetch(url, hotkey, *, client, since=None):
        return resp

    import reliquary.miner.engine as eng
    monkeypatch.setattr(eng, "fetch_verdicts", fake_fetch, raising=False)

    # _tick_verdicts = corps d'une itération, extrait pour testabilité
    asyncio.run(e._tick_verdicts("http://v", client=None))
    assert e._verdicts_since == 4.0
    assert e._mix.target_slots()["code"] >= e._mix.target_slots()["math"]
