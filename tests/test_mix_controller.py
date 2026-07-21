import random

from reliquary.miner.mix_controller import MixController


def test_cold_start_is_even_with_floor():
    mc = MixController(["math", "code"], total_slots=8, slot_floor=1)
    slots = mc.target_slots()
    assert sum(slots.values()) == 8
    assert slots["math"] >= 1 and slots["code"] >= 1
    assert abs(slots["math"] - slots["code"]) <= 1


def test_high_yield_env_gets_more_but_floor_respected():
    mc = MixController(["math", "code"], total_slots=8, slot_floor=1, alpha=1.0)
    for _ in range(20):
        mc.record_outcome("code", rewarded=True)
        mc.record_outcome("math", rewarded=False)
    slots = mc.target_slots()
    assert sum(slots.values()) == 8
    assert slots["code"] > slots["math"]
    assert slots["math"] >= 1


def test_sum_always_equals_total():
    mc = MixController(["math", "code"], total_slots=8, slot_floor=1, alpha=0.3)
    rng = random.Random(0)
    for _ in range(200):
        mc.record_outcome(rng.choice(["math", "code"]), rewarded=rng.random() < 0.4)
        assert sum(mc.target_slots().values()) == 8


def test_unknown_env_outcome_ignored():
    mc = MixController(["math"], total_slots=8, slot_floor=1)
    mc.record_outcome("nope", rewarded=True)
    assert mc.target_slots() == {"math": 8}
