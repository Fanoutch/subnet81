import reliquary.constants as c

def test_forced_seed_constants():
    assert c.FORCED_SEED_DOMAIN == "reliquary-forced-seed-v2"
    assert c.FORCED_SEED_STOCHASTIC_MAXPROB == 0.99
    assert c.FORCED_SEED_CONSISTENCY_FLOOR == 0.80
    assert c.FORCED_SEED_MIN_STOCH_POSITIONS == 30
    assert c.FORCED_SEED_ROLLOUT_FLOOR == 0.75
    assert c.FORCED_SEED_ROLLOUT_MIN_STOCH == 20
    assert c.FORCED_SEED_PROTOCOL_VERSION == 2
    assert isinstance(c.FORCED_SEED_ENFORCE, bool)
