from reliquary.miner.engine import MiningEngine


def _fake(active_envs, pool):
    e = object.__new__(MiningEngine)
    e.active_envs = list(active_envs)
    e._pool = pool
    return e


def test_pool_env_stats_counts_and_idxs_per_env():
    e = _fake(["math", "code"], [
        {"prompt_idx": 1, "env_name": "math"},
        {"prompt_idx": 2, "env_name": "math"},
        {"prompt_idx": 5, "env_name": "code"},
        {"prompt_idx": 9},  # legacy entry w/o env_name → defaults to active_envs[0]
    ])
    counts, in_pool = e._pool_env_stats()
    assert counts == {"math": 3, "code": 1}
    assert in_pool["math"] == {1, 2, 9}
    assert in_pool["code"] == {5}


def test_pool_env_stats_empty_pool():
    e = _fake(["math", "code"], [])
    counts, in_pool = e._pool_env_stats()
    assert counts == {"math": 0, "code": 0}
    assert in_pool == {"math": set(), "code": set()}


def test_pool_env_stats_single_env_all_math():
    e = _fake(["openmathinstruct"], [
        {"prompt_idx": 7, "env_name": "openmathinstruct"},
        {"prompt_idx": 8},  # legacy → openmathinstruct
    ])
    counts, in_pool = e._pool_env_stats()
    assert counts == {"openmathinstruct": 2}
    assert in_pool["openmathinstruct"] == {7, 8}
