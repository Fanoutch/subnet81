from reliquary.miner.mix_controller import pick_bake_env, entry_env_name


def test_pick_bake_env_picks_largest_deficit():
    # math veut 6, en a 5 (déficit 1) ; code veut 2, en a 0 (déficit 2) → code
    assert pick_bake_env({"math": 6, "code": 2}, {"math": 5, "code": 0}) == "code"


def test_pick_bake_env_empty_pool_picks_first_with_highest_target():
    assert pick_bake_env({"math": 8, "code": 0}, {}) == "math"


def test_pick_bake_env_tie_breaks_by_target_order():
    # déficits égaux (4 et 4) → premier dans target_slots
    assert pick_bake_env({"math": 4, "code": 4}, {"math": 0, "code": 0}) == "math"


def test_pick_bake_env_single_env_always_that_env():
    assert pick_bake_env({"math": 8}, {"math": 3}) == "math"


def test_entry_env_name_reads_key():
    assert entry_env_name({"env_name": "code", "prompt_idx": 1}, "math") == "code"


def test_entry_env_name_defaults_when_missing():
    assert entry_env_name({"prompt_idx": 1}, "math") == "math"


def test_entry_env_name_defaults_when_falsy():
    assert entry_env_name({"env_name": "", "prompt_idx": 1}, "math") == "math"
