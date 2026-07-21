from reliquary.miner.engine import MiningEngine, _select_continuous_subset


def _r(reward, bt=True):
    # rollout minimal : tokens uniques (anti-dédup), reward, bt_ok, q10/median hauts
    _r.n = getattr(_r, "n", 0) + 1
    return {"all_tokens": [_r.n, _r.n + 1000], "reward": reward, "bt_ok": bt,
            "q10_local": 0.9, "median_local": 0.9, "p_stop_local": 0.9, "in_eos": True}


def test_continuous_bimodal_accepts():
    # 4×~1.0 + 4×~0.0 → std ~0.5 ≥ 0.46
    rolls = [_r(1.0), _r(0.95), _r(0.9), _r(1.0), _r(0.0), _r(0.05), _r(0.1), _r(0.0)]
    subset = _select_continuous_subset(rolls, size=8, sigma_target=0.46)
    assert subset is not None and len(subset) == 8


def test_continuous_all_pass_rejects():
    rolls = [_r(1.0) for _ in range(8)]            # σ=0
    assert _select_continuous_subset(rolls, size=8, sigma_target=0.46) is None


def test_continuous_middling_rejects():
    rolls = [_r(0.5) for _ in range(8)]            # σ≈0
    assert _select_continuous_subset(rolls, size=8, sigma_target=0.46) is None


def test_continuous_picks_extremes_from_pool():
    # pool de 12 : doit composer 8 max-variance qui franchit le seuil
    rolls = ([_r(1.0), _r(0.98), _r(0.95), _r(0.92)] + [_r(0.5)] * 4 +
             [_r(0.05), _r(0.02), _r(0.0), _r(0.0)])
    subset = _select_continuous_subset(rolls, size=8, sigma_target=0.46)
    assert subset is not None


def test_dispatch_binary_unchanged():
    eng = object.__new__(MiningEngine)

    class _Math:
        continuous_reward = False

    rolls = [_r(1.0) for _ in range(3)] + [_r(0.0) for _ in range(5)]
    subset, k = eng._try_select(rolls, _Math())   # voie binaire (k-band) intacte
    assert subset is not None and k == 3
