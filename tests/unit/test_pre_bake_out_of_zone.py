"""Tests for the out_of_zone pre-filter in _pre_bake_entry.

The miner drops a baked group whose reward std σ would be rejected by
the validator's `is_in_zone(σ, bootstrap=False)` check. This saves the
GPU cost of finalize and the per-window slot of firing an entry that
the validator guarantees to reject.

The decision lives in a small pure helper so the threshold logic is
trivially testable without mocking vLLM or HF.
"""

from reliquary.miner.engine import _skip_for_out_of_zone


def test_all_ones_zero_std_skips():
    """σ=0 (degenerate, all-1.0 rewards) — must skip."""
    assert _skip_for_out_of_zone([1.0] * 8) is True


def test_all_zeros_zero_std_skips():
    """σ=0 (degenerate, all-0.0 rewards) — must skip."""
    assert _skip_for_out_of_zone([0.0] * 8) is True


def test_7_of_8_correct_under_threshold_skips():
    """σ≈0.33 for [1,1,1,1,1,1,0,1] — below 0.43 cutoff → skip.

    This is the exact distribution that caused the bulk of our prod
    rejections (rewards=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0]
    rejected with reason=out_of_zone, prompt=695470 on 2026-05-16).
    """
    assert _skip_for_out_of_zone([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0]) is True


def test_4_of_8_correct_keeps():
    """σ≈0.5 for [1,0,1,0,1,0,1,0] — above 0.43 → keep."""
    assert _skip_for_out_of_zone([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]) is False


def test_5_of_8_correct_keeps():
    """σ≈0.484 for [1,1,1,1,1,0,0,0] — above 0.43 → keep."""
    assert _skip_for_out_of_zone([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]) is False


def test_empty_rewards_skips():
    """Degenerate empty list — rewards_std returns 0.0, must skip
    (refusing to mint an empty group is the safe default).
    """
    assert _skip_for_out_of_zone([]) is True


def test_threshold_uses_strict_zone():
    """The miner hardcodes bootstrap=False (strict, SIGMA_MIN=0.43).

    During a real bootstrap phase the miner is slightly more
    conservative than the validator — acceptable per the spec.
    """
    # σ for [1,1,0,1,1,1,1,1] = sqrt(0.875*0.125) ≈ 0.331 — below 0.43.
    # In bootstrap (0.33 cutoff) this would barely keep; in strict it skips.
    assert _skip_for_out_of_zone([1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]) is True


# --- 2026-07-18: align with the auction-v2 validator ---------------------
# The validator dropped the k∈[3,5] "binary reward distribution guard"
# (REWARD_DISTRIBUTION is now vestigial); its only gate is sigma >= 0.43
# (steady), i.e. k∈[2,6] for binary M=8. The miner must NOT over-filter k=2/k=6
# (the highest auction-value hard prompts), and must still drop k=1/k=7
# (sigma 0.33 < 0.43, which the validator rejects).

def test_k2_correct_kept():
    """k=2 [0,0,0,0,0,1,0,1] σ≈0.433 ≥ 0.43 → validator accepts → must KEEP.
    Highest auction difficulty score (std·(1-mean)) — do not discard."""
    assert _skip_for_out_of_zone([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]) is False


def test_k6_correct_kept():
    """k=6 [1,1,1,1,0,1,1,0] σ≈0.433 ≥ 0.43 → validator accepts → must KEEP."""
    assert _skip_for_out_of_zone([1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]) is False


def test_k1_correct_skips():
    """k=1 [1,0,0,0,0,0,0,0] σ≈0.331 < 0.43 → validator rejects → must SKIP."""
    assert _skip_for_out_of_zone([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) is True
