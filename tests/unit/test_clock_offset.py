"""Clock-offset EMA: regression test for the HTTP Date floor-rounding bug.

Background
----------
The validator stamps every response with an HTTP Date header at 1-second
precision (RFC 7231). When the miner parses it back with
``parsedate_to_datetime(...).timestamp()`` it gets ``floor(T_validator)``.

If the miner naively computes ``offset = parsed_date - local_midpoint``
and feeds it into the EMA, the mean converges to ``-0.5 s`` even when
both clocks are perfectly NTP-synced. The validator's drand round check
is now zero-tolerance in BOTH directions, so any artificial lag
produces systematic ``STALE_ROUND`` rejects every time the validator
crosses a drand boundary (~8% of submissions on a 3 s quicknet period).

This test pins the converged offset to ~0 so the bug cannot regress.
"""

import random
from email.utils import formatdate


class _FakeResponse:
    def __init__(self, date_header: str):
        self.headers = {"date": date_header}


def _drive_ema(samples: int, seed: int = 0) -> float:
    """Feed ``samples`` synthetic /state responses into the EMA and return
    the final ``_DRAND_CLOCK_OFFSET_S``.

    Each sample picks a uniformly-random validator wall time, stamps the
    Date header at 1-s precision, and assumes the miner clock is perfectly
    NTP-synced (drift=0) with a 10 ms RTT.
    """
    from reliquary.miner import engine

    engine._VALIDATOR_OFFSET_EMA = None
    engine._DRAND_CLOCK_OFFSET_S = 0.0

    rng = random.Random(seed)
    base = 1_700_000_000.0  # arbitrary Unix epoch
    for _ in range(samples):
        t_v = base + rng.random()  # validator wall, uniform in [base, base+1)
        date_str = formatdate(timeval=t_v, usegmt=True)  # truncates to 1 s
        resp = _FakeResponse(date_str)
        t_send = t_v - 0.005   # 5 ms before validator stamped (one-way)
        t_recv = t_v + 0.005   # 5 ms after
        engine._apply_offset_from_validator_response(resp, t_send, t_recv)

    return engine._DRAND_CLOCK_OFFSET_S


def test_offset_mean_is_near_zero_post_floor_compensation():
    """With both clocks NTP-synced, the AVERAGE converged offset must be
    close to 0.

    Per-run EMA stationary noise (α=0.2 over uniform [-1,0] samples) is
    σ ≈ 0.1 s, so the per-run final can land in roughly [-0.2, +0.2]. We
    average across 30 independent EMA evolutions to get a tight estimate
    of the true mean.

    Pre-fix code (``+0.25`` bias): mean ≈ -0.25 s — solidly outside the
    band, miner runs ~250 ms behind the validator → systematic STALE_ROUND
    rejects on ~8% of submissions near every drand boundary.

    Post-fix code (``+0.5`` bias): mean ≈ 0 s — corrected clock matches
    validator wall, zero-tolerance drand check is satisfied except within
    a few-ms RTT window of round boundaries (handled separately by the
    ``RELIQUARY_DRAND_BOUNDARY_SAFETY_S`` sleep).
    """
    finals = [_drive_ema(samples=300, seed=s) for s in range(30)]
    mean = sum(finals) / len(finals)
    assert abs(mean) < 0.1, (
        f"mean converged offset over 30 EMA runs = {mean:+.3f}s; "
        "must be within ±0.1 s of zero. If this fails with mean ≈ -0.25 s, "
        "the +0.25 bias has been restored and the miner will lose ~8% of "
        "submissions to STALE_ROUND near every drand round boundary."
    )
