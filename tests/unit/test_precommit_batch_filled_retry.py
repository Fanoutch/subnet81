"""BATCH_FILLED retry on the precommit (port of upstream 790c0f3 / PR #197).

The upstream reference miner re-POSTs the SAME signed precommit on a
BATCH_FILLED using its generic (1, 2, 4) second backoff. ``drand_round`` is
covered by the precommit signature and the validator applies ZERO backward
tolerance, so a reused precommit is only admissible while its own 3-second
round is still running. These tests pin the two properties that make the port
safe: OFF by default, and never sleeping past the signed round's boundary.
"""

from __future__ import annotations

import asyncio

import pytest

from reliquary.miner import submitter as sub
from reliquary.protocol.submission import RejectReason


# --------------------------------------------------------------- env parsing
def test_retry_is_off_by_default(monkeypatch):
    monkeypatch.delenv(sub._PRECOMMIT_RETRY_DELAYS_ENV, raising=False)
    assert sub._precommit_retry_delays() == ()


def test_retry_profile_is_parsed(monkeypatch):
    monkeypatch.setenv(sub._PRECOMMIT_RETRY_DELAYS_ENV, "0.3, 0.6 ,1.0")
    assert sub._precommit_retry_delays() == (0.3, 0.6, 1.0)


def test_garbage_profile_disables_retry(monkeypatch):
    monkeypatch.setenv(sub._PRECOMMIT_RETRY_DELAYS_ENV, "0.3,banana")
    assert sub._precommit_retry_delays() == ()


def test_non_positive_delays_are_dropped(monkeypatch):
    monkeypatch.setenv(sub._PRECOMMIT_RETRY_DELAYS_ENV, "0,-1,0.4")
    assert sub._precommit_retry_delays() == (0.4,)


def test_margin_default_and_override(monkeypatch):
    monkeypatch.delenv(sub._PRECOMMIT_RETRY_MARGIN_ENV, raising=False)
    assert sub._precommit_retry_margin_s() == sub._DEFAULT_PRECOMMIT_RETRY_MARGIN_S
    monkeypatch.setenv(sub._PRECOMMIT_RETRY_MARGIN_ENV, "0.5")
    assert sub._precommit_retry_margin_s() == 0.5
    monkeypatch.setenv(sub._PRECOMMIT_RETRY_MARGIN_ENV, "nope")
    assert sub._precommit_retry_margin_s() == sub._DEFAULT_PRECOMMIT_RETRY_MARGIN_S


# ------------------------------------------------------- round-aware clamping
def test_sleep_is_clamped_to_the_signed_round(monkeypatch):
    """A 1.0s delay with only 0.8s of round left must not cross the boundary."""
    monkeypatch.setattr(sub, "_drand_round_headroom_s", lambda now=None: 0.8)
    assert sub._precommit_retry_sleep_s(1.0, 0.35) == pytest.approx(0.45)


def test_short_delay_is_used_verbatim_when_the_round_allows(monkeypatch):
    monkeypatch.setattr(sub, "_drand_round_headroom_s", lambda now=None: 2.6)
    assert sub._precommit_retry_sleep_s(0.3, 0.35) == pytest.approx(0.3)


def test_no_retry_when_the_round_is_about_to_flip(monkeypatch):
    """Headroom below the safety margin => give up, never gamble stale_round."""
    monkeypatch.setattr(sub, "_drand_round_headroom_s", lambda now=None: 0.2)
    assert sub._precommit_retry_sleep_s(0.3, 0.35) is None


def test_no_retry_when_the_drand_clock_is_unknown(monkeypatch):
    monkeypatch.setattr(sub, "_drand_round_headroom_s", lambda now=None: None)
    assert sub._precommit_retry_sleep_s(0.3, 0.35) is None


def test_upstream_backoff_would_be_rejected_at_median_headroom(monkeypatch):
    """Measured median headroom at a batch_filled is 1.16s.

    Upstream's first delay is 1.0s; with our margin that does not fit, which is
    exactly why the profile is clamped instead of copied.
    """
    monkeypatch.setattr(sub, "_drand_round_headroom_s", lambda now=None: 1.16)
    assert sub._precommit_retry_sleep_s(1.0, 0.35) == pytest.approx(0.81)
    # ...and 2.0s / 4.0s can never fit in a 3s round with any real margin.
    monkeypatch.setattr(sub, "_drand_round_headroom_s", lambda now=None: 2.9)
    assert sub._precommit_retry_sleep_s(2.0, 0.35) == pytest.approx(2.0)
    monkeypatch.setattr(sub, "_drand_round_headroom_s", lambda now=None: 1.0)
    assert sub._precommit_retry_sleep_s(4.0, 0.35) == pytest.approx(0.65)


def test_headroom_helper_matches_the_validator_clock(monkeypatch):
    """Genesis/period arithmetic must match compute_current_drand_round."""
    monkeypatch.setattr(
        sub, "_drand_round_headroom_s", sub._drand_round_headroom_s,
    )
    import reliquary.infrastructure.drand as drand

    monkeypatch.setattr(
        drand, "get_current_chain",
        lambda: {"genesis_time": 1692803367, "period": 3},
    )
    # 1.25s into a round leaves 1.75s.
    assert sub._drand_round_headroom_s(1692803367 + 3 * 100 + 1.25) == pytest.approx(1.75)
    # Exactly on a boundary reports 0.0 -> no retry budget.
    assert sub._drand_round_headroom_s(1692803367 + 3 * 100) == pytest.approx(0.0)


# ------------------------------------------------------------ end-to-end path
class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload


class _FakeClient:
    """Returns a scripted sequence of precommit verdicts, then a reveal OK."""

    def __init__(self, precommit_payloads):
        self._precommits = list(precommit_payloads)
        self.precommit_calls = 0
        self.bodies: list[bytes] = []

    async def post(self, url, content=None, headers=None, timeout=None):
        if url.endswith("/submit/precommit"):
            self.precommit_calls += 1
            self.bodies.append(content)
            return _Resp(self._precommits.pop(0))
        return _Resp({"accepted": True, "reason": "accepted"})

    async def aclose(self):
        return None


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


@pytest.fixture
def patched(monkeypatch):
    """Stub signing/serialisation so the test exercises the retry loop only."""
    class _PC:
        drand_round = 12345

        def model_dump_json(self):
            return '{"stub":1}'

    monkeypatch.setattr(
        sub, "_build_precommit", lambda request, **kw: (b"body", _PC()),
    )
    sleeps: list[float] = []

    async def _sleep(s):
        sleeps.append(s)

    monkeypatch.setattr(sub.asyncio, "sleep", _sleep)
    return sleeps


class _Req:
    window_start = 1
    prompt_idx = 7
    drand_round = 12345

    def model_dump(self, mode=None):
        return {}


def test_batch_filled_is_final_when_retry_is_off(patched, monkeypatch):
    monkeypatch.delenv(sub._PRECOMMIT_RETRY_DELAYS_ENV, raising=False)
    cli = _FakeClient([{"accepted": False, "reason": "batch_filled"}])
    out = _run(sub._submit_with_precommit(
        "http://v", _Req(), client=cli, timeout=5, wallet=object(),
        randomness="r",
    ))
    assert cli.precommit_calls == 1, "must not retry when the env var is unset"
    assert out.accepted is False
    assert out.reason is RejectReason.BATCH_FILLED
    assert patched == []


def test_batch_filled_retries_and_reuses_the_same_bytes(patched, monkeypatch):
    monkeypatch.setenv(sub._PRECOMMIT_RETRY_DELAYS_ENV, "0.3,0.6")
    monkeypatch.setattr(sub, "_drand_round_headroom_s", lambda now=None: 2.5)
    cli = _FakeClient([
        {"accepted": False, "reason": "batch_filled"},
        {"accepted": True, "reason": "accepted", "receipt_id": "r1"},
    ])
    out = _run(sub._submit_with_precommit(
        "http://v", _Req(), client=cli, timeout=5, wallet=object(),
        randomness="r",
    ))
    assert cli.precommit_calls == 2
    assert patched == [pytest.approx(0.3)]
    # The retry MUST reuse the identical signed bytes, not a re-signed envelope.
    assert cli.bodies[0] == cli.bodies[1]
    assert out.accepted is True


def test_retry_stops_when_the_round_has_no_headroom(patched, monkeypatch):
    monkeypatch.setenv(sub._PRECOMMIT_RETRY_DELAYS_ENV, "0.3,0.6")
    monkeypatch.setattr(sub, "_drand_round_headroom_s", lambda now=None: 0.1)
    cli = _FakeClient([{"accepted": False, "reason": "batch_filled"}])
    out = _run(sub._submit_with_precommit(
        "http://v", _Req(), client=cli, timeout=5, wallet=object(),
        randomness="r",
    ))
    assert cli.precommit_calls == 1, "no room in the signed round => no retry"
    assert out.reason is RejectReason.BATCH_FILLED
    assert patched == []


def test_other_rejects_are_never_retried(patched, monkeypatch):
    monkeypatch.setenv(sub._PRECOMMIT_RETRY_DELAYS_ENV, "0.3,0.6")
    monkeypatch.setattr(sub, "_drand_round_headroom_s", lambda now=None: 2.5)
    for reason in ("stale_round", "precommit_expired", "window_mismatch"):
        cli = _FakeClient([{"accepted": False, "reason": reason}])
        out = _run(sub._submit_with_precommit(
            "http://v", _Req(), client=cli, timeout=5, wallet=object(),
            randomness="r",
        ))
        assert cli.precommit_calls == 1, reason
        assert out.reason is RejectReason(reason)


def test_retry_budget_is_bounded(patched, monkeypatch):
    monkeypatch.setenv(sub._PRECOMMIT_RETRY_DELAYS_ENV, "0.3,0.6")
    monkeypatch.setattr(sub, "_drand_round_headroom_s", lambda now=None: 2.5)
    cli = _FakeClient([{"accepted": False, "reason": "batch_filled"}] * 3)
    out = _run(sub._submit_with_precommit(
        "http://v", _Req(), client=cli, timeout=5, wallet=object(),
        randomness="r",
    ))
    assert cli.precommit_calls == 3, "1 initial + exactly 2 retries"
    assert out.reason is RejectReason.BATCH_FILLED
