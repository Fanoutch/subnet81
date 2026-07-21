from reliquary.protocol.submission import Verdict, VerdictsResponse, RejectReason


def test_verdict_accepts_observability_fields():
    v = Verdict(
        merkle_root="a" * 64, accepted=True, reason=RejectReason.ACCEPTED, ts=1.0,
        rewarded=True, selected_for_batch=True, accepted_into_pool=True,
    )
    assert v.rewarded is True and v.selected_for_batch is True


def test_verdict_back_compat_without_fields():
    v = Verdict(merkle_root="b" * 64, accepted=False, reason=RejectReason.GRAIL_FAIL, ts=2.0)
    assert v.rewarded is None and v.window_n is None


def test_verdicts_response_parses_enriched_payload():
    payload = {"verdicts": [{
        "merkle_root": "c" * 64, "accepted": True, "reason": "accepted", "ts": 3.0,
        "rewarded": True, "selected_for_batch": False, "queue_wait_ms": 12.5,
    }]}
    resp = VerdictsResponse.model_validate(payload)
    assert resp.verdicts[0].rewarded is True
    assert resp.verdicts[0].selected_for_batch is False
