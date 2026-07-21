from reliquary.protocol.submission import BatchSubmissionRequest, RejectReason


def test_seed_mismatch_reason_exists():
    assert RejectReason.SEED_MISMATCH.value == "seed_mismatch"


def test_protocol_version_field_defaults_zero():
    f = BatchSubmissionRequest.model_fields["protocol_version"]
    assert f.default == 0
