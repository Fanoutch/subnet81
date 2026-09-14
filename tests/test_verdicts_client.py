from reliquary.miner.submitter import build_verdicts_url
from reliquary.protocol.submission import Verdict


# V1 (validateur 0a69244, #256/#261) : le cycle de vie détaillé
# (selection_status, outcome_code, proof_status…) n'est servi que sur
# ``details=true`` — sans lui on ne distingue pas « prompt déjà prouvé par
# un autre », « prouvé mais pas retenu » et « dette de preuve ».
def test_build_verdicts_url_no_since():
    assert build_verdicts_url("http://v:8080", "5Hdd...") == \
        "http://v:8080/verdicts/5Hdd...?details=true"


def test_build_verdicts_url_with_since():
    assert build_verdicts_url("http://v:8080", "5HddX", since=12.5) == \
        "http://v:8080/verdicts/5HddX?since=12.5&details=true"


def test_build_verdicts_url_encodes_hotkey():
    assert build_verdicts_url("http://v:8080", "a/b") == \
        "http://v:8080/verdicts/a%2Fb?details=true"


def test_detail_fields_are_kept_in_the_dump():
    v = Verdict.model_validate({
        "merkle_root": "ab" * 32, "window_n": 45829, "accepted": True,
        "reason": "accepted", "ts": 1.0,
        "stage": "final", "is_final": True, "selection_status": "not_selected",
        "outcome_code": "same_prompt_or_content_already_proven",
        "environment": "opencodeinstruct", "prompt_idx": 12,
        "proof_status": "skipped", "proof_reason": None,
        "ordering_policy": "fifo-ingress/v1", "body_received_ts": 2.0,
        "selection_target": 112, "selected_count": 112,
        "reason_details": {"k": 1}, "future_field": 1,
    })
    row = v.model_dump(mode="json")
    assert row["outcome_code"] == "same_prompt_or_content_already_proven"
    assert row["selection_status"] == "not_selected"
    assert row["prompt_idx"] == 12
    assert row["ordering_policy"] == "fifo-ingress/v1"
    assert "future_field" not in row
