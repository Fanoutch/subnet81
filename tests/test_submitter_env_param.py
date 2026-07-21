from reliquary.miner.submitter import build_state_url


def test_build_state_url_no_env():
    assert build_state_url("http://v:8080") == "http://v:8080/state"


def test_build_state_url_with_env():
    assert build_state_url("http://v:8080", "opencodeinstruct") == \
        "http://v:8080/state?env=opencodeinstruct"


def test_build_state_url_encodes():
    assert build_state_url("http://v:8080", "a b") == "http://v:8080/state?env=a%20b"
