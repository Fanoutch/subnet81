from reliquary.miner.submitter import build_verdicts_url


def test_build_verdicts_url_no_since():
    assert build_verdicts_url("http://v:8080", "5Hdd...") == \
        "http://v:8080/verdicts/5Hdd..."


def test_build_verdicts_url_with_since():
    assert build_verdicts_url("http://v:8080", "5HddX", since=12.5) == \
        "http://v:8080/verdicts/5HddX?since=12.5"


def test_build_verdicts_url_encodes_hotkey():
    assert build_verdicts_url("http://v:8080", "a/b") == \
        "http://v:8080/verdicts/a%2Fb"
