"""End-to-end on REAL data (requires network). Bounded scan for speed."""
import pytest

from scripts.build_opencode_cases import build
from reliquary.environment.code_grader import grade_completion


@pytest.mark.network
def test_real_join_and_grade(tmp_path):
    out = tmp_path / "cases.json"
    # Bounded scan: mirror ids are dense at the start of NVIDIA (PoC: 4/7).
    cases_map = build(out_path=str(out), scan_cap=150_000)
    assert len(cases_map) > 1000  # recovered a large share of the mirror

    fid = "e7ca4436b5c004b2c07534b50b1e4c83"  # factorial (known good id)
    assert fid in cases_map
    good = "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)"
    assert grade_completion(good, cases_map[fid], timeout_s=5) == 1.0
