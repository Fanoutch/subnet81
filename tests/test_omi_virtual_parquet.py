"""OMI env must address the full 14M dataset via VirtualParquetDataset.

Upstream f49ccd6 dropped the shard cap: the validator's ``len(env)`` is the
full dataset and ``window_prompt_range`` derives from it. A shard-capped miner
(universe 1.76M vs 14M) derives a different per-window slice the day
``PROMPT_RANGE_ENFORCE_FROM_WINDOW`` is armed -> 100% PROMPT_OUT_OF_RANGE.
These tests pin the switch + keep our eligible-sources selection optimization
(absolute indices, computed from a bounded prefix scan, disk-cached).
"""
import json

import pytest

import reliquary.environment.openmathinstruct as omi
from reliquary.environment.openmathinstruct import OpenMathInstructEnvironment


class _FakeDS:
    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, i):
        return self._rows[i % len(self._rows)]


def _env(monkeypatch, rows, eligible=None):
    monkeypatch.setattr(OpenMathInstructEnvironment, "_dataset_cache", _FakeDS(rows))
    monkeypatch.setattr(
        OpenMathInstructEnvironment, "_eligible_indices_cache", eligible
    )
    # Neither the dataset nor the eligible scan may touch the network here.
    monkeypatch.setenv("RELIQUARY_OMI_SOURCES", "")
    return OpenMathInstructEnvironment()


# ---- dataset loading: virtual parquet, pinned revision ----------------------

def test_default_repo_and_revision_match_validator():
    # Same pins as upstream origin/main (ca3ac67) — required for byte-identical
    # len(env) and idx->problem mapping.
    assert OpenMathInstructEnvironment._OMI_REPO == "nvidia/OpenMathInstruct-2"
    assert (
        OpenMathInstructEnvironment._OMI_REVISION
        == "469216e3f46f4dacf476b382e192485ea51a143e"
    )


def test_load_dataset_wraps_repo_in_virtual_parquet(monkeypatch):
    captured = {}

    class _FakeVPD:
        def __init__(self, repo, revision, *, columns=None, **kw):
            captured.update(repo=repo, revision=revision, columns=columns)

    monkeypatch.setattr(
        "reliquary.environment.virtual_parquet.VirtualParquetDataset", _FakeVPD
    )
    ds = omi._load_dataset("nvidia/OpenMathInstruct-2", "deadbeef")
    assert isinstance(ds, _FakeVPD)
    assert captured["repo"] == "nvidia/OpenMathInstruct-2"
    assert captured["revision"] == "deadbeef"
    assert set(captured["columns"]) == {"problem", "expected_answer"}


def test_len_reflects_dataset_not_shard_cap(monkeypatch):
    env = _env(monkeypatch, [{"problem": f"p{i}", "expected_answer": str(i)}
                             for i in range(7)])
    assert len(env) == 7


def test_get_problem_reads_row_with_modulo(monkeypatch):
    rows = [{"problem": "1+1?", "expected_answer": "2"},
            {"problem": "2+2?", "expected_answer": "4"}]
    env = _env(monkeypatch, rows)
    p = env.get_problem(3)  # 3 % 2 = 1
    assert p["prompt"].startswith("2+2?")
    assert "\\boxed{}" in p["prompt"]
    assert p["ground_truth"] == "4"


# ---- eligible-sources optimization over the 14M universe --------------------

def test_sources_empty_disables_eligible(monkeypatch):
    env = _env(monkeypatch, [{"problem": "p", "expected_answer": "1"}])
    assert env.eligible_indices is None


def test_eligible_scan_builds_absolute_indices(monkeypatch, tmp_path):
    # Two fake shards of problem_source columns; allowed source appears at
    # absolute indices 1 (shard 0) and 2 (shard 1, offset 2).
    monkeypatch.setattr(
        omi, "_iter_problem_source_columns",
        lambda repo, revision, max_shards: iter(
            [["augmented_math", "augmented_gsm8k"],
             ["augmented_gsm8k", "math"]][:max_shards]
        ),
    )
    got = omi._scan_eligible_indices(
        "r/r", "rev0", {"augmented_gsm8k"}, max_shards=2, cache_dir=tmp_path
    )
    assert got == [1, 2]


def test_eligible_scan_uses_disk_cache(monkeypatch, tmp_path):
    calls = {"n": 0}

    def _fake_iter(repo, revision, max_shards):
        calls["n"] += 1
        return iter([["augmented_gsm8k"]])

    monkeypatch.setattr(omi, "_iter_problem_source_columns", _fake_iter)
    a = omi._scan_eligible_indices(
        "r/r", "rev1", {"augmented_gsm8k"}, max_shards=1, cache_dir=tmp_path
    )
    b = omi._scan_eligible_indices(
        "r/r", "rev1", {"augmented_gsm8k"}, max_shards=1, cache_dir=tmp_path
    )
    assert a == b == [0]
    assert calls["n"] == 1  # second call served from disk cache
    cached = list(tmp_path.glob("*.json"))
    assert cached and json.loads(cached[0].read_text()) == [0]
