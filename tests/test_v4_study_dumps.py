"""Instrumentation étude v4 (etudev4.md §B) : B1 complété + writers B2-B5.

Contrat : hors chemin de décision, jamais d'exception, gaté par env var.
"""
import importlib
import json

import pytest

from tests.v4helpers import reload_constants


def _reload_engine(monkeypatch, version=None):
    reload_constants(monkeypatch, version)
    import reliquary.miner.engine as e

    importlib.reload(e)
    return e


@pytest.fixture(autouse=True)
def _restore(monkeypatch):
    yield
    _reload_engine(monkeypatch)


def test_b1_dump_has_k_ckpt_proto_cap_source(monkeypatch, tmp_path):
    e = _reload_engine(monkeypatch, 4)
    out = tmp_path / "dump.jsonl"
    monkeypatch.setenv("RELIQUARY_SAMPLE_DUMP", str(out))
    e.note_pick_source(42, "memo")
    e.dump_group_sample(
        prompt="p", prompt_idx=42, rewards=[1.0] * 3 + [0.0] * 13,
        env_name="opencodeinstruct", window_n=30001, checkpoint_n=17,
    )
    row = json.loads(out.read_text().splitlines()[0])
    assert row["k"] == 3
    assert row["checkpoint_n"] == 17
    assert row["protocol_version"] == 4
    assert row["cap"] == 8192
    assert row["source"] == "memo"
    # la source est consommée (pop) — un 2e dump du même idx = None
    e.dump_group_sample(
        prompt="p", prompt_idx=42, rewards=[1.0] * 3 + [0.0] * 13,
        env_name="opencodeinstruct", window_n=30002,
    )
    row2 = json.loads(out.read_text().splitlines()[1])
    assert row2["source"] is None
    monkeypatch.delenv("RELIQUARY_SAMPLE_DUMP")


def test_study_dump_writes_jsonl_and_never_raises(monkeypatch, tmp_path):
    e = _reload_engine(monkeypatch)
    out = tmp_path / "study.jsonl"
    monkeypatch.setenv("RELIQUARY_TEST_STUDY", str(out))
    e.study_dump("RELIQUARY_TEST_STUDY", {"window_n": 1, "env": "x"})
    row = json.loads(out.read_text().splitlines()[0])
    assert row["window_n"] == 1 and "ts" in row
    # env var absente = no-op ; chemin invalide = silencieux
    e.study_dump("RELIQUARY_ABSENT_VAR", {"a": 1})
    monkeypatch.setenv("RELIQUARY_TEST_STUDY", "/proc/impossible/x.jsonl")
    e.study_dump("RELIQUARY_TEST_STUDY", {"a": 1})  # ne doit pas lever
    monkeypatch.delenv("RELIQUARY_TEST_STUDY")


def test_note_pick_source_bounded(monkeypatch):
    e = _reload_engine(monkeypatch)
    for i in range(5000):
        e.note_pick_source(i, "scan")
    assert len(e._PICK_SOURCE) <= 4096
    e.note_pick_source("pas-un-int", "x")  # non-fatal
