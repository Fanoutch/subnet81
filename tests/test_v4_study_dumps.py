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
    # la source RESTE lisible (get, pas pop) : le writer B5 des soumissions
    # la relit après le dump (fix 18/08 — elle sortait None dans submits_v4)
    e.dump_group_sample(
        prompt="p", prompt_idx=42, rewards=[1.0] * 3 + [0.0] * 13,
        env_name="opencodeinstruct", window_n=30002,
    )
    row2 = json.loads(out.read_text().splitlines()[1])
    assert row2["source"] == "memo"
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


def test_b3_verdict_dump_persists_full_verdict(monkeypatch, tmp_path):
    """B3 : le dump verdicts doit porter TOUT le Verdict (H4 départage arrivée,
    H9 courses, H10 seal via seal_trigger_round), pas un sous-ensemble."""
    import asyncio

    from reliquary.miner.engine import MiningEngine
    import reliquary.miner.engine as eng
    from reliquary.miner.mix_controller import MixController
    from reliquary.protocol.submission import VerdictsResponse

    e = object.__new__(MiningEngine)
    e.active_envs = ["math", "code"]
    e._mix = MixController(["math", "code"], total_slots=8, slot_floor=1,
                           alpha=1.0)
    e._submitted_env = {"a" * 64: "code"}
    e._verdicts_since = 0.0

    class _W:
        class hotkey:
            ss58_address = "5HotkeyStub"
    e.wallet = _W()

    payload = {"verdicts": [{
        "merkle_root": "a" * 64, "window_n": 30001, "accepted": True,
        "reason": "accepted", "ts": 4.0, "rewarded": True,
        "canonical_rank": 3, "selected_for_batch": True,
        "seal_trigger_round": 29000123, "submitted_drand_round": 29000100,
        "arrival_drand_round": 29000101, "arrival_ts": 3.5,
        "accepted_into_pool": True, "prompt_hash_lead": "beef",
        "queue_wait_ms": 12.5,
    }]}
    resp = VerdictsResponse.model_validate(payload)

    async def fake_fetch(url, hotkey, *, client, since=None):
        return resp

    monkeypatch.setattr(eng, "fetch_verdicts", fake_fetch, raising=False)
    out = tmp_path / "verdicts.jsonl"
    monkeypatch.setenv("RELIQUARY_VERDICTS_DUMP", str(out))
    asyncio.run(e._tick_verdicts("http://v", client=None))
    monkeypatch.delenv("RELIQUARY_VERDICTS_DUMP")

    row = json.loads(out.read_text().splitlines()[0])
    # contrat existant conservé
    assert row["env"] == "code"
    assert row["verdict_ts"] == 4.0
    assert row["reason"] == "accepted"
    assert row["canonical_rank"] == 3
    # champs jusqu'ici perdus par le writer
    assert row["seal_trigger_round"] == 29000123
    assert row["submitted_drand_round"] == 29000100
    assert row["arrival_drand_round"] == 29000101
    assert row["arrival_ts"] == 3.5
    assert row["accepted_into_pool"] is True
    assert row["prompt_hash_lead"] == "beef"
    assert row["queue_wait_ms"] == 12.5


def test_note_pick_source_bounded(monkeypatch):
    e = _reload_engine(monkeypatch)
    for i in range(5000):
        e.note_pick_source(i, "scan")
    assert len(e._PICK_SOURCE) <= 4096
    e.note_pick_source("pas-un-int", "x")  # non-fatal
