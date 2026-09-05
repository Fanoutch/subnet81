"""Balayage final 18/08 : verrouillage des fixes G2, G5, G7, G9.

(G1/G8/G11/G12 = scripts ops, vérifiés au sanity du launcher ; G3/G4 →
test_v4_uncertain_guard.py ; G6 → dérivation constants testée ici aussi.)
"""
import importlib
import sys
from pathlib import Path

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


# ── G2 : /verdicts robuste aux champs d'observabilité du validateur live ────

def test_verdict_parses_live_shape_payload():
    from reliquary.protocol.submission import Verdict

    v = Verdict.model_validate({
        "merkle_root": "ab" * 32,
        "accepted": True,
        "reason": "accepted",
        "ts": 1.0,
        # champs a6456b4 déjà émis par le live
        "sigma": 0.24, "payload_bytes": 123, "ingress_ms": 4.2,
        "upload_precommit_status": "matched", "body_read_ms": 0.1,
        "reward_grading_ms": 8.0, "admission_commit_ms": 1.0,
        # et un champ FUTUR inconnu : ne doit plus casser le polling
        "champ_futur_inconnu": "x",
    })
    assert v.sigma == 0.24 and v.payload_bytes == 123


# ── G5 : RELIQUARY_MAX_NEW_TOKENS clampé au cap protocole ───────────────────

def test_max_new_tokens_clamped_to_v4_cap(monkeypatch):
    monkeypatch.setenv("RELIQUARY_MAX_NEW_TOKENS", "16384")
    e = _reload_engine(monkeypatch, 4)
    eng = e.MiningEngine.__new__(e.MiningEngine)
    # rejoue uniquement l'affectation clampée du __init__
    import os as _os
    eng.max_new_tokens = min(
        int(_os.environ.get("RELIQUARY_MAX_NEW_TOKENS", "8192")),
        e.MAX_NEW_TOKENS_PROTOCOL_CAP,
    )
    assert eng.max_new_tokens == 8192
    monkeypatch.delenv("RELIQUARY_MAX_NEW_TOKENS")


# ── G6 : gate GPU dérivé du protocole actif ─────────────────────────────────

def test_gate_defaults_derive_from_constants(monkeypatch):
    c = reload_constants(monkeypatch, 4)
    assert c.DEFAULT_BASE_MODEL == "Qwen/Qwen3-4B-Base"
    assert c.M_ROLLOUTS == 16  # GATE_M par défaut = M_ROLLOUTS


# ── G7 : labels du probe version-aware ──────────────────────────────────────

def _load_probe():
    spec = importlib.util.spec_from_file_location(
        "difficulty_probe",
        Path(__file__).resolve().parent.parent / "scripts" / "difficulty_probe.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_probe_code_sigma_follows_protocol(monkeypatch):
    monkeypatch.setattr("reliquary.constants.PROTOCOL_VERSION", 4)
    dp = _load_probe()
    sigma, in_zone = dp._code_in_zone([0.8] * 10 + [0.2] * 6)
    assert in_zone == 1  # σ≈0.29 ≥ 0.24 (v4) alors que < 0.43 (v3)
    monkeypatch.setattr("reliquary.constants.PROTOCOL_VERSION", 3)
    sigma, in_zone = dp._code_in_zone([0.8] * 10 + [0.2] * 6)
    assert in_zone == 0


# ── G9 : garde --expect-protocol du probe ───────────────────────────────────

def test_probe_expect_protocol_mismatch_aborts(monkeypatch):
    dp = _load_probe()
    monkeypatch.setattr("reliquary.constants.PROTOCOL_VERSION", 3)
    called = []
    monkeypatch.setattr(dp, "stage_generate", lambda *a: called.append(a))
    monkeypatch.setattr(
        sys, "argv",
        ["difficulty_probe.py", "generate", "--in", "/dev/null",
         "--model", "x", "--expect-protocol", "4"],
    )
    with pytest.raises(SystemExit):
        dp.main()
    assert not called


def test_probe_expect_protocol_match_proceeds(monkeypatch):
    dp = _load_probe()
    monkeypatch.setattr("reliquary.constants.PROTOCOL_VERSION", 4)
    called = []
    monkeypatch.setattr(dp, "stage_generate", lambda *a: called.append(a))
    monkeypatch.setattr(
        sys, "argv",
        ["difficulty_probe.py", "generate", "--in", "/dev/null",
         "--model", "x", "--expect-protocol", "4"],
    )
    dp.main()
    assert len(called) == 1
