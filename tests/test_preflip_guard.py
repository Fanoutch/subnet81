"""Garde pré-flip (2026-08-19) — le fix des picks tardifs.

Mesuré sur les 7 fenêtres du régime collection-100s : picks partis ≤5 s
après le flip → méd 5 admises, 0 fenêtre morte ; picks >5 s (5 fenêtres
sur 7 !) → méd 0 admise. Cause : le lot de collecte vLLM en vol au flip
(traînard p99 ~5000 tok = ~65 s) bloque la boucle de picks.

La garde borne le travail lancé en fin de cycle (cycle p10 = 333 s) :
  - avant LATE_BAKE_FROM (déf. 150 s) : lots pleins (labels prior riches) ;
  - LATE_BAKE_FROM → PREFLIP_GUARD_S (déf. 230 s) : lots BRIDÉS à
    LATE_BAKE_CAP tokens (déf. 1200 ≈ p90 → lot ≤ ~16 s) — ces lots sont
    post-collecte donc jamais soumis, le cap est sans enjeu de conformité,
    et la troncature est déjà étiquetée (n_truncated) donc exclue du prior ;
  - après PREFLIP_GUARD_S : AUCUN nouveau lot → GPU libre au flip,
    picks à +1 s.
"""
import os

import pytest

from reliquary.miner.engine import bake_guard_decision, late_bake_cap


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("RELIQUARY_LATE_BAKE_FROM", "RELIQUARY_PREFLIP_GUARD_S",
              "RELIQUARY_LATE_BAKE_CAP"):
        monkeypatch.delenv(k, raising=False)


def test_zones_defaults():
    assert bake_guard_decision(0) == "full"
    assert bake_guard_decision(149) == "full"
    assert bake_guard_decision(150) == "capped"
    assert bake_guard_decision(229) == "capped"
    assert bake_guard_decision(230) == "hold"
    assert bake_guard_decision(9999) == "hold"


def test_open_ts_unknown_is_full():
    # Au boot (pas encore de flip observé) : ne jamais bloquer le bake.
    assert bake_guard_decision(None) == "full"


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("RELIQUARY_LATE_BAKE_FROM", "60")
    monkeypatch.setenv("RELIQUARY_PREFLIP_GUARD_S", "90")
    assert bake_guard_decision(59) == "full"
    assert bake_guard_decision(60) == "capped"
    assert bake_guard_decision(90) == "hold"


def test_kill_switch_via_huge_values(monkeypatch):
    monkeypatch.setenv("RELIQUARY_LATE_BAKE_FROM", "999999")
    monkeypatch.setenv("RELIQUARY_PREFLIP_GUARD_S", "999999")
    assert bake_guard_decision(5000) == "full"


def test_late_bake_cap_default_and_env(monkeypatch):
    assert late_bake_cap() == 1200
    monkeypatch.setenv("RELIQUARY_LATE_BAKE_CAP", "800")
    assert late_bake_cap() == 800


def test_generate_honors_cap_override():
    """_generate_m_rollouts doit borner max_new au cap du lot bridé."""
    from reliquary.miner.bft import phase1_max_new_tokens
    # Le plumbing est : max_new = phase1(...) puis min(cap_override).
    # On teste la fonction de calcul du cap effectif, extraite pour ça.
    from reliquary.miner.engine import effective_gen_cap
    assert effective_gen_cap(8192, None) == 8192
    assert effective_gen_cap(8192, 1200) == 1200
    assert effective_gen_cap(600, 1200) == 600


def test_spec_proof_flag_and_slots(monkeypatch):
    from reliquary.miner.engine import (
        MiningEngine, spec_proof_enabled, spec_proof_slots,
    )
    monkeypatch.delenv("RELIQUARY_SPEC_PROOF", raising=False)
    assert spec_proof_enabled() is False  # défaut OFF, flip via launcher
    monkeypatch.setenv("RELIQUARY_SPEC_PROOF", "1")
    assert spec_proof_enabled() is True
    assert spec_proof_slots() == 4
    monkeypatch.setenv("RELIQUARY_SPEC_PROOF_SLOTS", "2")
    eng = MiningEngine.__new__(MiningEngine)
    assert eng._take_spec_proof_slot(100) is True
    assert eng._take_spec_proof_slot(100) is True
    assert eng._take_spec_proof_slot(100) is False   # quota épuisé
    assert eng._take_spec_proof_slot(101) is True    # nouvelle fenêtre
