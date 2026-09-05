"""Task 6 du port v4 : BFT éteint sous v4 — aucun rollout ``forced``.

Le validateur v4 fail-close toute claim ``forced`` (upstream
``validate_force_span`` : sans BFT il n'y a pas de span sanctionné par le
profil → tampering). Audit engine 2026-08-17 : l'UNIQUE origine de
``forced=True`` est l'assemblage BFT (``reliquary/miner/bft.py``), gaté par
``bft_applicable()`` ; tous les autres sites (engine 3832/3988/4116/4950) ne
font que propager ``gen.get("forced", False)``. Ces tests verrouillent le
gate.
"""
import pytest

from tests.v4helpers import reload_constants


@pytest.fixture(autouse=True)
def _restore_constants(monkeypatch):
    yield
    reload_constants(monkeypatch)


def test_v4_bft_not_applicable_anywhere(monkeypatch):
    monkeypatch.setattr("reliquary.constants.BFT_ENABLED", False)
    from reliquary.miner import bft

    assert not bft.bft_applicable("openmathinstruct")
    assert not bft.bft_applicable("opencodeinstruct")
    assert not bft.bft_applicable(None)


def test_v3_bft_still_math_only(monkeypatch):
    monkeypatch.setattr("reliquary.constants.BFT_ENABLED", True)
    from reliquary.miner import bft

    assert bft.bft_applicable("openmathinstruct")
    assert bft.bft_applicable(None)
    assert not bft.bft_applicable("opencodeinstruct")


def test_v4_phase1_budget_is_full_cap(monkeypatch):
    # Sans BFT la phase-1 = le cap configuré (pas le thinking budget 15616).
    monkeypatch.setattr("reliquary.constants.BFT_ENABLED", False)
    from reliquary.miner.bft import phase1_max_new_tokens

    assert phase1_max_new_tokens(8192, "openmathinstruct") == 8192
    assert phase1_max_new_tokens(8192, "opencodeinstruct") == 8192
