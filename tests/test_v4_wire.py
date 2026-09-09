"""Task 7 du port v4 : identité wire — 16 rollouts, domaine v4, caps.

Sous ``RELIQUARY_PROTOCOL_VERSION=4`` le modèle de soumission exige 16
rollouts et les constantes wire émettent version 4 / domaine
``reliquary-forced-seed-v4`` (dérivés, une seule source).
"""
import importlib

import pytest

from tests.v4helpers import reload_constants


@pytest.fixture(autouse=True)
def _restore(monkeypatch):
    yield
    reload_constants(monkeypatch)
    import reliquary.protocol.submission as s

    importlib.reload(s)


def test_v4_submission_requires_16_rollouts(monkeypatch):
    reload_constants(monkeypatch, 4)
    import reliquary.protocol.submission as s

    importlib.reload(s)
    assert s.M_ROLLOUTS == 16


def test_v3_submission_still_8_rollouts(monkeypatch):
    reload_constants(monkeypatch)
    import reliquary.protocol.submission as s

    importlib.reload(s)
    assert s.M_ROLLOUTS == 8


def test_v4_wire_identity(monkeypatch):
    c = reload_constants(monkeypatch, 4)
    assert c.FORCED_SEED_PROTOCOL_VERSION == 4
    assert c.FORCED_SEED_DOMAIN == "reliquary-forced-seed-v4"
    assert c.MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW == 32
    assert c.B_BATCH == 16
