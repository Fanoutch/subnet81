"""Audit v4 item 6 : budget de troncature par soumission (1 math / 3 code).

Le validateur (main ET v4) tolère par groupe jusqu'à 1 rollout « truncated »
(cap atteint, zéro EOS) — 3 en code. Notre garde locale v3 est volontairement
tout-ou-rien (fix 2026-08-05) et NE CHANGE PAS ; sous v4 (sans BFT, cap 8192
atteint honnêtement) elle jetterait des groupes payables → partition
bad/truncated + budget. EOS mid-stream/multiple = bad dans tous les cas.
"""
import importlib

import pytest

from tests.v4helpers import reload_constants

EOS = (151643, 151645)


def _reload_engine(monkeypatch, version=None):
    reload_constants(monkeypatch, version)
    import reliquary.miner.engine as e

    importlib.reload(e)
    return e


@pytest.fixture(autouse=True)
def _restore(monkeypatch):
    yield
    _reload_engine(monkeypatch)


def test_constants_budget_parity_upstream(monkeypatch):
    # Valeurs upstream main==v4 (constants.py:264-292) — non gatées là-bas.
    c = reload_constants(monkeypatch)
    assert c.MAX_TRUNCATED_PER_SUBMISSION == 1
    assert c.MAX_TRUNCATED_PER_SUBMISSION_BY_ENV == {"opencodeinstruct": 3}
    assert c.max_truncated_for_environment("openmathinstruct") == 1
    assert c.max_truncated_for_environment("opencodeinstruct") == 3
    assert c.max_truncated_for_environment("opencodeinstruct", bootstrap=True) == 1


def test_partition_bad_vs_truncated(monkeypatch):
    e = _reload_engine(monkeypatch, 4)
    cap = 6
    ok = [1, 2, 3, EOS[0]]                # un EOS, final → ok
    trunc = [1, 2, 3, 4, 5, 6]            # zéro EOS, longueur == cap → truncated
    bad_short = [1, 2, 3]                 # zéro EOS, sous le cap → bad
    bad_mid = [1, EOS[0], 3, EOS[0]]      # EOS mid-stream → bad
    n_bad, n_trunc = e.termination_partition(
        [ok, trunc, bad_short, bad_mid], EOS, cap
    )
    assert (n_bad, n_trunc) == (2, 1)


def test_v4_prebake_guard_semantics(monkeypatch):
    # La décision de drop de _pre_bake_entry est extraite dans
    # should_drop_for_termination : v4 = bad>0 OU truncated>budget(env).
    e = _reload_engine(monkeypatch, 4)
    cap = 6
    ok = [1, 2, 3, EOS[0]]
    trunc = [1, 2, 3, 4, 5, 6]
    # math : 1 tronqué toléré, 2 = drop
    assert not e.should_drop_for_termination(
        [ok, trunc], EOS, "openmathinstruct", cap
    )
    assert e.should_drop_for_termination(
        [ok, trunc, trunc], EOS, "openmathinstruct", cap
    )
    # code : 3 tolérés, 4 = drop
    assert not e.should_drop_for_termination(
        [ok, trunc, trunc, trunc], EOS, "opencodeinstruct", cap
    )
    assert e.should_drop_for_termination(
        [ok, trunc, trunc, trunc, trunc], EOS, "opencodeinstruct", cap
    )
    # un seul bad (EOS mid-stream) = drop immédiat
    assert e.should_drop_for_termination(
        [ok, [1, EOS[0], 3]], EOS, "opencodeinstruct", cap
    )


def test_v3_guard_stays_all_or_nothing(monkeypatch):
    e = _reload_engine(monkeypatch)
    cap = 6
    ok = [1, 2, 3, EOS[0]]
    trunc = [1, 2, 3, 4, 5, 6]
    # v3 : le moindre rollout sans EOS final = drop (comportement 2026-08-05)
    assert e.should_drop_for_termination([ok, trunc], EOS, "opencodeinstruct", cap)
    assert not e.should_drop_for_termination([ok, ok], EOS, "opencodeinstruct", cap)


def test_v4_too_many_truncated_uses_validator_budget(monkeypatch):
    e = _reload_engine(monkeypatch, 4)
    # code : 3 non-terminés sur 16 tolérés, 4 = drop
    assert not e.too_many_truncated(16, 13, "opencodeinstruct")
    assert e.too_many_truncated(16, 12, "opencodeinstruct")
    # math sans BFT : budget 1
    assert not e.too_many_truncated(16, 15, "openmathinstruct")
    assert e.too_many_truncated(16, 14, "openmathinstruct")


def test_v3_too_many_truncated_unchanged(monkeypatch):
    e = _reload_engine(monkeypatch)
    # code v3 : défaut local 0 toléré
    assert e.too_many_truncated(8, 7, "opencodeinstruct")
    assert not e.too_many_truncated(8, 8, "opencodeinstruct")
    # math v3 (BFT) : règle legacy — drop seulement si RIEN n'a terminé
    assert not e.too_many_truncated(8, 1, "openmathinstruct")
    assert e.too_many_truncated(8, 0, "openmathinstruct")
