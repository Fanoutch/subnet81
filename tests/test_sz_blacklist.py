"""Liste noire d'observation des prompts σ=0 (31/08).

35 % des picks σ=0 etaient des recidivistes : un pick casse n'est jamais
soumis, donc n'entre dans aucun cooldown, et la table statique le represente.
La liste noire OBSERVE au lieu de predire — insensible a la derive.
Proprietes verrouillees : OFF par defaut = zero effet ; facile ecarte long,
dur ecarte court puis redevient eligible ; borne memoire.
"""
import pytest
from reliquary.miner.engine import sz_blacklist_note, sz_blacklist_active


def test_off_par_defaut_aucun_effet(monkeypatch):
    monkeypatch.delenv("RELIQUARY_SZ_BLACKLIST", raising=False)
    bl = {}
    assert sz_blacklist_note(bl, 100, 42, [1.0]*16) is None
    assert bl == {}
    assert sz_blacklist_active({7: 999}, 100) == set()


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv("RELIQUARY_SZ_BLACKLIST", "1")
    monkeypatch.setenv("RELIQUARY_SZ_BLACKLIST_FACILE_FEN", "20000")
    monkeypatch.setenv("RELIQUARY_SZ_BLACKLIST_DUR_FEN", "300")


def test_tout_reussi_ecarte_durablement(on):
    """mean≈1 = le modele a APPRIS le prompt — il ne desapprend pas."""
    bl = {}
    assert sz_blacklist_note(bl, 37900, 42, [1.0]*16) == "facile"
    assert 42 in sz_blacklist_active(bl, 37900)
    assert 42 in sz_blacklist_active(bl, 37900 + 19000)


def test_tout_rate_redevient_eligible(on):
    """mean≈0 = trop dur AUJOURD'HUI — le modele progresse, on re-essaiera."""
    bl = {}
    assert sz_blacklist_note(bl, 37900, 42, [0.0]*16) == "dur"
    assert 42 in sz_blacklist_active(bl, 37900)
    assert 42 in sz_blacklist_active(bl, 37900 + 299)
    assert 42 not in sz_blacklist_active(bl, 37900 + 300)


def test_rewards_fractionnaires_uniformes_cote_dur(on):
    """σ=0 a rewards fractionnaires (tous 0.5) : pas 'appris' — cote court."""
    bl = {}
    assert sz_blacklist_note(bl, 100, 7, [0.5]*16) == "dur"


def test_rewards_vides_ignores(on):
    bl = {}
    assert sz_blacklist_note(bl, 100, 7, []) is None
    assert bl == {}


def test_purge_borne_la_memoire(on):
    bl = {i: 50 for i in range(100_001)}          # tous expires a fen 100
    bl[999_999] = 10_000
    act = sz_blacklist_active(bl, 100)
    assert act == {999_999}
    assert len(bl) == 1, "les expires doivent etre purges au-dela de 100k"


def test_le_dernier_verdict_gagne(on):
    """Un prompt re-observe met a jour son expiration (derive = info fraiche)."""
    bl = {}
    sz_blacklist_note(bl, 100, 42, [0.0]*16)      # dur -> expire 400
    sz_blacklist_note(bl, 200, 42, [1.0]*16)      # appris -> expire 20200
    assert 42 in sz_blacklist_active(bl, 15_000)


def test_persistance_sauvegarde_et_recharge(on, tmp_path, monkeypatch):
    """La liste survit a un restart : sauvegardee a chaque note, rechargee
    au premier acces du nouveau process (le watchdog redemarre ~1x/h)."""
    import json
    f = tmp_path / "sz.json"
    monkeypatch.setenv("RELIQUARY_SZ_BLACKLIST_FILE", str(f))

    class Moteur:
        _cached_window_n = 500
    from reliquary.miner.engine import MiningEngine as MinerEngine
    m = Moteur()
    m._sz_note = MinerEngine._sz_note.__get__(m)
    m._sz_load = MinerEngine._sz_load.__get__(m)
    m._sz_save = MinerEngine._sz_save.__get__(m)
    m._sz_active = MinerEngine._sz_active.__get__(m)
    m._sz_note(42, [1.0] * 16)
    assert f.exists() and json.load(open(f)) == {"42": 20500}

    m2 = Moteur()                      # « nouveau process »
    m2._sz_load = MinerEngine._sz_load.__get__(m2)
    m2._sz_active = MinerEngine._sz_active.__get__(m2)
    assert 42 in m2._sz_active()


def test_persistance_fichier_corrompu_repart_vide(on, tmp_path, monkeypatch):
    f = tmp_path / "sz.json"; f.write_text("{pas du json")
    monkeypatch.setenv("RELIQUARY_SZ_BLACKLIST_FILE", str(f))
    from reliquary.miner.engine import MiningEngine as MinerEngine
    class Moteur: _cached_window_n = 500
    m = Moteur()
    m._sz_load = MinerEngine._sz_load.__get__(m)
    m._sz_active = MinerEngine._sz_active.__get__(m)
    assert m._sz_active() == set()     # jamais fatal
