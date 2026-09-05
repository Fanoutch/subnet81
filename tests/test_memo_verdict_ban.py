"""Le mémo apprend des verdicts du validateur (04/09).

Mesuré depuis 41602 : 35 rejets validateur out_of_zone/reward_mismatch sur 29
prompts, 6 récidives (2046715 rejeté 3×) — notre correcteur local les note en
zone (k 4-9), le validateur re-note hors zone, et le mémo les REJOUE car il ne
connaît que la note locale. Contrat : un tel verdict retire le prompt du mémo
ET le met en liste noire durable (la table mémo se reconstruit au restart
depuis le dump, où il est encore « en zone » ; seule la liste noire persiste).
"""
import json
from types import SimpleNamespace as NS

import pytest

from reliquary.miner import engine as eng
from reliquary.miner.payable_memo import get_memo


@pytest.fixture
def moteur(tmp_path, monkeypatch):
    monkeypatch.setenv("RELIQUARY_SZ_BLACKLIST", "1")
    monkeypatch.setenv("RELIQUARY_SZ_BLACKLIST_FILE", str(tmp_path / "bl.json"))
    e = eng.MiningEngine.__new__(eng.MiningEngine)
    e._cached_window_n = 41700
    e._submitted_env = {"m1": "opencodeinstruct", "m2": "opencodeinstruct", "m3": "opencodeinstruct"}
    e._submitted_prompt = {"m1": 2046715, "m2": 777, "m3": 888}
    e._mix = NS(record_outcome=lambda env, ok: None)
    m = get_memo(); m.clear()
    for p in (2046715, 777, 888):
        m.update(p, True, window_n=41690)
    return e


def _v(merkle, reason=None, rewarded=None):
    return NS(merkle_root=merkle, ts=1.0, accepted=reason is None,
              reason=(NS(value=reason) if reason else None), rewarded=rewarded)


def test_out_of_zone_validateur_retire_du_memo_et_liste_noire(moteur, tmp_path):
    moteur._apply_verdicts(NS(verdicts=[_v("m1", "out_of_zone"), _v("m2", "reward_mismatch"), _v("m3", None, rewarded=True)]))
    m = get_memo()
    assert m.top_in_range(0, 10**7, n=5) == [888]          # 2046715 et 777 sortis
    assert {2046715, 777} <= moteur._sz_active()            # bannis, actifs
    assert 888 not in moteur._sz_active()
    bl = json.load(open(tmp_path / "bl.json"))               # persisté
    assert bl["2046715"] >= 41700 + 1000


def test_autres_rejets_ne_bannissent_pas(moteur):
    moteur._apply_verdicts(NS(verdicts=[_v("m1", "stale_round"), _v("m2", "hash_duplicate")]))
    assert set(get_memo().top_in_range(0, 10**7, n=5)) == {2046715, 777, 888}
    assert moteur._sz_active() == set()


def test_merkle_inconnu_ignore(moteur):
    moteur._apply_verdicts(NS(verdicts=[_v("inconnu", "out_of_zone")]))
    assert set(get_memo().top_in_range(0, 10**7, n=5)) == {2046715, 777, 888}
