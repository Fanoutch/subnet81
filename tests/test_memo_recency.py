"""Filtre de récence du slot mémo (RELIQUARY_MEMO_RESUBMIT_GAP_FEN, 03/09).

Mesure du 02/09 (150 fen archivées) : les rejeux mémo perdent ~0,4-0,5
entrée/fen en content_in_cooldown (30) + hash_duplicate différés (16) +
same_prompt_superseded (23) — tous des symptômes du MÊME défaut : rejouer un
prompt soumis trop récemment. Le filtre exclut du slot mémo tout prompt que
NOUS avons soumis dans les N dernières fenêtres. Défaut 0 = OFF (inchangé).
"""
import types

from reliquary.miner.engine import MiningEngine


def _eng():
    e = MiningEngine.__new__(MiningEngine)
    return e


def test_recents_exclus(monkeypatch):
    monkeypatch.setenv("RELIQUARY_MEMO_RESUBMIT_GAP_FEN", "40")
    e = _eng()
    e._last_submit_win = {101: 995, 102: 940, 103: 990}
    r = e._memo_recent_exclude(now_window=1000)
    assert r == {101, 103}, "soumis il y a <40 fen -> exclus ; 102 (60 fen) libre"


def test_defaut_zero_inchange(monkeypatch):
    monkeypatch.delenv("RELIQUARY_MEMO_RESUBMIT_GAP_FEN", raising=False)
    e = _eng()
    e._last_submit_win = {101: 999}
    assert e._memo_recent_exclude(now_window=1000) == set()


def test_purge_bornee(monkeypatch):
    monkeypatch.setenv("RELIQUARY_MEMO_RESUBMIT_GAP_FEN", "40")
    e = _eng()
    e._last_submit_win = {i: 0 for i in range(30000)}
    e._last_submit_win[7] = 999
    e._memo_recent_exclude(now_window=1000)
    assert len(e._last_submit_win) < 25000, "les entrées antiques sont purgées"
    assert e._last_submit_win.get(7) == 999


def test_sans_etat_ok(monkeypatch):
    monkeypatch.setenv("RELIQUARY_MEMO_RESUBMIT_GAP_FEN", "40")
    e = _eng()
    assert e._memo_recent_exclude(now_window=1000) == set()
