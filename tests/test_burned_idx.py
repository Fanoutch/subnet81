"""Veto des index brûlés par CONTENU (04/09).

Le validateur garde un cooldown de contenu à vie (1e6 fenêtres) : un texte
sélectionné une fois, sous n'importe quel index, est mort. /state n'expose que
le cooldown par index ; les jumeaux de texte (4,5 % du dataset) passent notre
filtre et meurent `content_in_cooldown` — 8 % de nos candidats, 129/129 prédits
par l'instantané R2. Le fichier .npy (index brûlés, calculé hors mineur) est
relu quand il change ; absent ou illisible = aucun effet.
"""
import numpy as np
import pytest

from reliquary.miner import engine as eng
from reliquary.miner.payable_memo import PayableMemo


def test_burned_off_when_unset(monkeypatch):
    monkeypatch.delenv("RELIQUARY_BURNED_IDX", raising=False)
    e = eng.MiningEngine.__new__(eng.MiningEngine)
    assert e._burned_active() == frozenset()


def test_burned_load_and_reload(tmp_path, monkeypatch):
    p = tmp_path / "burned.npy"
    np.save(p, np.array([3, 7, 11], dtype=np.int64))
    monkeypatch.setenv("RELIQUARY_BURNED_IDX", str(p))
    e = eng.MiningEngine.__new__(eng.MiningEngine)
    assert e._burned_active() == frozenset({3, 7, 11})
    # même mtime → cache ; nouveau fichier plus récent → rechargé
    np.save(p, np.array([42], dtype=np.int64))
    import os
    st = os.stat(p)
    os.utime(p, (st.st_atime + 5, st.st_mtime + 5))
    e._burned_checked_at = 0.0           # force le contrôle de mtime
    assert e._burned_active() == frozenset({42})


def test_burned_unreadable_is_empty(tmp_path, monkeypatch):
    p = tmp_path / "bad.npy"
    p.write_text("pas un npy")
    monkeypatch.setenv("RELIQUARY_BURNED_IDX", str(p))
    e = eng.MiningEngine.__new__(eng.MiningEngine)
    assert e._burned_active() == frozenset()


def test_memo_head_pick_respects_exclude_and_run():
    m = PayableMemo()
    m.update(10, True, window_n=41000)
    m.update(20, True, window_n=41001)
    m.update(30, True, window_n=31000)
    assert eng.memo_head_pick(m, (0, 100), exclude={20}, run_start=32791) == 10
    assert eng.memo_head_pick(m, (0, 100), exclude={10, 20}, run_start=32791) == 30
    assert eng.memo_head_pick(m, (0, 100), exclude={10, 20, 30}, run_start=32791) is None
    assert eng.memo_head_pick(m, None, exclude=set(), run_start=0) is None


def test_memo_head_slots_default_off(monkeypatch):
    monkeypatch.delenv("RELIQUARY_MEMO_HEAD_SLOTS", raising=False)
    assert eng._memo_head_slots() == 0
    monkeypatch.setenv("RELIQUARY_MEMO_HEAD_SLOTS", "2")
    assert eng._memo_head_slots() == 2


def test_picked_this_window_resets_on_new_window():
    """Bug 41590 (04/09) : un prompt de balayage gradé en zone devenait
    aussitôt le mémo le plus frais de la tranche et était REPRIS comme tête du
    bake suivant de la même fenêtre → doublon → same_prompt_superseded."""
    e = eng.MiningEngine.__new__(eng.MiningEngine)
    assert e._picked_this_window(100) == set()
    e._note_picked(100, [1, 2, 3])
    e._note_picked(100, [4])
    assert e._picked_this_window(100) == {1, 2, 3, 4}
    assert e._picked_this_window(101) == set()        # nouvelle fenêtre = vide
    e._note_picked(101, [9])
    assert e._picked_this_window(101) == {9}
