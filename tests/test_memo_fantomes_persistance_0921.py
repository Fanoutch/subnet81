"""21/09 soir : les étiquettes fantômes survivent au restart.

Constat : au démarrage le mémo ne relisait que ``RELIQUARY_SAMPLE_DUMP``
(production). Les bakes fantômes écrivent dans ``RELIQUARY_GHOST_DUMP`` : le
restart de 17:09 a effacé ~875 étiquettes en zone du mémo, et la mémoire
« déjà étiqueté » des fantômes (qui ré-étiquetaient donc les mêmes prompts).

Ce que ces tests verrouillent :
1. Sans fichier fantôme (ou FEED=0), l'amorçage est IDENTIQUE à l'historique.
2. Fusion par ordre de fenêtre, chaque fichier gardé dans son ordre ; à fenêtre
   égale, le fantôme (qui tourne dans le trou, après la production) passe après.
3. Mêmes règles qu'en vol : fantôme en zone sans tronqué -> payable ; hors zone
   -> retiré du mémo ; en zone avec tronqué -> ignoré ; ``fed: false`` -> ignoré.
4. La mémoire « déjà étiqueté » est reconstruite depuis le fichier fantôme.
"""
from __future__ import annotations

import asyncio
import json
import random

from reliquary.miner import engine
from reliquary.miner import ghost_bake as gb
from reliquary.miner.payable_memo import PayableMemo

from tests.test_ghost_bake_0921 import _Env, _eng


def _write(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(r if isinstance(r, str) else json.dumps(r))
            fh.write("\n")
    return str(path)


def _row(idx, w, in_zone=True, trunc=0, **kw):
    r = {"prompt_idx": idx, "window_n": w, "in_zone": in_zone,
         "n_truncated": trunc, "completion_lens": [100] * 16}
    r.update(kw)
    return r


def _state(m: PayableMemo):
    return (set(m._payable), dict(m._conf), dict(m._last_w), dict(m._vol))


# ───────────────────────────── fusion dans le mémo ───────────────────────────
def test_sans_fantome_identique_a_l_historique(tmp_path):
    prod = _write(tmp_path / "s.jsonl", [
        _row(1, 10), _row(2, 11, in_zone=False), _row(3, 9), _row(1, 12),
        _row(4, None), "pas du json", _row(5, 13, trunc=2)])
    a, b = PayableMemo(), PayableMemo()
    a.load_jsonl(prod)
    b.load_merged(prod, str(tmp_path / "absent.jsonl"))
    assert _state(a) == _state(b)
    assert [k for k in a._payable] == [k for k in b._payable]
    assert a._payable == b._payable


def test_fantome_en_zone_ajoute(tmp_path):
    prod = _write(tmp_path / "s.jsonl", [_row(1, 10)])
    ghost = _write(tmp_path / "g.jsonl", [_row(7, 10, fed=True)])
    m = PayableMemo()
    m.load_merged(prod, ghost)
    assert set(m._payable) == {1, 7}
    assert m._last_w[7] == 10 and m._vol[7] == 1600


def test_fantome_hors_zone_retire_une_mesure_plus_ancienne(tmp_path):
    prod = _write(tmp_path / "s.jsonl", [_row(1, 9)])
    ghost = _write(tmp_path / "g.jsonl", [_row(1, 10, in_zone=False)])
    m = PayableMemo()
    m.load_merged(prod, ghost)
    assert 1 not in m._payable


def test_production_plus_recente_l_emporte(tmp_path):
    prod = _write(tmp_path / "s.jsonl", [_row(1, 12)])
    ghost = _write(tmp_path / "g.jsonl", [_row(1, 10, in_zone=False)])
    m = PayableMemo()
    m.load_merged(prod, ghost)
    assert 1 in m._payable                 # re-mesuré en zone APRÈS le fantôme
    prod2 = _write(tmp_path / "s2.jsonl", [_row(2, 12, in_zone=False)])
    ghost2 = _write(tmp_path / "g2.jsonl", [_row(2, 10)])
    m2 = PayableMemo()
    m2.load_merged(prod2, ghost2)
    assert 2 not in m2._payable            # re-mesuré hors zone après


def test_a_fenetre_egale_le_fantome_passe_apres(tmp_path):
    prod = _write(tmp_path / "s.jsonl", [_row(1, 10)])
    ghost = _write(tmp_path / "g.jsonl", [_row(1, 10, in_zone=False)])
    m = PayableMemo()
    m.load_merged(prod, ghost)
    assert 1 not in m._payable


def test_ordre_interne_des_fichiers_preserve(tmp_path):
    # une ligne de production sans fenêtre ou en désordre garde sa place
    prod = _write(tmp_path / "s.jsonl", [
        _row(1, 20), _row(1, 15, in_zone=False)])
    ghost = _write(tmp_path / "g.jsonl", [_row(9, 18)])
    m = PayableMemo()
    m.load_merged(prod, ghost)
    assert 1 not in m._payable              # dernière ligne du fichier = hors zone
    assert 9 in m._payable


def test_fantome_ignore_si_tronque_ou_non_alimente(tmp_path):
    prod = _write(tmp_path / "s.jsonl", [_row(1, 9), _row(2, 9)])
    ghost = _write(tmp_path / "g.jsonl", [
        _row(1, 10, trunc=1),                    # en zone + tronqué : rien
        _row(2, 10, in_zone=False, fed=False),   # étape A : rien
        _row(3, 10, fed=False),                  # étape A : rien
        "corrompu"])
    m = PayableMemo()
    m.load_merged(prod, ghost)
    assert set(m._payable) == {1, 2}


# ───────────────────────── mémoire « déjà étiqueté » ─────────────────────────
def test_seen_reconstruit_depuis_le_fichier(tmp_path):
    ghost = _write(tmp_path / "g.jsonl", [
        _row(10, 46600), _row(10, 46610, in_zone=False), _row(11, 46605),
        _row(12, None), "corrompu"])
    assert gb.ghost_seen_from_dump(ghost) == {10: 46610, 11: 46605}
    assert gb.ghost_seen_from_dump(str(tmp_path / "absent")) == {}


def test_le_lot_ne_reetiquette_pas_apres_restart(tmp_path, monkeypatch):
    ghost = _write(tmp_path / "g.jsonl", [_row(10, 46600), _row(11, 46605)])
    monkeypatch.setenv("RELIQUARY_GHOST_DUMP", ghost)
    e, *_ = _eng(monkeypatch)
    e.__dict__.pop("_ghost_seen", None)       # moteur neuf (restart)
    asyncio.run(e._ghost_lot(_Env()))
    fired = set(e._vllm_backend.kwargs["prompt_indices"])
    assert fired == {12, 13}
    assert e._ghost_seen[10] == 46600 and e._ghost_seen[12] == 46610


# ──────────────────────── cohérence en vol (hors zone) ───────────────────────
def test_en_vol_hors_zone_retire_aussi_du_memo(monkeypatch):
    monkeypatch.setenv("RELIQUARY_GHOST_FEED", "1")
    e, memo, _ = _eng(monkeypatch)
    asyncio.run(e._ghost_lot(_Env()))
    removed = {idx for idx, payable, _ in memo.calls if not payable}
    assert removed == {11, 13}
    assert set(e._sz_notes) == {11, 13}


# ───────────────────────────── amorçage du moteur ────────────────────────────
def test_amorcage_charge_les_deux_fichiers(tmp_path, monkeypatch):
    prod = _write(tmp_path / "s.jsonl", [_row(1, 10)])
    ghost = _write(tmp_path / "g.jsonl", [_row(7, 10)])
    monkeypatch.setenv("RELIQUARY_SAMPLE_DUMP", prod)
    monkeypatch.setenv("RELIQUARY_GHOST_DUMP", ghost)
    monkeypatch.setenv("RELIQUARY_GHOST_FEED", "1")
    m = PayableMemo()
    engine._memo_bootstrap(m)
    assert set(m._payable) == {1, 7}


def test_amorcage_sans_feed_ignore_les_fantomes(tmp_path, monkeypatch):
    prod = _write(tmp_path / "s.jsonl", [_row(1, 10)])
    ghost = _write(tmp_path / "g.jsonl", [_row(7, 10)])
    monkeypatch.setenv("RELIQUARY_SAMPLE_DUMP", prod)
    monkeypatch.setenv("RELIQUARY_GHOST_DUMP", ghost)
    monkeypatch.setenv("RELIQUARY_GHOST_FEED", "0")
    m = PayableMemo()
    engine._memo_bootstrap(m)
    assert set(m._payable) == {1}


def test_amorcage_sans_dump_ne_leve_pas(monkeypatch):
    monkeypatch.delenv("RELIQUARY_SAMPLE_DUMP", raising=False)
    m = PayableMemo()
    engine._memo_bootstrap(m)
    assert m.size() == 0
