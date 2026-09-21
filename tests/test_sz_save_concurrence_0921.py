"""21/09 : la sauvegarde de la liste noire doit survivre à la concurrence.

La liste noire (``sz_blacklist.json``) écarte désormais ~65 000 prompts pour
20 000 fenêtres : elle DOIT survivre aux redémarrages. Or ``_sz_save`` échouait
régulièrement (25 Tracebacks en 2 h le 21/09), pour deux raisons :
1. un fichier temporaire ``.tmp`` PARTAGÉ entre fils : un fil fait
   ``os.replace`` du ``.tmp`` que l'autre écrivait -> ``FileNotFoundError`` ;
2. le dict est parcouru pendant qu'un autre fil y ajoute -> « dictionary
   changed size during iteration ».
Une sauvegarde ratée n'est pas grave en soi (la suivante réécrit tout), mais un
redémarrage juste après perd les dernières notes.
"""
from __future__ import annotations

import json
import logging
import threading

from reliquary.miner import engine


def _eng():
    return engine.MiningEngine.__new__(engine.MiningEngine)


def test_sauvegarde_simple(tmp_path, monkeypatch):
    f = tmp_path / "bl.json"
    monkeypatch.setenv("RELIQUARY_SZ_BLACKLIST_FILE", str(f))
    _eng()._sz_save({1: 100, 2: 200})
    assert json.loads(f.read_text()) == {"1": 100, "2": 200}


def test_sauvegardes_concurrentes_sans_erreur(tmp_path, monkeypatch, caplog):
    f = tmp_path / "bl.json"
    monkeypatch.setenv("RELIQUARY_SZ_BLACKLIST_FILE", str(f))
    e = _eng()
    bl = {i: i for i in range(20_000)}
    stop = threading.Event()

    def mutate():                      # un autre fil note des hors-zone
        # BORNÉ : non borné, le dict gonfle à des millions de clés dès que
        # les sauvegardes réussissent, et le test tue la machine.
        for k in range(10**6, 10**6 + 30_000):
            if stop.is_set():
                break
            bl[k] = k

    def save():
        for _ in range(10):
            e._sz_save(bl)

    caplog.set_level(logging.WARNING)
    m = threading.Thread(target=mutate)
    m.start()
    savers = [threading.Thread(target=save) for _ in range(8)]
    for t in savers:
        t.start()
    for t in savers:
        t.join()
    stop.set()
    m.join()
    assert not [r for r in caplog.records if "non sauvegardee" in r.getMessage()]
    data = json.loads(f.read_text())            # fichier final valide
    assert all(str(i) in data for i in range(20_000))


def test_aucun_fichier_temporaire_ne_traine(tmp_path, monkeypatch):
    f = tmp_path / "bl.json"
    monkeypatch.setenv("RELIQUARY_SZ_BLACKLIST_FILE", str(f))
    e = _eng()
    ts = [threading.Thread(target=e._sz_save, args=({i: i},)) for i in range(16)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert [p.name for p in tmp_path.iterdir()] == ["bl.json"]
