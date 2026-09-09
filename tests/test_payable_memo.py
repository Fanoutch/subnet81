"""Slot mémo (2026-08-18) : recycler les payables déjà mesurés en vedettes.

Banc : armement 69 % → 75 % en remplaçant le 3e slot du sprint par le meilleur
ex-payable connu de la tranche (1 083 réapparitions mesurées sur 3 j, 51 % de
persistance, 34 % complètement ratées aujourd'hui).

Contrat :
- ``PayableMemo.load_jsonl`` charge l'historique (in_zone & non tronqué,
  DERNIÈRE mesure fait foi) ;
- ``update`` maintient la table au fil du grading ;
- ``best_in_range(lo, hi, exclude)`` = l'ex-payable LE PLUS FRAIS de la
  tranche, hors exclus ; None sinon ;
- le chargement ne propage jamais d'erreur (fichier absent/corrompu).
"""
import json

from reliquary.miner.payable_memo import PayableMemo


def _write(tmp_path, rows):
    p = tmp_path / "dump.jsonl"
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(p)


def row(idx, in_zone=True, trunc=0):
    return {"prompt_idx": idx, "in_zone": in_zone, "n_truncated": trunc}


class TestLoad:
    def test_loads_payables_only(self, tmp_path):
        m = PayableMemo()
        m.load_jsonl(_write(tmp_path, [
            row(10), row(20, in_zone=False), row(30, trunc=2), row(40),
        ]))
        assert m.best_in_range(0, 100) in (10, 40)
        assert m.size() == 2

    def test_last_measurement_wins(self, tmp_path):
        m = PayableMemo()
        m.load_jsonl(_write(tmp_path, [
            row(10),                    # payable...
            row(10, in_zone=False),     # ...puis re-mesuré non payable
            row(20, in_zone=False),
            row(20),                    # l inverse
        ]))
        assert m.best_in_range(0, 100) == 20
        assert m.size() == 1

    def test_missing_or_corrupt_file_is_safe(self, tmp_path):
        m = PayableMemo()
        m.load_jsonl(str(tmp_path / "absent.jsonl"))     # absent
        p = tmp_path / "bad.jsonl"
        p.write_text("{pas du json}\n" + json.dumps(row(5)) + "\n")
        m.load_jsonl(str(p))                              # ligne corrompue ignorée
        assert m.best_in_range(0, 100) == 5


class TestBestInRange:
    def test_freshest_first(self, tmp_path):
        m = PayableMemo()
        m.load_jsonl(_write(tmp_path, [row(10), row(50), row(30)]))
        # 30 est le plus récent
        assert m.best_in_range(0, 100) == 30

    def test_range_and_exclusion(self, tmp_path):
        m = PayableMemo()
        m.load_jsonl(_write(tmp_path, [row(10), row(50), row(300)]))
        assert m.best_in_range(0, 100, exclude={50}) == 10
        assert m.best_in_range(200, 400) == 300
        assert m.best_in_range(400, 500) is None

    def test_update_overrides(self, tmp_path):
        m = PayableMemo()
        m.update(7, True)
        assert m.best_in_range(0, 10) == 7
        m.update(7, False)
        assert m.best_in_range(0, 10) is None
        m.update(8, True)
        m.update(9, True)
        assert m.best_in_range(0, 10) == 9   # le plus frais


class TestEngineHook:
    def test_dump_group_sample_updates_memo(self, tmp_path, monkeypatch):
        """Le puits central de grading alimente la mémo en continu."""
        monkeypatch.setenv("RELIQUARY_SAMPLE_DUMP", str(tmp_path / "d.jsonl"))
        from reliquary.miner import engine as eng
        from reliquary.miner.payable_memo import get_memo
        get_memo().clear()
        # groupe payable : sigma élevé (rewards mixtes), non tronqué
        eng.dump_group_sample(prompt="p", prompt_idx=123,
                              rewards=[1, 1, 0, 0, 1, 0, 1, 0],
                              env_name="opencodeinstruct")
        assert get_memo().best_in_range(0, 1000) == 123
        # groupe unanime : non payable → retiré
        eng.dump_group_sample(prompt="p", prompt_idx=123,
                              rewards=[1] * 8, env_name="opencodeinstruct")
        assert get_memo().best_in_range(0, 1000) is None


class TestTopInRange:
    """Mémo de tête (04/09) : run courant d'abord, puis confirmations, puis
    fraîcheur. Mesuré : zone→zone 90 % dans le run courant, 86 % run
    précédent, 77 % ère v4 ; hors zone→zone 42 % (donc un raté fait sortir,
    comme avant, mais un prompt confirmé plusieurs fois passe devant)."""

    def test_confirmations_then_freshness(self):
        m = PayableMemo()
        m.update(10, True, window_n=40000)
        m.update(10, True, window_n=40010)      # 2 confirmations
        m.update(20, True, window_n=40020)      # 1 confirmation, plus frais
        assert m.top_in_range(0, 100, n=2) == [10, 20]
        assert m.best_in_range(0, 100) == 20    # contrat historique inchangé

    def test_out_of_zone_resets(self):
        m = PayableMemo()
        m.update(10, True, window_n=1)
        m.update(10, True, window_n=2)
        m.update(10, False, window_n=3)         # sort de la table
        assert m.top_in_range(0, 100) == []
        m.update(10, True, window_n=4)          # repart à 1 confirmation
        m.update(20, True, window_n=5)
        m.update(20, True, window_n=6)
        assert m.top_in_range(0, 100, n=2) == [20, 10]

    def test_current_run_first(self):
        m = PayableMemo()
        for _ in range(5):
            m.update(10, True, window_n=31000)  # ancien run, 5 confirmations
        m.update(20, True, window_n=41000)      # run courant, 1 confirmation
        assert m.top_in_range(0, 100, n=2, run_start=32791) == [20, 10]
        assert m.top_in_range(0, 100, n=2) == [10, 20]   # sans borne : confs

    def test_range_and_exclude(self):
        m = PayableMemo()
        for i in (5, 50, 150):
            m.update(i, True, window_n=1)
        assert m.top_in_range(0, 100, exclude={50}, n=5) == [5]

    def test_load_passes_window(self, tmp_path):
        m = PayableMemo()
        m.load_jsonl(_write(tmp_path, [
            dict(row(10), window_n=31000), dict(row(20), window_n=41000),
        ]))
        assert m.top_in_range(0, 100, n=2, run_start=32791) == [20, 10]
