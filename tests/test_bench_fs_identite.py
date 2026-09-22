"""Taux de tokens identiques entre deux passages du banc forced-seed (22/09).
Remplace l'empreinte globale, trop stricte : le chemin actuel change lui-même
d'empreinte d'une répétition à l'autre (bascules numériques près d'une
frontière de CDF)."""
import importlib.util
import pathlib

_p = pathlib.Path(__file__).resolve().parents[1] / "ops" / "bench_fs_h100.py"
_spec = importlib.util.spec_from_file_location("bench_fs_h100", _p)
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


def test_taux_identite():
    ref = [[1, 2, 3, 4], [5, 6, 7, 8]]
    other = [[1, 2, 3, 4], [5, 6, 9, 9]]
    seq, pos = bench.identity_rates(ref, other)
    assert seq == 0.5
    assert pos == (1.0 + 0.5) / 2


def test_taux_identite_parfait():
    ref = [[1, 2], [3, 4]]
    assert bench.identity_rates(ref, [r[:] for r in ref]) == (1.0, 1.0)


def test_modes_connus():
    assert set(bench.MODES) >= {"nu_greedy", "nu_t1", "libre_greedy", "libre_t1",
                                "noop", "fs", "fast"}
