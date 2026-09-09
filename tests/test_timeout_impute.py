"""Imputation des timeouts dans la décision de zone (01/09).

Mesure fondatrice : 16,3 % de nos envois rejetés out_of_zone au verdict
(référence 1,7-5,5 %) — nos timeouts locaux (1 s vs leurs 5 s) comptés à 0
fabriquaient une fausse dispersion. Le rollout tué est un INCONNU : la
décision de zone l'impute à la moyenne des rollouts réellement notés.
"""
import pytest
from reliquary.miner.engine import timeout_imputed_for_zone, _safe_reward_ex
from reliquary.environment.code_grader import grade_structured_cases_ex

CASES = [{"entry": {"kind": "function", "name": "f"}, "args": [2],
          "kwargs": {}, "expected": 4, "compare": "exact"}]


def test_timeout_reel_leve_le_drapeau():
    r, tmo = grade_structured_cases_ex(
        "def f(x):\n  while True:\n    pass", CASES, timeout_s=0.8)
    assert (r, tmo) == (0.0, True)


def test_code_juste_note_sans_drapeau():
    r, tmo = grade_structured_cases_ex("def f(x):\n  return x*x", CASES, timeout_s=5.0)
    assert r == 1.0 and tmo is False


def test_crash_n_est_pas_un_timeout():
    r, tmo = grade_structured_cases_ex("def f(x):\n  raise ValueError", CASES, timeout_s=5.0)
    assert r == 0.0 and tmo is False


def test_off_par_defaut(monkeypatch):
    monkeypatch.delenv("RELIQUARY_TIMEOUT_IMPUTE", raising=False)
    rw = [0.0]*10 + [0.72]*6
    fl = [True]*10 + [False]*6
    assert timeout_imputed_for_zone(rw, fl) == rw


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv("RELIQUARY_TIMEOUT_IMPUTE", "1")


def test_le_scenario_du_16_pourcent(on):
    """10 timeouts + 6 notés uniformes 0,72 : le sigma imputé s'effondre —
    le groupe sera correctement jeté localement au lieu d'être rejeté chez eux."""
    from reliquary.miner.engine import _std
    rw = [0.0]*10 + [0.72]*6
    fl = [True]*10 + [False]*6
    out = timeout_imputed_for_zone(rw, fl)
    assert out == pytest.approx([0.72]*16)
    assert _std(out) < 0.01 < 0.24 < _std(rw)


def test_vraie_dispersion_survit(on):
    """Timeouts + notés réellement disperses : sigma reste au-dessus du seuil."""
    from reliquary.miner.engine import _std
    rw = [0.0]*4 + [1.0, 1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 1.0, 0.0, 1.0, 0.5, 0.0]
    fl = [True]*4 + [False]*12
    assert _std(timeout_imputed_for_zone(rw, fl)) >= 0.24


def test_garde_fou_moins_de_4_notes(on):
    rw = [0.0]*14 + [0.9, 0.8]; fl = [True]*14 + [False]*2
    assert timeout_imputed_for_zone(rw, fl) == rw


def test_sans_timeout_vecteur_intact(on):
    rw = [0.5]*16
    assert timeout_imputed_for_zone(rw, [False]*16) == rw


def test_safe_reward_ex_env_sans_ex():
    class Env:
        def compute_reward(self, p, c): return 0.7
    assert _safe_reward_ex(Env(), {}, "x") == (0.7, False)


def test_safe_reward_ex_env_avec_ex():
    class Env:
        def compute_reward_ex(self, p, c): return 0.3, True
    assert _safe_reward_ex(Env(), {}, "x") == (0.3, True)
