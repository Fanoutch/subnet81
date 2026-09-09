"""Portefeuille 2 priors (2026-08-15) : vedette 1 = top-1 v4.1 ; vedette 2 =
le plus lourd selon le modèle 2 (v4) parmi le top-50 v4.1. Table d'espérance
mesurée : bande 8-16k ≈ 3× l'espérance des picks 6-8k actuels."""
import sys
sys.path.insert(0, "/root/subnet81/reliquary-miner-priv")
from reliquary.miner import prompt_predictor as pp
from reliquary.miner.engine import WindowRanking, _load_predictor_2


class _Env:
    name = "code"
    def __init__(self, texts): self._t = texts
    def __len__(self): return len(self._t)
    def get_problem(self, i): return {"prompt": self._t[i]}


def _model(pairs):
    return pp.train_word_priors(
        [{"prompt": p, "target": t} for p, t in pairs], k=0.1)


def _setup():
    # m1 aime "alpha" (léger), m2 aime "omega" (lourd)
    m1 = _model([("alpha alpha", 1.0), ("omega omega", 0.4), ("zzz", 0.0)])
    m2 = _model([("omega omega", 1.0), ("alpha alpha", 0.1), ("zzz", 0.0)])
    texts = ["alpha alpha", "omega omega", "alpha omega", "zzz", "zzz"]
    return _Env(texts), m1, m2


def test_best_then_heavy_disjoint():
    env, m1, m2 = _setup()
    r = WindowRanking()
    key = (1, "r", "code")
    first = r.best(env, m1, key, (0, 5), set())
    assert first == 0  # top-1 de m1 = "alpha alpha"
    heavy = r.best_heavy(env, m2, key, (0, 5), set())
    assert heavy == 1  # le plus "omega" selon m2 parmi le top m1
    # jamais le même prompt deux fois
    assert heavy != first
    nxt = r.best(env, m1, key, (0, 5), set())
    assert nxt not in (first, heavy)


def test_heavy_none_without_model_or_build():
    env, m1, m2 = _setup()
    r = WindowRanking()
    key = (1, "r", "code")
    assert r.best_heavy(env, None, key, (0, 5), set()) is None
    # classement pas construit pour cette clé -> décline
    assert r.best_heavy(env, m2, (2, "x", "code"), (0, 5), set()) is None


def test_heavy_respects_cooldown():
    env, m1, m2 = _setup()
    r = WindowRanking()
    key = (1, "r", "code")
    r.best(env, m1, key, (0, 5), set())
    heavy = r.best_heavy(env, m2, key, (0, 5), {1})  # 1 en cooldown
    assert heavy not in (None, 1)


def test_loader_env(monkeypatch, tmp_path):
    monkeypatch.delenv("RELIQUARY_PROMPT_PREDICTOR_2", raising=False)
    assert _load_predictor_2() is None
    m = _model([("aa", 1.0)])
    p = tmp_path / "m.json"
    pp.save_model(m, str(p))
    monkeypatch.setenv("RELIQUARY_PROMPT_PREDICTOR_2", str(p))
    assert _load_predictor_2() is not None
    monkeypatch.setenv("RELIQUARY_PROMPT_PREDICTOR_2", "/nulle/part.json")
    assert _load_predictor_2() is None
