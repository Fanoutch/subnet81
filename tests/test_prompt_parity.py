"""RELIQUARY_PROMPT_PARITY : partition pair/impair de la sélection de prompts.

But : deux mineurs sous la même coldkey (H100=impair, H200=pair) ne peuvent
JAMAIS choisir le même prompt, donc jamais se voler un hash_duplicate entre
eux. Le filtre doit tenir sur TOUS les chemins de sélection :
  1. tirage uniforme (avec et sans eligible_indices) ;
  2. best-of-N sous prédicteur (hérite par récursion) ;
  3. WindowRanking._build (le chemin réellement emprunté en prod).
`PROMPT_PARITY` est lu une fois à l'import — les tests patchent la constante
module, comme le ferait un process lancé avec l'env var posée.
"""

import random

import pytest

from reliquary.miner import engine


class _FakeEnv:
    """Env minimal : len + get_problem, texte = l'indice (score déterministe)."""

    eligible_indices = None

    def __init__(self, n=100):
        self._n = n

    def __len__(self):
        return self._n

    def get_problem(self, idx):
        return {"prompt": str(idx)}


@pytest.fixture
def parity(monkeypatch):
    def _set(value):
        monkeypatch.setattr(engine, "PROMPT_PARITY", value)
    return _set


def test_parity_ok_off_accepts_everything(parity):
    parity(None)
    assert engine._parity_ok(0) and engine._parity_ok(1) and engine._parity_ok(7)


@pytest.mark.parametrize("want", [0, 1])
def test_parity_ok_filters(parity, want):
    parity(want)
    for idx in range(10):
        assert engine._parity_ok(idx) == (idx % 2 == want)


@pytest.mark.parametrize("want", [0, 1])
def test_uniform_draw_respects_parity(parity, want):
    parity(want)
    env = _FakeEnv(100)
    rng = random.Random(42)
    for _ in range(200):
        idx = engine.pick_prompt_idx(env, set(), rng=rng, prompt_range=(10, 60))
        assert 10 <= idx < 60
        assert idx % 2 == want


def test_eligible_pool_respects_parity(parity):
    parity(1)
    env = _FakeEnv(100)
    env.eligible_indices = [2, 4, 6, 7]
    rng = random.Random(0)
    for _ in range(50):
        assert engine.pick_prompt_idx(env, set(), rng=rng) == 7


def test_exhausted_parity_raises_not_hangs(parity):
    # Tous les prompts de la bonne parité en cooldown -> RuntimeError explicite
    # (jamais une boucle infinie ni un indice de mauvaise parité).
    parity(0)
    env = _FakeEnv(20)
    cooldown = set(range(0, 20, 2))
    with pytest.raises(RuntimeError):
        engine.pick_prompt_idx(env, cooldown, rng=random.Random(1),
                               prompt_range=(0, 20))


@pytest.mark.parametrize("want", [0, 1])
def test_best_of_n_predictor_path_respects_parity(parity, monkeypatch, want):
    parity(want)
    from reliquary.miner import prompt_predictor as _pp
    # Score = l'indice : le "meilleur" candidat est le plus grand indice légal.
    monkeypatch.setattr(_pp, "score_prompt", lambda model, text: int(text))
    env = _FakeEnv(100)
    rng = random.Random(7)
    for _ in range(30):
        idx = engine.pick_prompt_idx(
            env, set(), rng=rng, prompt_range=(0, 100),
            predictor={"fake": True}, n_candidates=8,
        )
        assert idx % 2 == want


@pytest.mark.parametrize("want", [0, 1])
def test_window_ranking_build_respects_parity(parity, monkeypatch, want):
    parity(want)
    from reliquary.miner import prompt_predictor as _pp
    monkeypatch.setattr(_pp, "score_prompt", lambda model, text: int(text))
    ranking = engine.WindowRanking()
    env = _FakeEnv(60)
    key = (28500, "randomness", "opencodeinstruct")
    got = ranking.best(env, {"fake": True}, key, (0, 50), cooldown=set())
    # Meilleur score = plus grand indice DE LA BONNE PARITÉ (49 impair, 48 pair).
    assert got == (48 if want == 0 else 49)
    assert ranking._ranked, "classement vide"
    assert all(i % 2 == want for i in ranking._ranked)


def test_window_ranking_off_keeps_full_slice(parity, monkeypatch):
    parity(None)
    from reliquary.miner import prompt_predictor as _pp
    monkeypatch.setattr(_pp, "score_prompt", lambda model, text: int(text))
    ranking = engine.WindowRanking()
    env = _FakeEnv(60)
    got = ranking.best(env, {"fake": True}, (1, "r", "e"), (0, 50), cooldown=set())
    assert got == 49
    assert len(ranking._ranked) == 50


def test_env_var_parsing_contract():
    # Reproduit l'expression d'import : seuls les chiffres activent le filtre.
    def parse(raw):
        raw = raw.strip()
        return int(raw) % 2 if raw.isdigit() else None
    assert parse("") is None
    assert parse("0") == 0
    assert parse("1") == 1
    assert parse("2") == 0          # modulo : 2 = pair
    assert parse("even") is None    # ⚠️ les mots ne sont PAS reconnus
    assert parse("odd") is None
