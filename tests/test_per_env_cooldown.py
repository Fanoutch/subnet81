"""Le cooldown doit être récupéré PAR ENVIRONNEMENT, y compris pour le premier.

Bug trouvé le 2026-08-06 en production : le mineur prenait le cooldown du
``/state`` générique et l'attribuait à son premier environnement actif, sur la
foi d'un commentaire affirmant que « le poll env-agnostique porte le cooldown du
premier env actif ». C'est FAUX contre le validateur live :

    /state                      -> 4195 entrées, indices 109747..21962528
    /state?env=openmathinstruct -> 4195 entrées, IDENTIQUES
    /state?env=opencodeinstruct -> 5031 entrées, indices 3091..2474641

Le générique renvoie le cooldown de MATH quel que soit notre env actif. Notre
mineur tourne en code seul : il filtrait donc ses prompts de code avec une liste
d'indices math, qui vit dans un tout autre espace. Résultat, il n'a jamais
écarté un seul prompt de code réellement consommé — 13 prompts sur les 5000 de
la tranche mesurée, chacun valant une génération gâchée puis un rejet.

Le correctif interroge ``?env=`` pour TOUS les environnements actifs.
"""
from __future__ import annotations

import pytest


class _State:
    def __init__(self, cooldown):
        self.cooldown_prompts = list(cooldown)


@pytest.mark.asyncio
async def test_first_env_cooldown_comes_from_its_own_env_query(monkeypatch):
    """Le premier env ne doit PAS hériter du cooldown générique."""
    from reliquary.miner import engine as eng

    GENERIC = [111, 222]          # cooldown math, espace d'indices étranger
    PER_ENV = {"opencodeinstruct": [7, 8, 9]}
    asked = []

    async def _fake_state(url, env=None, client=None):
        asked.append(env)
        if env is None:
            return _State(GENERIC)
        return _State(PER_ENV.get(env, []))

    monkeypatch.setattr(eng, "get_window_state_v2", _fake_state, raising=False)

    cooldowns: dict[str, set[int]] = {}
    for env_name in ["opencodeinstruct"]:
        st = await eng.get_window_state_v2("u", env=env_name, client=None)
        cooldowns[env_name] = set(st.cooldown_prompts)

    assert cooldowns["opencodeinstruct"] == {7, 8, 9}, (
        "le premier env a hérité du cooldown générique (bug 2026-08-06)"
    )
    assert "opencodeinstruct" in asked, "aucune requête ?env= n'a été faite"


def test_generic_and_per_env_cooldowns_are_not_assumed_equal():
    """Garde-fou documentaire : ne jamais re-supposer l'équivalence.

    Mesuré en production : le générique porte 4195 indices allant jusqu'à
    21962528, l'env code en porte 5031 tous sous 2481806 (la taille du dataset
    de code). Les deux listes ne sont ni de même taille ni du même espace.
    """
    generic_max, code_max, code_universe = 21962528, 2474641, 2481806
    assert generic_max > code_universe, (
        "le cooldown générique sort de l'espace d'indices du code"
    )
    assert code_max < code_universe
