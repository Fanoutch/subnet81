"""Préchargement du checkpoint (27/08).

Le téléchargement HF pèse 57 s en médiane — jusqu'à 9 min 25 observé — et il
est attendu dans le chemin critique alors qu'il n'écrit que des fichiers : il
ne touche pas le GPU. HF publie la révision 100 à 350 s avant que le
validateur ne bascule dessus. La sortir du chemin critique fait passer l'arrêt
de production de 67 s à ~33 s par avancée.

Deux propriétés de sûreté dominent ces tests :
  1. le préchargement ne PURGE JAMAIS le cache HF — purger effacerait le
     checkpoint en cours d'utilisation ;
  2. tout échec est avalé — le chemin normal doit pouvoir prendre le relais.
"""
from __future__ import annotations

import asyncio

import pytest

from reliquary.miner import checkpoint_prefetch as cp


class _C:
    def __init__(self, cid): self.commit_id = cid


def _commits(*ids):
    return lambda repo: [_C(i) for i in ids]


# ---- réglages -------------------------------------------------------------

def test_desactive_par_defaut(monkeypatch):
    """Sans la variable, aucune tâche ne doit être créée par l'appelant."""
    monkeypatch.delenv("RELIQUARY_CHECKPOINT_PREFETCH", raising=False)
    assert cp.prefetch_enabled() is False
    monkeypatch.setenv("RELIQUARY_CHECKPOINT_PREFETCH", "1")
    assert cp.prefetch_enabled() is True


def test_cadence_de_sondage(monkeypatch):
    monkeypatch.delenv("RELIQUARY_CHECKPOINT_PREFETCH_POLL_S", raising=False)
    assert cp.prefetch_poll_seconds() == 30.0
    monkeypatch.setenv("RELIQUARY_CHECKPOINT_PREFETCH_POLL_S", "5")
    assert cp.prefetch_poll_seconds() == 5.0
    # une valeur absurde retombe sur le défaut plutôt que de casser la boucle
    monkeypatch.setenv("RELIQUARY_CHECKPOINT_PREFETCH_POLL_S", "nimporte")
    assert cp.prefetch_poll_seconds() == 30.0
    monkeypatch.setenv("RELIQUARY_CHECKPOINT_PREFETCH_POLL_S", "-3")
    assert cp.prefetch_poll_seconds() == 30.0


# ---- choix de la cible ----------------------------------------------------

def test_rien_a_faire_sans_repo():
    assert cp.choose_prefetch_target(None, "abc", _commits("z"), set()) is None


def test_rien_a_faire_si_la_derniere_est_deja_active():
    """Le cas nominal entre deux publications : ne pas re-tirer ce qu'on a."""
    assert cp.choose_prefetch_target("r", "abc", _commits("abc"), set()) is None


def test_rien_a_faire_si_deja_prechargee():
    assert cp.choose_prefetch_target("r", "abc", _commits("neuf"), {"neuf"}) is None


def test_cible_la_revision_nouvellement_publiee():
    assert cp.choose_prefetch_target("r", "abc", _commits("neuf", "abc"), set()) == "neuf"


def test_depot_vide_ne_leve_pas():
    assert cp.choose_prefetch_target("r", "abc", lambda repo: [], set()) is None


# ---- téléchargement -------------------------------------------------------

def test_prefetch_once_telecharge_et_reussit():
    vus = []
    def dl(repo, rev): vus.append((repo, rev))
    ok = asyncio.run(cp.prefetch_once("r", "neuf", download_fn=dl))
    assert ok is True and vus == [("r", "neuf")]


def test_prefetch_once_avale_l_echec():
    """Repli OUVERT : un échec de préchargement ne doit jamais remonter."""
    def dl(repo, rev): raise RuntimeError("HF down")
    assert asyncio.run(cp.prefetch_once("r", "neuf", download_fn=dl)) is False


# ---- boucle ---------------------------------------------------------------

def _run_loop(get_active, commits_fn, dl, rounds=3):
    async def _noop(_): return None
    return asyncio.run(cp.prefetch_loop(
        get_active=get_active, list_commits_fn=commits_fn,
        download_fn=dl, sleep_fn=_noop, poll_s=0, max_rounds=rounds,
    ))


def test_la_boucle_ne_telecharge_qu_une_fois_la_meme_revision():
    vus = []
    def dl(repo, rev): vus.append(rev)
    ok = _run_loop(lambda: ("r", "vieux"), _commits("neuf", "vieux"), dl, rounds=5)
    assert vus == ["neuf"], "5 tours, 1 seul téléchargement attendu"
    assert ok == 1


def test_la_boucle_suit_les_publications_successives():
    """v1 actif -> v2 publie -> on precharge v2 ; le mineur bascule, v3 sort."""
    etat = {"rev": "v1"}
    pub = {"cur": ["v2", "v1"]}
    vus = []
    def dl(repo, rev):
        vus.append(rev)
        etat["rev"] = rev            # le mineur bascule dessus ensuite
        pub["cur"] = ["v3", "v2", "v1"]
    async def _noop(_): return None
    asyncio.run(cp.prefetch_loop(
        get_active=lambda: ("r", etat["rev"]),
        list_commits_fn=lambda repo: [_C(c) for c in pub["cur"]],
        download_fn=dl, sleep_fn=_noop, poll_s=0, max_rounds=3))
    assert vus == ["v2", "v3"]


def test_une_erreur_de_sondage_ne_tue_pas_la_boucle():
    appels = {"n": 0}
    def commits(repo):
        appels["n"] += 1
        if appels["n"] == 1: raise RuntimeError("API HF en vrac")
        return [_C("neuf")]
    vus = []
    ok = _run_loop(lambda: ("r", "vieux"), commits, lambda r, v: vus.append(v), rounds=3)
    assert vus == ["neuf"], "la boucle doit survivre au tour en erreur"
    assert ok == 1


def test_un_get_active_qui_leve_ne_tue_pas_la_boucle():
    n = {"i": 0}
    def get_active():
        n["i"] += 1
        if n["i"] == 1: raise RuntimeError("etat indisponible")
        return ("r", "vieux")
    vus = []
    _run_loop(get_active, _commits("neuf"), lambda r, v: vus.append(v), rounds=3)
    assert vus == ["neuf"]


# ---- LA propriété de sûreté ----------------------------------------------

def _noms_utilises_dans_le_code(mod):
    """Tous les identifiants RÉELLEMENT référencés — docstrings exclues."""
    import ast, inspect
    tree = ast.parse(inspect.getsource(mod))
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name): out.add(n.id)
        elif isinstance(n, ast.Attribute): out.add(n.attr)
        elif isinstance(n, ast.arg): out.add(n.arg)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names: out.add((a.asname or a.name).split(".")[0])
    return out


def test_le_prechargement_ne_purge_JAMAIS_le_cache():
    """⚠️ Le test qui compte.

    ``_hf_download`` appelle ``_prune_hf_revisions(repo_id, revision)``, qui
    efface toutes les révisions SAUF celle qu'on vient de tirer. Appelé depuis
    le préchargement, il supprimerait le checkpoint EN COURS D'UTILISATION —
    le mineur perdrait le modèle sous ses pieds. Le préchargement doit donc
    recevoir un ``download_fn`` nu, jamais ``_hf_download``.
    """
    noms = _noms_utilises_dans_le_code(cp)
    assert "_prune_hf_revisions" not in noms
    assert "_hf_download" not in noms


def test_le_module_ne_touche_pas_au_gpu():
    """Le préchargement n'écrit que des fichiers : aucun chargement en VRAM.
    C'est ce qui permet de le lancer AVANT que le validateur ne bascule."""
    noms = _noms_utilises_dans_le_code(cp)
    for interdit in ("load_fn", "torch", "vllm", "cuda", "LLM"):
        assert interdit not in noms, f"{interdit} n'a rien à faire ici"
