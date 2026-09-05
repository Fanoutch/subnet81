"""Scores de prompts PRÉ-CALCULÉS hors ligne (24/08).

Mesuré : le classement de tranche coûte **2,80 s en tête de chaque fenêtre**,
sur le thread de la boucle asyncio (donc aucun POST ne peut partir pendant ce
temps). Décomposition rebanchée sur la box : lecture parquet 0,95 s,
`get_problem` 0,30 s, notation 0,91 s, tri 0,002 s.

Or `score_prompt`, `risk_short` et `volume_score` sont des fonctions **pures**
de (texte, modèle) : ni horloge, ni aléatoire, ni état. Les 2,48 M de scores
tiennent en 30 Mo (3 tableaux float32) et se calculent une fois hors ligne.
Le classement devient alors « trancher 5 000 flottants + argsort » ≈ 2 ms.

DEUX PROPRIÉTÉS DE SÛRETÉ, toutes deux testées ici :
  1. **Trois tableaux séparés**, pas un score combiné : `SHORT_RISK_LAMBDA` et
     `VOLUME_MU` sont des variables d'environnement réglables. Les figer dans
     le fichier obligerait à tout régénérer à chaque essai de réglage.
  2. **Empreinte** : si l'un des trois modèles change (ré-entraînement
     nocturne du prior), le fichier est PÉRIMÉ. Servir un classement périmé
     est pire que de ne rien pré-calculer — il faut retomber sur la notation
     en direct, pas continuer en silence.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from reliquary.miner import prompt_scores as ps


def _modele_bidon(poids: dict) -> dict:
    return {"kind": "test", "intercept": 0.0, "weights": poids,
            "center": 0.0, "scale": 1.0}


def test_empreinte_change_avec_chaque_modele():
    """L'empreinte doit dépendre des TROIS modèles : si le prior est
    ré-entraîné mais pas le malus, le fichier est quand même périmé."""
    a = _modele_bidon({"x": 1.0})
    b = _modele_bidon({"x": 2.0})

    base = ps.fingerprint(predictor=a, risk=a, volume=a, revision="r1")

    assert ps.fingerprint(predictor=b, risk=a, volume=a, revision="r1") != base
    assert ps.fingerprint(predictor=a, risk=b, volume=a, revision="r1") != base
    assert ps.fingerprint(predictor=a, risk=a, volume=b, revision="r1") != base
    # La révision du dataset compte aussi : les indices changeraient de sens.
    assert ps.fingerprint(predictor=a, risk=a, volume=a, revision="r2") != base
    # Stable pour des entrées identiques.
    assert ps.fingerprint(predictor=a, risk=a, volume=a, revision="r1") == base


def test_ecriture_puis_relecture_conserve_les_valeurs(tmp_path: Path):
    np = pytest.importorskip("numpy")
    chemin = tmp_path / "scores.npz"

    ps.save(
        chemin,
        score=np.array([0.5, 1.5, 2.5], dtype="float32"),
        risk=np.array([0.1, 0.2, 0.3], dtype="float32"),
        volume=np.array([9.0, 8.0, 7.0], dtype="float32"),
        fingerprint="abc123",
    )
    table = ps.load(chemin, expected_fingerprint="abc123")

    assert table is not None
    assert len(table) == 3
    assert table.combined(idx=1, risk_lambda=0.0, volume_mu=0.0) == pytest.approx(1.5)


def test_les_trois_tableaux_restent_separables(tmp_path: Path):
    """λ et μ s'appliquent À LA LECTURE — changer un réglage ne doit pas
    exiger de régénérer les 2,48 M de scores."""
    np = pytest.importorskip("numpy")
    chemin = tmp_path / "scores.npz"
    ps.save(chemin,
            score=np.array([1.0], dtype="float32"),
            risk=np.array([2.0], dtype="float32"),
            volume=np.array([4.0], dtype="float32"),
            fingerprint="f")
    table = ps.load(chemin, expected_fingerprint="f")

    # score - lambda*risk + mu*volume
    assert table.combined(0, risk_lambda=0.5, volume_mu=0.25) == pytest.approx(
        1.0 - 0.5 * 2.0 + 0.25 * 4.0
    )
    assert table.combined(0, risk_lambda=0.0, volume_mu=0.0) == pytest.approx(1.0)


def test_bande_de_volume_penalise_la_distance_a_la_cible(tmp_path: Path):
    """Le malus de BANDE (03/09) est inerte à β=0 et pénalise |volume−cible|.
    Il CENTRE la sélection sur une cible de volume plutôt que de la maximiser
    (μ) : viser ~8800 tokens (cible −0,12) resserre le traînard/l'arrivée."""
    np = pytest.importorskip("numpy")
    chemin = tmp_path / "scores.npz"
    ps.save(chemin,
            score=np.array([1.0, 1.0, 1.0], dtype="float32"),
            risk=np.zeros(3, dtype="float32"),
            volume=np.array([-1.0, 0.0, 1.0], dtype="float32"),
            fingerprint="f")
    table = ps.load(chemin, expected_fingerprint="f")

    # β=0 => strictement inchangé
    assert table.combined(2, risk_lambda=0.0, volume_mu=0.0) == pytest.approx(
        table.combined(2, risk_lambda=0.0, volume_mu=0.0,
                       volume_band_mu=0.0, volume_target=-0.12)
    )
    # malus = β·|volume − cible| ; cible 0.0, β=1 => vol=1 perd exactement 1.0
    assert table.combined(2, risk_lambda=0.0, volume_mu=0.0,
                          volume_band_mu=1.0, volume_target=0.0) == pytest.approx(0.0)
    # centre : à cible 0.0, le prompt de volume 0 bat les extrêmes ±1
    trio = [table.combined(i, risk_lambda=0.0, volume_mu=0.0,
                           volume_band_mu=1.0, volume_target=0.0) for i in range(3)]
    assert trio[1] > trio[0] and trio[1] > trio[2]


def test_empreinte_differente_refuse_le_fichier(tmp_path: Path):
    """Cœur de la sûreté : un modèle ré-entraîné doit INVALIDER le cache.

    On retourne None (→ notation en direct) plutôt que de lever : le mineur
    doit continuer à miner, juste plus lentement.
    """
    np = pytest.importorskip("numpy")
    chemin = tmp_path / "scores.npz"
    ps.save(chemin,
            score=np.array([1.0], dtype="float32"),
            risk=np.array([0.0], dtype="float32"),
            volume=np.array([0.0], dtype="float32"),
            fingerprint="ANCIENNE")

    assert ps.load(chemin, expected_fingerprint="NOUVELLE") is None


def test_fichier_absent_retourne_none(tmp_path: Path):
    assert ps.load(tmp_path / "nexiste_pas.npz", expected_fingerprint="f") is None


def test_fichier_corrompu_retourne_none_sans_lever(tmp_path: Path):
    """Un disque abîmé ne doit pas tuer le mineur."""
    chemin = tmp_path / "scores.npz"
    chemin.write_bytes(b"ceci n'est pas un npz")

    assert ps.load(chemin, expected_fingerprint="f") is None


# ---- câblage dans WindowRanking._build --------------------------------------

class _EnvBidon:
    """Environnement minimal : compte les lectures pour prouver qu'on les évite."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.lectures = 0

    def get_problem(self, idx):
        self.lectures += 1
        return {"prompt": f"probleme {idx}"}


def test_avec_table_aucune_lecture_de_prompt(monkeypatch, tmp_path):
    """LE point : avec la table, le classement ne lit AUCUN prompt.

    C'est la lecture parquet (0,95 s) + `get_problem` (0,30 s) qui disparaît,
    en plus de la notation (0,91 s).
    """
    np = pytest.importorskip("numpy")
    from reliquary.miner import engine as eng

    chemin = tmp_path / "scores.npz"
    ps.save(chemin,
            score=np.array([0.1, 0.9, 0.5, 0.7], dtype="float32"),
            risk=np.zeros(4, dtype="float32"),
            volume=np.zeros(4, dtype="float32"),
            fingerprint="EMPREINTE")
    monkeypatch.setattr(eng, "_SCORE_TABLE",
                        ps.load(chemin, expected_fingerprint="EMPREINTE"))

    env = _EnvBidon(4)
    r = eng.WindowRanking()
    r._build(env, model=None, prompt_range=(0, 4), cooldown=set())

    assert env.lectures == 0            # rien n'a été lu
    assert r._ranked[:2] == [1, 3]      # trié par score décroissant


def test_sans_table_le_comportement_est_inchange(monkeypatch):
    """Repli : sans table, on lit et on note comme avant."""
    from reliquary.miner import engine as eng
    from reliquary.miner import prompt_predictor as _pp

    monkeypatch.setattr(eng, "_SCORE_TABLE", None)
    monkeypatch.setattr(_pp, "score_prompt", lambda m, t: float(len(t)))

    env = _EnvBidon(3)
    r = eng.WindowRanking()
    r._build(env, model={}, prompt_range=(0, 3), cooldown=set())

    assert env.lectures == 3            # chemin historique : on lit tout
    assert len(r._ranked) == 3


def test_la_table_respecte_le_cooldown(monkeypatch, tmp_path):
    """Le cooldown doit continuer d'écarter, table ou pas."""
    np = pytest.importorskip("numpy")
    from reliquary.miner import engine as eng

    chemin = tmp_path / "s.npz"
    ps.save(chemin, score=np.array([9.0, 8.0, 7.0], dtype="float32"),
            risk=np.zeros(3, dtype="float32"),
            volume=np.zeros(3, dtype="float32"), fingerprint="F")
    monkeypatch.setattr(eng, "_SCORE_TABLE",
                        ps.load(chemin, expected_fingerprint="F"))

    r = eng.WindowRanking()
    r._build(_EnvBidon(3), model=None, prompt_range=(0, 3), cooldown={0})

    assert 0 not in r._ranked
