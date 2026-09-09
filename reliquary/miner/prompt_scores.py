"""Table de scores de prompts PRÉ-CALCULÉE hors ligne (24/08).

POURQUOI — mesuré ce jour-là sur 56 fenêtres, moteur en marche :
le classement de tranche coûte **2,80 s p50 (4,6 p90, 14 s max) en tête de
chaque fenêtre**, et il tourne sur le thread de la boucle asyncio : pendant ce
temps aucun ``GET /state`` ni aucun POST ne part. Rebanché sur la box, il se
décompose en lecture parquet 0,95 s, ``get_problem`` 0,30 s, notation 0,91 s,
tri 0,002 s.

Or les trois fonctions de notation (``score_prompt``, ``risk_short``,
``volume_score``) sont **pures** : fonction du texte et du modèle, sans
horloge, sans aléatoire, sans état. Les 2 481 806 prompts tiennent donc en
3 tableaux ``float32`` de 10 Mo, calculables une fois pour toutes hors ligne
(``scripts/precompute_prompt_scores.py``). Le classement se réduit alors à
trancher 5 000 flottants et trier.

Enjeu chiffré : 1 seconde d'arrivée vaut **+1,46 place** et 390 tokens ;
−3 s doublent les payées par fenêtre (0,44 → 0,85, mesuré sur 174 fenêtres).

DEUX CHOIX DE CONCEPTION, tous deux délibérés :

1. **Trois tableaux séparés, pas un score combiné.** ``SHORT_RISK_LAMBDA`` et
   ``VOLUME_MU`` sont des variables d'environnement qu'on règle en A/B ; les
   figer dans le fichier obligerait à régénérer 2,48 M de scores à chaque
   essai. La combinaison se fait à la lecture, pour trois multiplications.

2. **Empreinte vérifiée, repli silencieux.** Si l'un des modèles change
   (l'entraînement du prior tourne toutes les nuits), la table est périmée —
   et servir un classement périmé est PIRE que ne rien pré-calculer. On
   retourne alors ``None`` et l'appelant retombe sur la notation en direct :
   le mineur continue à miner, seulement plus lentement.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def fingerprint(*, predictor: Any, risk: Any, volume: Any, revision: str) -> str:
    """Empreinte des entrées dont la table dépend.

    Les TROIS modèles comptent : un prior ré-entraîné sans que le malus bouge
    périme quand même la table. La révision du dataset aussi — les indices
    changeraient de signification.
    """
    h = hashlib.sha256()
    for obj in (predictor, risk, volume):
        if obj is None:
            h.update(b"\x00none\x00")
        else:
            h.update(json.dumps(obj, sort_keys=True,
                                separators=(",", ":")).encode("utf-8"))
        h.update(b"\x1f")
    h.update(str(revision).encode("utf-8"))
    return h.hexdigest()


class ScoreTable:
    """Vue en lecture seule sur les trois tableaux, combinés à la demande."""

    __slots__ = ("_score", "_risk", "_volume")

    def __init__(self, score, risk, volume) -> None:
        self._score = score
        self._risk = risk
        self._volume = volume

    def __len__(self) -> int:
        return len(self._score)

    def combined(
        self,
        idx: int,
        *,
        risk_lambda: float,
        volume_mu: float,
        volume_band_mu: float = 0.0,
        volume_target: float = 0.0,
    ) -> float:
        """``score − λ·risk + μ·volume − β·|volume − cible|``.

        Les trois premiers termes sont la formule EXACTE de
        ``WindowRanking._build``. Le quatrième (BANDE, 03/09) est un MALUS de
        distance à une cible de volume : à arrivée égale (round 2), on génère
        ~10235 tokens contre ~8300 pour les meneurs les plus constants
        (v5.9 choisit incidemment des prompts longs → gros traînard → 63 %
        seulement de nos payées arrivent au round 2 contre 83-88 % chez eux).
        Viser une bande centrée sur ``volume_target`` (en unités volume_score,
        −0,12 ≈ 8800 tokens) resserre le traînard donc l'arrivée, SANS tomber
        sous la barre (8800 > ~8300 payable au round 2). ``β=0`` (défaut) =
        comportement historique strictement inchangé.
        """
        v = float(self._score[idx])
        if risk_lambda:
            v -= risk_lambda * float(self._risk[idx])
        if volume_mu:
            v += volume_mu * float(self._volume[idx])
        if volume_band_mu:
            v -= volume_band_mu * abs(float(self._volume[idx]) - volume_target)
        return v


def save(path, *, score, risk, volume, fingerprint: str) -> None:
    """Écrit la table. ``score``/``risk``/``volume`` sont des tableaux numpy."""
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        score=np.asarray(score, dtype="float32"),
        risk=np.asarray(risk, dtype="float32"),
        volume=np.asarray(volume, dtype="float32"),
        fingerprint=np.array(fingerprint),
    )


def load(path, *, expected_fingerprint: str) -> Optional[ScoreTable]:
    """Charge la table si elle est FRAÎCHE, sinon ``None``.

    Ne lève jamais : un fichier absent, périmé ou corrompu doit dégrader vers
    la notation en direct, pas arrêter le mineur.
    """
    try:
        import numpy as np

        p = Path(path)
        if not p.exists():
            return None
        with np.load(p, allow_pickle=False) as z:
            got = str(z["fingerprint"])
            if got != expected_fingerprint:
                logger.warning(
                    "table de scores PÉRIMÉE (empreinte %s… attendue %s…) — "
                    "retour à la notation en direct",
                    got[:12], expected_fingerprint[:12],
                )
                return None
            return ScoreTable(z["score"], z["risk"], z["volume"])
    except Exception:
        logger.warning("table de scores illisible — notation en direct",
                       exc_info=True)
        return None
