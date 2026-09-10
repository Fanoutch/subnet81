"""Politique du hot-swap de checkpoint — décision et paramètres de la gate.

Extrait d'``engine._load_checkpoint`` pour être testable seul : la branche
qui décide de SERVIR ou non un moteur dont les poids ont été échangés à
chaud est un chemin de CONFORMITÉ (un mauvais échange = SEED_MISMATCH en
masse), elle mérite ses propres tests.

Trois modes, via ``RELIQUARY_HOT_SWAP`` :

``off`` (défaut)
    Aucun échange. Comportement historique : reconstruction complète.
``shadow``
    On échange, on passe la gate, on **journalise le taux**, puis on
    reconstruit quand même. Sert à CALIBRER le plancher sur un moteur sain
    avant d'armer : le plancher 0,80 date de v4 et n'a jamais été revu sous
    le sampling plat (T=1,0 / top_p 1,0 / top_k 0). Coût : le temps de
    l'échange + la sonde s'ajoutent au gel, pour quelques fenêtres.
``armed``
    On sert le moteur échangé dès que la gate passe.

Toute valeur inconnue retombe sur ``off`` : une faute de frappe dans le
launcher ne doit pas armer un chemin de conformité.
"""
import os

GATE_TOKENS_DEFAULT = 48
GATE_FLOOR_DEFAULT = 0.80

_MODES = {"0": "off", "": "off", "1": "armed", "shadow": "shadow"}


def hot_swap_mode() -> str:
    """``off`` | ``shadow`` | ``armed`` — lu à chaque avancée de checkpoint."""
    return _MODES.get(os.environ.get("RELIQUARY_HOT_SWAP", "0").strip().lower(),
                      "off")


def hot_swap_decision(mode: str, *, swapped: bool, gate_ok: bool) -> str:
    """``keep`` (servir le moteur échangé) ou ``rebuild`` (reconstruction).

    ``rebuild`` est le repli sûr : c'est exactement le comportement d'avant
    le hot-swap, donc aucun retour ici n'expose à un rejet de conformité.
    """
    if mode == "armed" and swapped and gate_ok:
        return "keep"
    return "rebuild"


def hot_swap_gate_params() -> tuple[int, float]:
    """(n_tokens, plancher) de la gate, réglables sans redéploiement de code.

    Le plancher se règle DEPUIS LA MESURE du mode ombre, jamais d'intuition.
    """
    try:
        n = int(os.environ.get("RELIQUARY_HOT_SWAP_GATE_TOKENS", ""))
    except ValueError:
        n = GATE_TOKENS_DEFAULT
    try:
        floor = float(os.environ.get("RELIQUARY_HOT_SWAP_GATE_FLOOR", ""))
    except ValueError:
        floor = GATE_FLOOR_DEFAULT
    return n, floor
