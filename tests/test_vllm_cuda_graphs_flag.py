"""CUDA graphs (``enforce_eager=False``) pilotables par variable d'env.

Mesuré au banc le 2026-07-22 (H100, 128 séquences) : 2116 tok/s avec CUDA
graphs contre 1725 en eager, soit **+23%** à batch égal — et mieux qu'un batch
de 256 séquences en eager. Conformité vérifiée
(scripts/validate_vllm_forced_seed_group.py, GATE_EAGER=0) :
seed_consistency 0.9880 groupe / 0.9531 pire rollout, contre 0.9900 / 0.9592 en
eager — planchers validateur 0.80 / 0.75.

``enforce_eager=True`` reste le DÉFAUT : c'est le réglage du bring-up, retenu
parce que le JIT des kernels GDN de ce modèle hybride crashait. Les graphes ne
s'activent que sur demande explicite, pour qu'une box neuve démarre toujours
sur le chemin éprouvé.
"""

from __future__ import annotations

import pytest


def _eager(monkeypatch, value=None):
    if value is None:
        monkeypatch.delenv("RELIQUARY_VLLM_CUDA_GRAPHS", raising=False)
    else:
        monkeypatch.setenv("RELIQUARY_VLLM_CUDA_GRAPHS", value)
    import importlib

    import reliquary.miner.vllm_backend as vb
    importlib.reload(vb)
    return vb.vllm_enforce_eager()


def test_defaut_reste_eager(monkeypatch):
    """Sans variable, on garde le chemin du bring-up (pas de graphes)."""
    assert _eager(monkeypatch) is True


def test_cuda_graphs_actives_explicitement(monkeypatch):
    """=1 active les graphes (donc enforce_eager False)."""
    assert _eager(monkeypatch, "1") is False


def test_valeur_zero_reste_eager(monkeypatch):
    assert _eager(monkeypatch, "0") is True


def test_valeur_invalide_retombe_sur_le_defaut_sur(monkeypatch):
    """Une faute de frappe ne doit pas activer silencieusement un chemin
    dont la conformite n'a pas ete validee sur cette box."""
    assert _eager(monkeypatch, "yes") is True
