"""Câblage du template v5 DANS get_problem — le prompt réellement soumis.

⚠️ DEUX PIÈGES RENCONTRÉS EN ÉCRIVANT CE FICHIER, tous deux corrigés ici :

1. Une première version répliquait la règle de composition dans le test au
   lieu d'appeler `get_problem`. Elle passait avant même que le code soit
   câblé — un test qui passe du premier coup ne prouve rien.
2. Une seconde version basculait la version de protocole par
   `importlib.reload()`. Les rechargements laissaient `constants` et
   l'environnement figés en v5 pour la suite : 13 tests d'AUTRES fichiers
   tombaient alors que le code de production était sain. Un test qui casse
   ses voisins est un test cassé.

`render_active_prompt` importe `PROTOCOL_VERSION` À L'INTÉRIEUR de la
fonction : un `monkeypatch` sur l'attribut suffit donc, sans rechargement et
sans effet de bord sur les autres tests.

Le dataset réel est lourd (14 M de lignes) : on instancie l'environnement sans
passer par `__init__` et on injecte une ligne. Tout le reste — contrat de
grader, hachage, cases — est le code de production.
"""
from __future__ import annotations

import hashlib

import pytest

from reliquary.environment.opencodeinstruct import OpenCodeInstructEnvironment

LIGNE = {
    "input": "Write a function that adds two numbers.",
    "output": "def add(a, b):\n    return a + b",
    "id": "test-1",
}


def _prompt(monkeypatch, version: int) -> str:
    """Prompt produit par le VRAI get_problem sous la version demandée."""
    monkeypatch.setattr("reliquary.constants.PROTOCOL_VERSION", version)
    env = object.__new__(OpenCodeInstructEnvironment)
    env._dataset = [LIGNE]
    env._cases_by_id = {}
    return env.get_problem(0)["prompt"]


def test_v4_le_prompt_ne_contient_PAS_le_preambule_du_template(monkeypatch):
    """SÛRETÉ DU REPLI : sous v4, comportement legacy strictement inchangé."""
    p = _prompt(monkeypatch, 4)
    assert not p.startswith("Solve the following programming problem")
    assert p.startswith("Write a function that adds two numbers.")


def test_v5_le_prompt_EST_enveloppe_dans_le_template_du_validateur(monkeypatch):
    p = _prompt(monkeypatch, 5)
    assert p.startswith("Solve the following programming problem step by step.\n\n")
    assert p.rstrip().endswith("last fenced Python code block.")


def test_v5_l_enonce_et_le_contrat_survivent_intacts_dans_le_rendu(monkeypatch):
    """Le template enveloppe, il ne retire rien de l'énoncé ni du contrat."""
    legacy = _prompt(monkeypatch, 4)
    v5 = _prompt(monkeypatch, 5)
    assert legacy in v5


def test_le_problem_id_suit_le_prompt_soumis(monkeypatch):
    """Le problem_id est un sha256 du prompt : il DOIT changer avec lui, sinon
    on annoncerait au validateur un id qui ne correspond pas au prompt."""
    monkeypatch.setattr("reliquary.constants.PROTOCOL_VERSION", 4)
    e4 = object.__new__(OpenCodeInstructEnvironment)
    e4._dataset = [LIGNE]; e4._cases_by_id = {}
    r4 = e4.get_problem(0)

    monkeypatch.setattr("reliquary.constants.PROTOCOL_VERSION", 5)
    e5 = object.__new__(OpenCodeInstructEnvironment)
    e5._dataset = [LIGNE]; e5._cases_by_id = {}
    r5 = e5.get_problem(0)

    assert r4["id"] != r5["id"]
    assert r5["id"] == hashlib.sha256(r5["prompt"].encode()).hexdigest()[:16]
