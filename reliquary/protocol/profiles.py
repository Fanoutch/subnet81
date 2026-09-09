"""Profils de PROMPT par protocole — port v5 (upstream PR #190, live 23/08).

Jusqu'en v4, le prompt envoyé au modèle était une simple concaténation
``problème + contrat``. Depuis v5 le validateur le rend via un TEMPLATE
versionné, et publie ce template dans son ``generation_contract``. Le prompt
déterminant les tokens, et les tokens le forced-seed, un écart d'un seul
caractère fait rejeter 100 % des rollouts.

⚠️ LES TEMPLATES CI-DESSOUS SONT RELEVÉS SUR LE VALIDATEUR LIVE, pas déduits
du code : `/health` → ``generation_contract.environments.<env>.prompt_template``
(relevé le 24/08, image ``cba84ce``). ``tests/test_prompt_template_v5.py`` les
fige octet pour octet. Avant tout redéploiement après un changement upstream,
re-comparer le ``sha256()`` au contrat publié.

Propriété de sûreté : sous protocole < 5, ``render_active_prompt`` renvoie
``None`` et l'appelant garde le chemin legacy — le repli v4 reste byte-exact.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from string import Template

# Placeholders admis dans un template. `$problem` est OBLIGATOIRE : sans lui
# le prompt ne contiendrait pas l'énoncé, ce qui passerait silencieusement.
_PLACEHOLDERS = frozenset({"problem", "contract"})


@dataclass(frozen=True)
class PromptTemplateProfile:
    """Texte de prompt exact et règle de rendu pour un environnement.

    Le template utilise les placeholders `$problem` et `$contract` de
    ``string.Template`` — et non ``str.format`` — pour qu'une accolade
    littérale (`\\boxed{}` en math) n'ait pas à être échappée.
    """

    template_id: str
    template: str

    def __post_init__(self) -> None:
        if not self.template_id:
            raise ValueError("l'identifiant de template ne peut pas être vide")
        inconnus = {
            m.group("named") or m.group("braced")
            for m in Template.pattern.finditer(self.template)
            if (m.group("named") or m.group("braced"))
        } - _PLACEHOLDERS
        if inconnus:
            raise ValueError(
                f"template {self.template_id!r} : placeholders inconnus "
                f"{sorted(inconnus)}"
            )
        if "$problem" not in self.template and "${problem}" not in self.template:
            raise ValueError(
                f"template {self.template_id!r} doit contenir $problem"
            )

    def sha256(self) -> str:
        """Empreinte du template — sert à vérifier la parité avec le contrat
        publié sans comparer des chaînes multi-lignes dans les journaux."""
        return hashlib.sha256(self.template.encode("utf-8")).hexdigest()

    def render(self, *, problem: str, contract: str = "") -> str:
        return Template(self.template).substitute(
            problem=problem, contract=contract,
        )


# Templates v5, relevés sur le validateur live (image cba84ce, 24/08).
_TEMPLATES_V5 = {
    "opencodeinstruct": PromptTemplateProfile(
        template_id="opencodeinstruct-step-by-step-v1",
        template=(
            "Solve the following programming problem step by step.\n"
            "\n"
            "$problem$contract\n"
            "\n"
            "After your reasoning, provide the final implementation in the "
            "last fenced Python code block."
        ),
    ),
    "openmathinstruct": PromptTemplateProfile(
        template_id="openmathinstruct-step-by-step-v1",
        template=(
            "Solve the following math problem step by step.\n"
            "\n"
            "$problem\n"
            "\n"
            "Put your final answer within \\boxed{}."
        ),
    ),
}


def prompt_template_for(
    environment: str, *, protocol_version: int | None = None,
) -> PromptTemplateProfile | None:
    """Template du protocole actif, ou ``None`` si le protocole n'en a pas.

    ``None`` signifie « garder le chemin legacy » : c'est le cas de tout
    protocole < 5, et d'un environnement non profilé.
    """
    if protocol_version is None:
        from reliquary.constants import PROTOCOL_VERSION

        protocol_version = PROTOCOL_VERSION
    if protocol_version < 5:
        return None
    return _TEMPLATES_V5.get(environment)


def render_active_prompt(
    environment: str, *, problem: str, contract: str = "",
    protocol_version: int | None = None,
) -> str | None:
    """Prompt rendu pour le protocole actif, ou ``None`` pour rester en legacy.

    Ne lève jamais : un environnement inconnu retombe sur ``None`` plutôt que
    de faire tomber le bake en cours de fenêtre.
    """
    tpl = prompt_template_for(environment, protocol_version=protocol_version)
    if tpl is None:
        return None
    return tpl.render(problem=problem, contract=contract)
