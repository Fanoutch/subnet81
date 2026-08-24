"""Port v5 : le prompt passe d'une concaténation à un TEMPLATE versionné.

Le validateur (PR #190, live depuis le 23/08) rend désormais le prompt via
``string.Template`` au lieu de concaténer `problème + contrat`. Le prompt
changeant, les tokens changent, donc le forced-seed ne correspond plus : sans
ce port, 100 % de nos rollouts sont rejetés.

⚠️ LA RÉFÉRENCE EST LE CONTRAT PUBLIÉ, pas notre lecture du code upstream. Le
template exact est exposé dans `/health` → `generation_contract.environments.
<env>.prompt_template.template`. Les tests ci-dessous figent cette chaîne
OCTET POUR OCTET (relevée le 24/08 sur le validateur live, image cba84ce) :
un écart d'un espace suffit à tout faire rejeter.

Propriété de sûreté exigée : en v4 le rendu doit être STRICTEMENT identique à
l'ancien comportement, pour que le port ne casse pas le repli.
"""
from __future__ import annotations

import pytest

# Templates relevés sur le validateur LIVE le 24/08 (image cba84ce).
TEMPLATE_CODE_LIVE = (
    "Solve the following programming problem step by step.\n"
    "\n"
    "$problem$contract\n"
    "\n"
    "After your reasoning, provide the final implementation in the last "
    "fenced Python code block."
)
TEMPLATE_MATH_LIVE = (
    "Solve the following math problem step by step.\n"
    "\n"
    "$problem\n"
    "\n"
    "Put your final answer within \\boxed{}."
)


def test_le_template_code_porte_est_byte_identique_au_contrat_live():
    """Le template embarqué doit être l'exact octet publié par le validateur."""
    from reliquary.protocol.profiles import prompt_template_for

    tpl = prompt_template_for("opencodeinstruct", protocol_version=5)
    assert tpl is not None, "aucun template v5 pour opencodeinstruct"
    assert tpl.template == TEMPLATE_CODE_LIVE
    assert tpl.template_id == "opencodeinstruct-step-by-step-v1"


def test_le_template_math_porte_est_byte_identique_au_contrat_live():
    from reliquary.protocol.profiles import prompt_template_for

    tpl = prompt_template_for("openmathinstruct", protocol_version=5)
    assert tpl is not None
    assert tpl.template == TEMPLATE_MATH_LIVE
    assert tpl.template_id == "openmathinstruct-step-by-step-v1"


def test_le_rendu_v5_insere_probleme_et_contrat_aux_bonnes_places():
    from reliquary.protocol.profiles import prompt_template_for

    tpl = prompt_template_for("opencodeinstruct", protocol_version=5)
    rendu = tpl.render(problem="ÉNONCÉ", contract="\n\nCONTRAT")
    assert rendu == (
        "Solve the following programming problem step by step.\n"
        "\n"
        "ÉNONCÉ\n\nCONTRAT\n"
        "\n"
        "After your reasoning, provide the final implementation in the last "
        "fenced Python code block."
    )


def test_en_v4_aucun_template_donc_comportement_inchange():
    """SÛRETÉ DU REPLI : le port ne doit rien changer sous v4."""
    from reliquary.protocol.profiles import render_active_prompt

    assert render_active_prompt(
        "opencodeinstruct", problem="P", contract="C", protocol_version=4,
    ) is None


def test_en_v5_le_rendu_actif_produit_le_prompt_complet():
    from reliquary.protocol.profiles import render_active_prompt

    rendu = render_active_prompt(
        "opencodeinstruct", problem="P", contract="C", protocol_version=5,
    )
    assert rendu is not None
    assert rendu.startswith("Solve the following programming problem")
    assert "PC" in rendu


def test_un_environnement_inconnu_ne_leve_pas_et_retombe_sur_le_legacy():
    """Un env non profilé ne doit pas faire tomber le bake."""
    from reliquary.protocol.profiles import render_active_prompt

    assert render_active_prompt(
        "env_inexistant", problem="P", contract="C", protocol_version=5,
    ) is None


def test_le_template_doit_contenir_problem_sinon_erreur():
    """Garde-fou : un template sans $problem est une erreur de config."""
    from reliquary.protocol.profiles import PromptTemplateProfile

    with pytest.raises(ValueError, match="problem"):
        PromptTemplateProfile(template_id="x", template="pas de placeholder")


def test_le_hash_du_template_permet_de_verifier_la_parite():
    """Le sha256 sert à comparer notre template à celui du contrat sans
    dépendre d'une comparaison de chaîne dans les journaux."""
    import hashlib

    from reliquary.protocol.profiles import prompt_template_for

    tpl = prompt_template_for("opencodeinstruct", protocol_version=5)
    attendu = hashlib.sha256(TEMPLATE_CODE_LIVE.encode("utf-8")).hexdigest()
    assert tpl.sha256() == attendu
