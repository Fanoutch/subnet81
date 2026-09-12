"""Quota de soumissions par hotkey et par fenêtre — parité V1/fill-closed.

⚡ MESURÉ LE 10/09, EN VOL. Notre port plafonnait à ``2·B_BATCH`` = 32 quelle
que soit la version, alors que le validateur live (image ``6b17632``, = tip de
``codex/observed-checkpoint-restart``, mergée le jour même dans ``main``
par #240) calcule ``2 · FILL_CLOSED_TARGET_GROUPS_PER_ENV`` = **512** sous
fill-closed. La doc mergée le même jour (#241, ``docs/mining.md``) l'écrit
noir sur blanc : « 512 attempts with V6 fill-closed enabled, or 32 in V4/V5 ».

Ce que la bride coûtait, mesuré sur 12 fenêtres du journal de la box :
50 à 70 groupes GÉNÉRÉS par fenêtre, 46 à 59 marqués PAYABLES… et
exactement 32 envoyés. **On jetait 14 à 27 groupes payables par fenêtre.**
Et ``/state`` montrait au même instant ``admitted[opencodeinstruct] = 91``
sur un ``admission_budget`` de 512 : la place existait.

⚠️ Le plafond protocolaire n'est PAS une cible : chaque envoi porte une
preuve GRAIL sur la même carte que la génération (file mesurée : 4 preuves
concurrentes → 5,76 s p50). D'où l'override d'environnement, pour monter par
paliers et replier en UNE variable sans redéploiement de code.
"""
import os
import subprocess
import sys


def _quota(**env_extra) -> int:
    """Valeur EFFECTIVE dans un interpréteur neuf : la constante est calculée
    à l'import, un monkeypatch en cours de process ne la refléterait pas."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("RELIQUARY_")}
    env.update(env_extra)
    out = subprocess.run(
        [sys.executable, "-c",
         "from reliquary import constants as c; "
         "print(c.MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW)"],
        env=env, check=True, capture_output=True, text=True,
    )
    return int(out.stdout.strip())


class TestParityWithTheLiveValidator:
    def test_v6_matches_the_fill_closed_ceiling(self):
        assert _quota(RELIQUARY_PROTOCOL_VERSION="6") == 512

    def test_v5_is_unchanged(self):
        """Non-régression : le repli protocolaire doit rester byte-identique."""
        assert _quota(RELIQUARY_PROTOCOL_VERSION="5") == 32

    def test_v4_is_unchanged(self):
        assert _quota(RELIQUARY_PROTOCOL_VERSION="4") == 32


class TestStagedOverride:
    def test_override_allows_a_staged_raise(self):
        assert _quota(RELIQUARY_PROTOCOL_VERSION="6",
                      RELIQUARY_MAX_SUBMISSIONS_PER_WINDOW="64") == 64

    def test_override_allows_falling_back_to_the_old_bridle(self):
        """Repli en UNE variable, sans redéploiement ni revert de code."""
        assert _quota(RELIQUARY_PROTOCOL_VERSION="6",
                      RELIQUARY_MAX_SUBMISSIONS_PER_WINDOW="32") == 32

    def test_override_is_clamped_to_the_protocol_ceiling(self):
        """Au-delà du plafond le validateur répond RATE_LIMITED : des preuves
        GPU brûlées pour rien. On refuse de dépasser."""
        assert _quota(RELIQUARY_PROTOCOL_VERSION="6",
                      RELIQUARY_MAX_SUBMISSIONS_PER_WINDOW="10000") == 512
        assert _quota(RELIQUARY_PROTOCOL_VERSION="5",
                      RELIQUARY_MAX_SUBMISSIONS_PER_WINDOW="10000") == 32

    def test_garbage_override_falls_back_to_the_default(self):
        assert _quota(RELIQUARY_PROTOCOL_VERSION="6",
                      RELIQUARY_MAX_SUBMISSIONS_PER_WINDOW="beaucoup") == 512

    def test_non_positive_override_falls_back_to_the_default(self):
        assert _quota(RELIQUARY_PROTOCOL_VERSION="6",
                      RELIQUARY_MAX_SUBMISSIONS_PER_WINDOW="0") == 512


class TestCeilingMustNotBeDerivedLocally:
    """GARDE-PIÈGE (12/09) — le plafond ne doit PAS dériver de NOTRE
    `CHECKPOINT_PUBLISH_INTERVAL_WINDOWS`.

    Upstream a changé la formule le 12/09 (image live `84da25f`) :
    `2 * FILL_CLOSED_TARGET_GROUPS_PER_ENV` est devenu
    `2 * FILL_CLOSED_EMISSIONS_PER_WINDOW * B_BATCH`, où
    `FILL_CLOSED_EMISSIONS_PER_WINDOW = CHECKPOINT_PUBLISH_INTERVAL_WINDOWS`.
    Motif upstream : ils ont rendu la CIBLE d'admission réductible
    (`FILL_CLOSED_PICKS_PER_WINDOW`) — d'où la décrue observée des
    `admission_budgets` 512 → 320 → 224 — donc épingler le quota sur la
    constante immuable le garde à 512 quand le budget, lui, descend.

    ⚠️ Le piège : chez eux `CHECKPOINT_PUBLISH_INTERVAL_WINDOWS` vaut **16**,
    chez nous **10** (constante de leur stub validateur, `validator/service.py`
    — jamais lue par le mineur). Reproduire leur formule avec NOTRE constante
    donnerait `2 × 10 × 16 = 320` : on se sous-plafonnerait de 37 % en silence.
    Le `256` en dur est donc correct PARCE QU'il ne dérive pas.

    Test de caractérisation, pas de TDD : il fige un invariant contre une
    « simplification » future, il ne décrit pas un comportement neuf.
    """

    def test_ceiling_is_512_and_not_the_derived_value(self):
        from reliquary import constants as c
        derive = 2 * c.CHECKPOINT_PUBLISH_INTERVAL_WINDOWS * c.B_BATCH
        assert _quota(RELIQUARY_PROTOCOL_VERSION="6") == 512
        assert derive != 512, (
            "notre CHECKPOINT_PUBLISH_INTERVAL_WINDOWS a rejoint la valeur "
            "upstream (16) : la garde perd son sens, relire le commentaire"
        )

    def test_the_local_constant_is_validator_stub_only(self):
        """Si le mineur se met à la lire, la divergence 10≠16 cesse d'être inerte."""
        import pathlib
        root = pathlib.Path(__file__).resolve().parent.parent / "reliquary"
        users = [
            p for p in root.rglob("*.py")
            if "CHECKPOINT_PUBLISH_INTERVAL_WINDOWS" in p.read_text()
            and p.name != "constants.py"
        ]
        assert [p.name for p in users] == ["service.py"], (
            f"nouveau lecteur de la constante : {[str(p) for p in users]}"
        )
