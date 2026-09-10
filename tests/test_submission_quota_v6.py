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
