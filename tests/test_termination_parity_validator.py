"""Parité EXACTE avec le prédicat de terminaison du validateur.

Cause racine des 21 verdicts ``bad_termination`` (fenêtres 27585→27595,
``reject_stage=termination_preflight``) :

* Le validateur (``server.py::_preflight`` / ``admission._classify_termination``)
  exige **exactement UN EOS, en DERNIÈRE position** de la complétion.
* Notre contrôle local ne testait que ``last_token in eos_ids`` — un EOS au
  MILIEU passait inaperçu.
* Et le chemin vLLM (production) ne tronquait pas au premier EOS, alors que le
  chemin HF le fait depuis toujours (``first_eos_index``, engine.py:2461). Un
  EOS mid-stream partait donc tel quel dans la soumission.

``ignore_eos=True`` (ajouté pour récupérer ``stop_reason``) aggrave le cas :
l'EOS du modèle n'arrête plus la génération, donc il peut apparaître au milieu.

Ces tests verrouillent le prédicat validateur et la troncature.
"""
import pytest

from reliquary.miner.engine import validator_termination_ok, truncate_at_first_eos

EOS = {248044, 248046}  # <|endoftext|>, <|im_end|> du checkpoint 4B


class TestValidatorPredicate:
    """Miroir de admission._classify_termination (branche EOS)."""

    def test_single_eos_at_end_is_ok(self):
        assert validator_termination_ok([1, 2, 3, 248046], EOS) is True

    def test_no_eos_is_rejected(self):
        assert validator_termination_ok([1, 2, 3], EOS) is False

    def test_eos_in_middle_is_rejected(self):
        assert validator_termination_ok([1, 248046, 3], EOS) is False

    def test_two_eos_is_rejected(self):
        """Le cas qu'ignore_eos rend possible : endoftext au milieu, im_end
        à la fin → deux positions EOS → bad_termination."""
        assert validator_termination_ok([1, 248044, 3, 248046], EOS) is False

    def test_empty_completion_is_rejected(self):
        assert validator_termination_ok([], EOS) is False

    def test_eos_only_completion_is_ok(self):
        assert validator_termination_ok([248046], EOS) is True


class TestTruncateAtFirstEos:
    """Parité avec le chemin HF : couper juste après le premier EOS."""

    def test_truncates_after_first_eos(self):
        assert truncate_at_first_eos([1, 248046, 3, 4], EOS) == [1, 248046]

    def test_keeps_sequence_without_eos(self):
        assert truncate_at_first_eos([1, 2, 3], EOS) == [1, 2, 3]

    def test_already_ending_with_eos_unchanged(self):
        assert truncate_at_first_eos([1, 2, 248046], EOS) == [1, 2, 248046]

    def test_two_eos_keeps_only_first(self):
        assert truncate_at_first_eos([1, 248044, 3, 248046], EOS) == [1, 248044]

    def test_truncated_result_always_passes_validator(self):
        """Propriété : toute séquence contenant un EOS devient valide après
        troncature — c'est ce qui transforme un rejet en soumission saine."""
        for seq in ([1, 248046, 3], [248044, 9, 248046], [1, 2, 248044]):
            out = truncate_at_first_eos(seq, EOS)
            assert validator_termination_ok(out, EOS) is True

    def test_no_eos_stays_invalid_after_truncation(self):
        """Sans EOS il n'y a rien à sauver : le rollout reste tronqué (cap) et
        la garde §5 doit le jeter — on ne fabrique pas une fin artificielle."""
        out = truncate_at_first_eos([1, 2, 3], EOS)
        assert validator_termination_ok(out, EOS) is False
