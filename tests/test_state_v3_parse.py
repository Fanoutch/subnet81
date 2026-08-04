"""Le /state du validateur v3 (70e795b) porte 3 champs de plus qu'en v2 :
``protocol_version``, ``generation_profile_id``, ``generation_contract``.

Notre ``GrpoBatchState`` est ``extra="forbid"`` : sans ces champs déclarés,
CHAQUE poll /state lève ValidationError → la boucle du mineur retry en
silence et ne génère JAMAIS (observé live 2026-08-03 : 19 min de polling à
~3/s, 0 génération, GPU 0%). Parité requise avec origin/main.
"""
from reliquary.protocol.submission import GrpoBatchState

V3_STATE = {
    "state": "open",
    "window_n": 27580,
    "anchor_block": 27580,
    "cooldown_prompts": [4174, 4302],
    "valid_submissions": 0,
    "checkpoint_n": 103,
    "checkpoint_repo_id": "ReliquaryForge/qwen3.5-4b-reliquary-v4",
    "checkpoint_revision": "7ea1150e81ca229519b1d137ef5f8ec2c63ce361",
    "randomness": "a42a203fd7f0" + "0" * 52,
    "protocol_version": 3,
    "generation_profile_id": "qwen35-4b-auction-v3",
    "generation_contract": {
        "profile_id": "qwen35-4b-auction-v3",
        "model_id": "Qwen/Qwen3.5-4B",
        "protocol_version": 3,
    },
}


def test_state_v3_payload_parses():
    s = GrpoBatchState.model_validate(V3_STATE)
    assert s.window_n == 27580
    assert s.protocol_version == 3
    assert s.generation_profile_id == "qwen35-4b-auction-v3"
    assert s.generation_contract["model_id"] == "Qwen/Qwen3.5-4B"


def test_state_v2_payload_still_parses():
    v2 = {k: v for k, v in V3_STATE.items()
          if k not in ("protocol_version", "generation_profile_id",
                       "generation_contract")}
    s = GrpoBatchState.model_validate(v2)
    assert s.protocol_version is None
    assert s.generation_profile_id is None
    assert s.generation_contract is None


def test_unknown_extra_still_forbidden():
    import pytest
    with pytest.raises(Exception):
        GrpoBatchState.model_validate({**V3_STATE, "champ_inconnu": 1})
