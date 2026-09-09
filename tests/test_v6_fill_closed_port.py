"""Port de conformité v6 « fill-closed » (upstream PR #224,
`integration/reliquary-v1-final` @ 589fd43, 2026-09-07).

Le déploiement annoncé (docs/mining.md, docker/.env, CI) est le profil
``qwen3-4b-base-dapo-fill-closed-v6`` / protocol_version 6. Génération
byte-identique v5 ; ce qui change pour le mineur est le WIRE :
  - ``/state`` porte un objet ``fill_closed`` dès que la capacité est active
    (notre ``GrpoBatchState`` est ``extra="forbid"`` → sans le champ, chaque
    poll lève ValidationError et le mineur ne génère jamais, cf. 2026-08-03) ;
  - ``RejectReason.reveal_not_selected`` (HTTP 200) ;
  - ``generation_profile_id`` v6 annoncé sur le wire + domaine forced-seed v6.
"""
import hashlib
import importlib
import json

import pytest
from pydantic import ValidationError

from reliquary.protocol.submission import (
    BatchSubmissionResponse,
    GrpoBatchState,
    RejectReason,
)

V6_FILL_CLOSED = {
    "phase": "collecting",
    "generation_beacon_chain": "52db9ba70e0cc7a6",
    "generation_beacon_chain_hash": "a" * 64,
    "generation_beacon_round": 30123456,
    "precommit_cutoff_ts": 1789000000.0,
    "precommit_seconds": 1767.0,
    "max_window_seconds": 1800.0,
    "picks_emitted": 3,
    "picks_target": 16,
    "picks_by_environment": {"openmathinstruct": 3, "opencodeinstruct": 3},
    "admission_budgets": {"openmathinstruct": 512, "opencodeinstruct": 512},
    "admitted": {"openmathinstruct": 70, "opencodeinstruct": 64},
    "proven": {"openmathinstruct": 60, "opencodeinstruct": 55},
    "in_flight": {"openmathinstruct": 4, "opencodeinstruct": 3},
    "remaining": {"openmathinstruct": 442, "opencodeinstruct": 448},
}

V6_STATE = {
    "state": "open",
    "window_n": 44000,
    "anchor_block": 44000,
    "cooldown_prompts": [4174, 4302],
    "valid_submissions": 12,
    "checkpoint_n": 7,
    "checkpoint_repo_id": "ReliquaryForge/qwen3-4b-reliquary-v6",
    "checkpoint_revision": "7ea1150e81ca229519b1d137ef5f8ec2c63ce361",
    "protocol_version": 6,
    "generation_profile_id": "qwen3-4b-base-dapo-fill-closed-v6",
    "generation_contract": {
        "profile_id": "qwen3-4b-base-dapo-fill-closed-v6",
        "protocol_version": 6,
        "throughput_tiebreak": None,
    },
    "fill_closed": V6_FILL_CLOSED,
    "randomness": "a42a203fd7f0" + "0" * 52,
}


def test_state_v6_avec_fill_closed_se_parse():
    st = GrpoBatchState.model_validate(V6_STATE)
    assert st.window_n == 44000
    assert st.fill_closed is not None
    assert st.fill_closed.phase == "collecting"
    assert st.fill_closed.picks_target == 16
    assert st.fill_closed.remaining["opencodeinstruct"] == 448


def test_state_v5_sans_fill_closed_reste_valide():
    v5 = {k: v for k, v in V6_STATE.items() if k != "fill_closed"}
    st = GrpoBatchState.model_validate(v5)
    assert st.fill_closed is None


def test_fill_closed_tolere_un_champ_inconnu_ajoute_par_upstream():
    """Le sous-objet est de la télémétrie : un champ de plus ne doit pas
    tuer le poll /state (contrairement au niveau racine, volontairement
    strict)."""
    st = GrpoBatchState.model_validate(
        {**V6_STATE, "fill_closed": {**V6_FILL_CLOSED, "nouveau_champ": 1}}
    )
    assert st.fill_closed.phase == "collecting"


def test_reveal_not_selected_est_un_motif_connu():
    resp = BatchSubmissionResponse.model_validate(
        {"accepted": False, "reason": "reveal_not_selected"}
    )
    assert resp.reason is RejectReason.REVEAL_NOT_SELECTED


@pytest.fixture(autouse=True)
def _constants_propres(monkeypatch):
    """Tout test qui recharge ``reliquary.constants`` sous un autre protocole
    le rend à la suite avec l'env par défaut (pas de pollution croisée)."""
    yield
    monkeypatch.undo()
    import reliquary.constants as c
    importlib.reload(c)


@pytest.fixture
def constants_v6(monkeypatch):
    monkeypatch.setenv("RELIQUARY_PROTOCOL_VERSION", "6")
    monkeypatch.delenv("RELIQUARY_GENERATION_PROFILE_ID", raising=False)
    import reliquary.constants as c
    return importlib.reload(c)


def test_v6_annonce_le_profil_fill_closed(constants_v6):
    assert constants_v6.PROTOCOL_VERSION == 6
    assert constants_v6.GENERATION_PROFILE_ID == "qwen3-4b-base-dapo-fill-closed-v6"
    assert constants_v6.generation_profile_id({"RELIQUARY_PROTOCOL_VERSION": "6"}) \
        == "qwen3-4b-base-dapo-fill-closed-v6"


def test_v6_domaine_forced_seed_et_generation_identiques_a_v5(constants_v6):
    c = constants_v6
    assert c.FORCED_SEED_DOMAIN == "reliquary-forced-seed-v6"
    assert c.FORCED_SEED_PROTOCOL_VERSION == 6
    # v6 = v5 champ pour champ côté génération (upstream profiles.py)
    assert (c.M_ROLLOUTS, c.B_BATCH, c.T_PROTO, c.TOP_P_PROTO, c.TOP_K_PROTO) \
        == (16, 16, 1.0, 1.0, 0)
    assert c.MAX_NEW_TOKENS_PROTOCOL_CAP == 8192
    assert c.BFT_ENABLED is False
    from reliquary.protocol.profiles import prompt_template_for
    assert prompt_template_for("opencodeinstruct", protocol_version=6) \
        is prompt_template_for("opencodeinstruct", protocol_version=5)


def test_v5_reste_inchange(monkeypatch):
    monkeypatch.setenv("RELIQUARY_PROTOCOL_VERSION", "5")
    monkeypatch.delenv("RELIQUARY_GENERATION_PROFILE_ID", raising=False)
    import reliquary.constants as c
    c = importlib.reload(c)
    assert c.GENERATION_PROFILE_ID == "qwen3-4b-base-dapo-reasoning-v5"
    assert c.FORCED_SEED_DOMAIN == "reliquary-forced-seed-v5"


def test_fill_closed_phase_draining_et_beacon_absents():
    """Phase ``draining`` sans les 3 ``generation_beacon_*`` (fenêtre en
    vidage, beacon non encore publié) → parse, champs à None."""
    fc = {k: v for k, v in V6_FILL_CLOSED.items()
          if not k.startswith("generation_beacon")}
    st = GrpoBatchState.model_validate({**V6_STATE, "fill_closed": {**fc, "phase": "draining"}})
    assert st.fill_closed.phase == "draining"
    assert st.fill_closed.generation_beacon_chain is None
    assert st.fill_closed.generation_beacon_round is None
    assert st.fill_closed.precommit_seconds == 1767.0


def test_fill_closed_exige_phase():
    with pytest.raises(ValidationError):
        GrpoBatchState.model_validate(
            {**V6_STATE, "fill_closed": {"picks_target": 16}}
        )


def test_state_v6_racine_reste_stricte():
    """La racine de /state garde ``extra="forbid"`` : un champ inconnu à la
    RACINE = upgrade protocole à auditer, pas de tolérance silencieuse."""
    with pytest.raises(ValidationError):
        GrpoBatchState.model_validate({**V6_STATE, "champ_inconnu": 1})


def test_state_checkpoint_n_absent_ou_none_se_parse():
    """``checkpoint_n`` devient optionnel (rapport A §4.2) ; un int reste
    parsé tel quel et un négatif reste refusé."""
    sans = {k: v for k, v in V6_STATE.items() if k != "checkpoint_n"}
    assert GrpoBatchState.model_validate(sans).checkpoint_n is None
    assert GrpoBatchState.model_validate({**V6_STATE, "checkpoint_n": None}).checkpoint_n is None
    assert GrpoBatchState.model_validate(V6_STATE).checkpoint_n == 7
    with pytest.raises(ValidationError):
        GrpoBatchState.model_validate({**V6_STATE, "checkpoint_n": -1})


# Contrat v5 relevé sur le validateur live (image 84dcc57) — dict-égal à la
# reconstruction depuis profiles.py de la branche upstream (rapport A §3).
_TPL_CODE = ("Solve the following programming problem step by step.\n\n"
             "$problem$contract\n\nAfter your reasoning, provide the final "
             "implementation in the last fenced Python code block.")
_TPL_MATH = ("Solve the following math problem step by step.\n\n$problem\n\n"
             "Put your final answer within \\boxed{}.")
CONTRACT_V5 = {
    "collection_seconds": 100,
    "environments": {
        "opencodeinstruct": {
            "answer_format": None, "bft": None, "max_new_tokens": 8192,
            "prompt_template": {
                "id": "opencodeinstruct-step-by-step-v1",
                "renderer": "dollar-substitution-v1",
                "sha256": "47f2d9e16e786ae0245c85632934e08c640be8cf4b959973e9aba5f296a63380",
                "template": _TPL_CODE,
            },
        },
        "openmathinstruct": {
            "answer_format": "boxed", "bft": None, "max_new_tokens": 8192,
            "prompt_template": {
                "id": "openmathinstruct-step-by-step-v1",
                "renderer": "dollar-substitution-v1",
                "sha256": "7f2343051fdd2a4179ddab8202cdfd765438aa3b31022c4ef8a06a5f502899c4",
                "template": _TPL_MATH,
            },
        },
    },
    "model_id": "Qwen/Qwen3-4B-Base",
    "model_revision": "906bfd4b4dc7f14ee4320094d8b41684abff8539",
    "profile_id": "qwen3-4b-base-dapo-reasoning-v5",
    "prompt_encoding": "raw",
    "protocol_version": 5,
    "sampling": {"do_sample": False, "rollouts": 16, "temperature": 1.0,
                 "top_k": 0, "top_p": 1.0},
    "throughput_tiebreak": {"bucket_tokens_per_round": 50, "token_cap": 8192},
    "upload_grace_seconds": 33,
}
CONTRACT_SHA_V5 = "19e98f5a3ddac1980efe66fd80db1ec0f8db87a5e60934efd5d0e8985435eadd"
CONTRACT_SHA_V6 = "1696eef2a8ff52284842f2253d6f699b50bc657dc93b20fc61a257db7d449385"


def _contract_sha(contract: dict) -> str:
    """Hachage canonique du validateur (shared/training_payload.py:74-83)."""
    return hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_contrat_v6_sha256(constants_v6):
    """Contrat v6 = v5 sauf profile_id / protocol_version / throughput_tiebreak
    → sha canonique ``1696eef2…`` ; nos templates et notre profil annoncé
    sont ceux du contrat."""
    from reliquary.protocol.profiles import prompt_template_for

    assert _contract_sha(CONTRACT_V5) == CONTRACT_SHA_V5
    contract_v6 = {
        **CONTRACT_V5,
        "profile_id": constants_v6.GENERATION_PROFILE_ID,
        "protocol_version": constants_v6.PROTOCOL_VERSION,
        "throughput_tiebreak": None,
    }
    assert _contract_sha(contract_v6) == CONTRACT_SHA_V6
    for env_name, env_c in contract_v6["environments"].items():
        tpl = prompt_template_for(env_name, protocol_version=6)
        assert tpl.template_id == env_c["prompt_template"]["id"]
        assert tpl.sha256() == env_c["prompt_template"]["sha256"]
        assert tpl.template == env_c["prompt_template"]["template"]
    assert contract_v6["sampling"]["rollouts"] == constants_v6.M_ROLLOUTS
    assert contract_v6["environments"]["opencodeinstruct"]["max_new_tokens"] \
        == constants_v6.MAX_NEW_TOKENS_PROTOCOL_CAP


def test_neutral_kwargs_min_length_none():
    """``min_length: None`` (parité upstream, transformers 5.9 sans warning) ;
    ``min_new_tokens`` reste 0 → aucun processeur de longueur ajouté."""
    from reliquary.miner.forced_seed_sampler import _NEUTRAL_PROCESSOR_KWARGS

    assert _NEUTRAL_PROCESSOR_KWARGS["min_length"] is None
    assert _NEUTRAL_PROCESSOR_KWARGS["min_new_tokens"] == 0
