from reliquary.miner.engine import MiningEngine
from reliquary.miner.mix_controller import MixController
from reliquary.protocol.submission import VerdictsResponse


def _engine_with_mix(envs):
    e = object.__new__(MiningEngine)
    e.active_envs = list(envs)
    e._mix = MixController(envs, total_slots=8, slot_floor=1, alpha=1.0)
    e._submitted_env = {}
    return e


def test_apply_verdicts_records_rewarded_outcome_for_mapped_env():
    e = _engine_with_mix(["math", "code"])
    e._submitted_env = {"a" * 64: "code"}
    payload = {"verdicts": [{"merkle_root": "a" * 64, "accepted": True,
               "reason": "accepted", "ts": 9.0, "rewarded": True}]}
    e._apply_verdicts(VerdictsResponse.model_validate(payload))
    slots = e._mix.target_slots()
    assert slots["code"] >= slots["math"]   # code a payé → reçoit ≥


def test_apply_verdicts_ignores_unknown_merkle_root():
    e = _engine_with_mix(["math"])
    payload = {"verdicts": [{"merkle_root": "f" * 64, "accepted": True,
               "reason": "accepted", "ts": 1.0, "rewarded": True}]}
    # merkle inconnu → pas de crash, pas d'outcome
    assert e._apply_verdicts(VerdictsResponse.model_validate(payload)) == 1.0


def test_apply_verdicts_skips_when_rewarded_is_none():
    e = _engine_with_mix(["math"])
    e._submitted_env = {"b" * 64: "math"}
    payload = {"verdicts": [{"merkle_root": "b" * 64, "accepted": False,
               "reason": "grail_fail", "ts": 2.0}]}  # rewarded absent → None
    e._apply_verdicts(VerdictsResponse.model_validate(payload))  # ne lève pas


def test_apply_verdicts_returns_max_ts_for_cursor():
    e = _engine_with_mix(["math"])
    payload = {"verdicts": [
        {"merkle_root": "c" * 64, "accepted": True, "reason": "accepted", "ts": 3.0},
        {"merkle_root": "d" * 64, "accepted": True, "reason": "accepted", "ts": 7.5},
    ]}
    assert e._apply_verdicts(VerdictsResponse.model_validate(payload)) == 7.5


# ── V1 / fill-closed : le signal par groupe ne porte PLUS le paiement ────────
# Vérifié sur le validateur live le 2026-09-10 : sous fill-closed le paiement
# est une PART agrégée (`rewards_by_hotkey` de l'archive R2). Dans /verdicts,
# `rewarded` vaut False pour les groupes NON retenus et est ABSENT (None) pour
# les autres — y compris les payés. Alimenter le MixController avec ça, c'est
# ne lui donner QUE des échecs : son EMA de rendement tombe à 0 pour chaque
# env. Inerte à un seul env, mais il pilote la répartition dès qu'on en ouvre
# deux — c'est-à-dire au moment précis où le levier compte.
# Mesure du jour : 71 verdicts `rewarded=False` sur une fenêtre où R2 nous
# donnait 3e sur 19, part 0,01247.

def _engine_v6(envs, monkeypatch):
    import reliquary.constants as c
    monkeypatch.setattr(c, "PROTOCOL_VERSION", 6, raising=False)
    return _engine_with_mix(envs)


def test_v6_un_rewarded_false_ne_penalise_pas_l_env(monkeypatch):
    """Sous fill-closed, `rewarded=False` n'est PAS une preuve de non-paiement
    exploitable : il ne doit pas faire chuter le rendement de son env."""
    e = _engine_v6(["math", "code"], monkeypatch)
    e._submitted_env = {"a" * 64: "code"}
    avant = e._mix.target_slots()
    payload = {"verdicts": [{"merkle_root": "a" * 64, "accepted": True,
               "reason": "accepted", "ts": 9.0, "rewarded": False}]}
    e._apply_verdicts(VerdictsResponse.model_validate(payload))
    assert e._mix.target_slots() == avant, (
        "le mix a bougé sur un signal qui ne porte pas le paiement sous v6"
    )


def test_v5_conserve_l_apprentissage_du_rendement(monkeypatch):
    """Garde : sous v5 le signal est bon, on continue d'apprendre."""
    import reliquary.constants as c
    monkeypatch.setattr(c, "PROTOCOL_VERSION", 5, raising=False)
    e = _engine_with_mix(["math", "code"])
    e._submitted_env = {"a" * 64: "code", "b" * 64: "math"}
    payload = {"verdicts": [
        {"merkle_root": "a" * 64, "accepted": True, "reason": "accepted",
         "ts": 9.0, "rewarded": True},
        {"merkle_root": "b" * 64, "accepted": True, "reason": "accepted",
         "ts": 9.1, "rewarded": False},
    ]}
    e._apply_verdicts(VerdictsResponse.model_validate(payload))
    slots = e._mix.target_slots()
    assert slots["code"] > slots["math"], "v5 doit toujours apprendre du rendement"
