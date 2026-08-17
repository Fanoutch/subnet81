"""Task 1 du port v4 : constantes gatées sur RELIQUARY_PROTOCOL_VERSION=4.

Contrat = profil upstream ``qwen3-4b-base-dapo-v4`` @ 8c38992. Le défaut
(sans flag) doit rester byte-identique au live v3.
"""
import pytest

from tests.v4helpers import reload_constants


@pytest.fixture(autouse=True)
def _restore_constants(monkeypatch):
    yield
    reload_constants(monkeypatch)


def test_default_is_v3_byte_identical(monkeypatch):
    c = reload_constants(monkeypatch)  # env nettoyé
    assert c.PROTOCOL_VERSION == 3
    assert c.M_ROLLOUTS == 8
    assert (c.T_PROTO, c.TOP_P_PROTO, c.TOP_K_PROTO) == (0.6, 0.95, 20)
    assert c.BFT_ENABLED is True
    assert c.MAX_NEW_TOKENS_PROTOCOL_CAP == 16384
    assert c.B_BATCH == 8
    assert c.MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW == 8
    assert c.MIN_EOS_PROBABILITY == 0.01
    assert c.SAMPLING_MEDIAN_LOW_MAX == 0.30
    assert c.SAMPLING_LOW_Q10_MAX == 0.025
    assert c.FORCED_SEED_ROLLOUT_FLOOR == 0.75
    assert c.FORCED_SEED_CONSISTENCY_FLOOR == 0.80
    assert c.RAW_COMPLETION_PROMPTS is False
    assert c.OMI_TRAIN_SHARDS_ONLY is False
    assert c.FORCED_SEED_DOMAIN == "reliquary-forced-seed-v3"
    assert c.GENERATION_PROFILE_ID == "qwen35-4b-auction-v3"
    assert c.DEFAULT_BASE_MODEL == "Qwen/Qwen3.5-2B"
    assert c.DEFAULT_BASE_MODEL_REVISION is None


def test_v4_values(monkeypatch):
    c = reload_constants(monkeypatch, 4)
    assert c.PROTOCOL_VERSION == 4
    assert c.M_ROLLOUTS == 16
    assert (c.T_PROTO, c.TOP_P_PROTO, c.TOP_K_PROTO) == (1.0, 1.0, 0)
    assert c.BFT_ENABLED is False
    assert c.MAX_NEW_TOKENS_PROTOCOL_CAP == 8192
    assert c.B_BATCH == 16
    assert c.MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW == 32
    assert c.MIN_EOS_PROBABILITY == 0.001
    assert c.SAMPLING_MEDIAN_LOW_MAX == 0.05
    assert c.SAMPLING_LOW_Q10_MAX == 0.0002
    assert c.FORCED_SEED_ROLLOUT_FLOOR == 0.70
    assert c.FORCED_SEED_CONSISTENCY_FLOOR == 0.80  # inchangé en v4
    assert c.RAW_COMPLETION_PROMPTS is True
    assert c.OMI_TRAIN_SHARDS_ONLY is True
    assert c.FORCED_SEED_DOMAIN == "reliquary-forced-seed-v4"
    # Le batcher v4 rejette GENERATION_CONTRACT_MISMATCH toute soumission ET
    # tout precommit dont le profile_id != qwen3-4b-base-dapo-v4 (upstream
    # batcher.py:1909-1918 + 2236) — bloquant trouvé par l'audit 2026-08-17.
    assert c.GENERATION_PROFILE_ID == "qwen3-4b-base-dapo-v4"
    # env-aware : un env dict explicite doit suivre SON protocol_version
    assert c.generation_profile_id({}) == "qwen35-4b-auction-v3"
    assert (
        c.generation_profile_id({"RELIQUARY_PROTOCOL_VERSION": "4"})
        == "qwen3-4b-base-dapo-v4"
    )
    # audit item 7 : modèle de départ + révision épinglée (profil upstream)
    assert c.DEFAULT_BASE_MODEL == "Qwen/Qwen3-4B-Base"
    assert (
        c.DEFAULT_BASE_MODEL_REVISION
        == "906bfd4b4dc7f14ee4320094d8b41684abff8539"
    )


def test_v4_vllm_max_model_len(monkeypatch):
    # audit item 13 : contexte vLLM = cap + 1024 sous v4, 16384 en v3
    import importlib

    reload_constants(monkeypatch, 4)
    import reliquary.cli.main as m

    importlib.reload(m)
    assert m.vllm_max_model_len() == 8192 + 1024
    reload_constants(monkeypatch)
    importlib.reload(m)
    assert m.vllm_max_model_len() == 16384


def test_v4_cap_env_override_still_wins(monkeypatch):
    monkeypatch.setenv("RELIQUARY_MAX_NEW_TOKENS_PROTOCOL_CAP", "4096")
    c = reload_constants(monkeypatch, 4)
    assert c.MAX_NEW_TOKENS_PROTOCOL_CAP == 4096
    monkeypatch.delenv("RELIQUARY_MAX_NEW_TOKENS_PROTOCOL_CAP")


def test_v4_import_does_not_crash_on_bft_assert(monkeypatch):
    # v4 : cap 8192 < BFT 15616+512 — l'assert de cohérence BFT du module ne
    # doit s'appliquer que si BFT_ENABLED, sinon l'import même explose.
    c = reload_constants(monkeypatch, 4)
    assert c.MAX_NEW_TOKENS_PROTOCOL_CAP < (
        c.BFT_THINKING_BUDGET + c.BFT_ANSWER_BUDGET
    )
