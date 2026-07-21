from reliquary.miner.engine import MiningEngine


def test_engine_entry_env_name_reads_key():
    fake = object.__new__(MiningEngine)        # pas d'__init__ (pas de GPU)
    fake.active_envs = ["openmathinstruct"]
    assert MiningEngine._entry_env_name(fake, {"env_name": "opencodeinstruct"}) \
        == "opencodeinstruct"


def test_engine_entry_env_name_defaults_to_first_active_env():
    fake = object.__new__(MiningEngine)
    fake.active_envs = ["openmathinstruct"]
    assert MiningEngine._entry_env_name(fake, {"prompt_idx": 7}) == "openmathinstruct"
