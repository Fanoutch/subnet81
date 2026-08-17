"""Task 4 du port v4 : manifest OMI restreint aux shards ``train-*``.

⚠️ Changement CONSENSUS : ``len(env)`` est le prompt-range partagé avec le
validateur ; un manifest divergent = 100 % PROMPT_MISMATCH silencieux. Le
filtre doit s'appliquer aux DEUX chemins de listing (cache local + fs
distant), comme upstream 8c38992.
"""
import pytest

from reliquary.environment.virtual_parquet import VirtualParquetDataset
from tests.v4helpers import reload_constants


@pytest.fixture(autouse=True)
def _restore_constants(monkeypatch):
    yield
    reload_constants(monkeypatch)


def _vp(prefix):
    return VirtualParquetDataset(
        "owner/repo", "rev", columns=["problem"], filename_prefix=prefix,
    )


def test_shard_included_prefix_filters_basename():
    vp = _vp("train-")
    assert vp._shard_included("data/train-00001-of-00500.parquet")
    assert not vp._shard_included("data/train_1M-00001.parquet")
    assert not vp._shard_included("data/train_5M-00001.parquet")
    # le préfixe s'applique au BASENAME, pas au chemin complet
    assert vp._shard_included("train-huggingface/data/train-00001.parquet")


def test_shard_included_none_accepts_everything():
    vp = _vp(None)
    assert vp._shard_included("data/train_1M-00001.parquet")


def test_v4_load_dataset_refuses_local_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr("reliquary.constants.OMI_TRAIN_SHARDS_ONLY", True)
    (tmp_path / "dataset_info.json").write_text("{}")
    from reliquary.environment.openmathinstruct import _load_dataset

    with pytest.raises(RuntimeError, match="train- shard filter"):
        _load_dataset(str(tmp_path), "rev")


def test_v4_load_dataset_passes_prefix(monkeypatch):
    monkeypatch.setattr("reliquary.constants.OMI_TRAIN_SHARDS_ONLY", True)
    captured = {}

    class _FakeVP:
        def __init__(self, repo, revision, **kwargs):
            captured.update(kwargs)

    from reliquary.environment import virtual_parquet as vp_mod

    monkeypatch.setattr(vp_mod, "VirtualParquetDataset", _FakeVP)
    from reliquary.environment.openmathinstruct import _load_dataset

    _load_dataset("owner/repo", "rev")
    assert captured["filename_prefix"] == "train-"


def test_v3_load_dataset_no_prefix(monkeypatch):
    monkeypatch.setattr("reliquary.constants.OMI_TRAIN_SHARDS_ONLY", False)
    captured = {}

    class _FakeVP:
        def __init__(self, repo, revision, **kwargs):
            captured.update(kwargs)

    from reliquary.environment import virtual_parquet as vp_mod

    monkeypatch.setattr(vp_mod, "VirtualParquetDataset", _FakeVP)
    from reliquary.environment.openmathinstruct import _load_dataset

    _load_dataset("owner/repo", "rev")
    assert captured["filename_prefix"] is None
