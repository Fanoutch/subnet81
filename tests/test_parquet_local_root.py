"""Lecture du parquet depuis un miroir LOCAL (24/08).

Mesuré ce jour-là : le classement de tranche coûte 2,80 s en tête de chaque
fenêtre, dont **0,95 s de lecture parquet sur le réseau HuggingFace**. La
tranche `[lo,hi)` est tirée uniformément dans 2 481 806 indices à chaque
fenêtre, donc les 5-6 row-groups sont TOUJOURS froids : le LRU de 64
row-groups en RAM ne sert quasiment jamais, et il n'existe aucun cache disque.
Pire, 5,4 % des fenêtres subissent un `handshake timed out` qui fait exploser
le budget de classement (12 s) et coûte +9,5 s sur le premier groupe.

Le fichier fait 1,39 Go et il reste 85 Go sur la box : le miroir local est
gratuit.

⚠️ PROPRIÉTÉ DE SÛRETÉ : `len(dataset)` est le **consensus prompt-range** —
le validateur en dérive la tranche de chaque fenêtre. Un miroir qui ne
contiendrait pas exactement les mêmes lignes donnerait une tranche différente
et 100 % de `prompt_out_of_range`. Le chemin local doit donc rendre
EXACTEMENT le même `len()` et les mêmes lignes que le chemin distant.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from reliquary.environment.virtual_parquet import VirtualParquetDataset


def _write_parquet(path: Path, rows: list[dict], row_group_size: int = 2) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    table = pa.Table.from_pylist(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, row_group_size=row_group_size)


@pytest.fixture
def miroir(tmp_path: Path) -> Path:
    """Miroir local : <racine>/data/train-00000-of-00001.parquet"""
    rows = [{"input": f"probleme {i}", "structured_cases": f"cas {i}"}
            for i in range(6)]
    _write_parquet(tmp_path / "data" / "train-00000-of-00001.parquet", rows)
    return tmp_path


def test_lit_depuis_la_racine_locale_sans_reseau(miroir: Path):
    """Avec ``local_root``, aucun HfFileSystem n'est instancié."""
    ds = VirtualParquetDataset(
        "R0mAI/opencodeinstruct-curated", "deadbeef",
        columns=["input", "structured_cases"],
        local_root=str(miroir),
    )

    assert len(ds) == 6
    assert ds[0]["input"] == "probleme 0"
    assert ds[5]["input"] == "probleme 5"


def test_len_et_lignes_identiques_a_une_lecture_directe(miroir: Path):
    """`len()` est le consensus prompt-range : le miroir doit rendre
    exactement le même nombre de lignes, dans le même ORDRE, qu'une lecture
    directe du parquet. Un décalage d'une seule ligne déplace la tranche de
    chaque fenêtre et donne 100 % de `prompt_out_of_range`."""
    pq = pytest.importorskip("pyarrow.parquet")

    attendu = pq.read_table(
        miroir / "data" / "train-00000-of-00001.parquet"
    ).to_pylist()

    ds = VirtualParquetDataset(
        "repo/x", "rev", columns=["input", "structured_cases"],
        local_root=str(miroir),
    )

    assert len(ds) == len(attendu)
    assert [ds[i]["input"] for i in range(len(ds))] == \
           [r["input"] for r in attendu]


def test_le_prefixe_de_shard_est_respecte_en_local(tmp_path: Path):
    """`filename_prefix` filtre le manifest — il vaut « train- » en v4/v5 et
    conditionne `len()`. Le chemin local doit le respecter à l'identique."""
    _write_parquet(tmp_path / "data" / "train-00000.parquet",
                   [{"input": "a"}, {"input": "b"}])
    _write_parquet(tmp_path / "data" / "valid-00000.parquet",
                   [{"input": "c"}])

    ds = VirtualParquetDataset(
        "repo/x", "rev", columns=["input"],
        local_root=str(tmp_path), filename_prefix="train-",
    )

    assert len(ds) == 2  # le shard « valid- » est exclu


def test_racine_absente_echoue_bruyamment(tmp_path: Path):
    """Un miroir manquant doit lever, PAS retomber silencieusement sur le
    réseau : un fallback muet donnerait un `len()` correct mais réintroduirait
    la latence qu'on cherche à supprimer, sans qu'on le sache."""
    ds = VirtualParquetDataset(
        "repo/x", "rev", local_root=str(tmp_path / "nexiste_pas"),
    )

    with pytest.raises(Exception):
        len(ds)


# ---- câblage dans l'environnement opencode ---------------------------------

def test_env_var_active_le_miroir(monkeypatch, miroir: Path):
    """`RELIQUARY_PARQUET_LOCAL_ROOT` doit suffire à basculer sur le disque."""
    import reliquary.environment.opencodeinstruct as oci

    monkeypatch.setenv("RELIQUARY_PARQUET_LOCAL_ROOT", str(miroir))
    ds = oci._load_dataset("R0mAI/opencodeinstruct-curated", "deadbeef")

    assert getattr(ds, "_local_root", None) == str(miroir)
    assert len(ds) == 6


def test_sans_env_var_le_comportement_est_inchange(monkeypatch):
    """Absence de la variable = chemin distant historique, byte-identique."""
    import reliquary.environment.opencodeinstruct as oci

    monkeypatch.delenv("RELIQUARY_PARQUET_LOCAL_ROOT", raising=False)
    ds = oci._load_dataset("R0mAI/opencodeinstruct-curated", "deadbeef")

    assert getattr(ds, "_local_root", "absent") is None


def test_len_inattendu_leve_au_lieu_de_desynchroniser(monkeypatch, miroir: Path):
    """GARDE DE SÛRETÉ : `len()` est le consensus prompt-range.

    Un miroir incomplet donnerait un `len()` différent de celui du validateur,
    donc une tranche décalée et 100 % de `prompt_out_of_range` — un échec
    TOTAL et silencieux. On exige donc que la longueur attendue soit vérifiée
    et que toute divergence lève bruyamment.
    """
    import reliquary.environment.opencodeinstruct as oci

    monkeypatch.setenv("RELIQUARY_PARQUET_LOCAL_ROOT", str(miroir))
    monkeypatch.setenv("RELIQUARY_PARQUET_EXPECTED_LEN", "999999")

    ds = oci._load_dataset("R0mAI/opencodeinstruct-curated", "deadbeef")
    with pytest.raises(RuntimeError, match="prompt-range"):
        len(ds)


def test_len_attendu_correct_passe(monkeypatch, miroir: Path):
    import reliquary.environment.opencodeinstruct as oci

    monkeypatch.setenv("RELIQUARY_PARQUET_LOCAL_ROOT", str(miroir))
    monkeypatch.setenv("RELIQUARY_PARQUET_EXPECTED_LEN", "6")

    assert len(oci._load_dataset("R0mAI/opencodeinstruct-curated", "x")) == 6
