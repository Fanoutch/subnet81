"""Dump JSONL des groupes gradés → données d'entraînement du prédicteur.

Chaque groupe que le mineur grade est un échantillon étiqueté « gratuit »
(texte du prompt → 8 rewards → in_zone). ~400/heure en régime, produits
pendant que le mineur travaille — plus besoin d'un run de probe dédié et lent.

Format = celui qu'attend ``scripts/train_prompt_predictor.py`` :
lignes JSON avec au minimum ``prompt``, ``rewards``, ``in_zone``.

Contrainte de sûreté : écriture SEULE, jamais dans le chemin de décision.
Une panne de dump (disque plein, permission) ne doit JAMAIS faire tomber un
bake — d'où le test de tolérance aux erreurs.
"""
import json

import pytest

from reliquary.miner.engine import dump_group_sample


def test_writes_one_jsonl_row_with_trainer_schema(tmp_path, monkeypatch):
    out = tmp_path / "samples.jsonl"
    monkeypatch.setenv("RELIQUARY_SAMPLE_DUMP", str(out))

    dump_group_sample(
        prompt="def add(a, b):", prompt_idx=42,
        rewards=[1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        env_name="opencodeinstruct",
    )

    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert len(rows) == 1
    r = rows[0]
    # champs exigés par train_prompt_predictor.py
    assert r["prompt"] == "def add(a, b):"
    assert r["rewards"] == [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    assert isinstance(r["in_zone"], bool)
    # contexte utile pour l'analyse
    assert r["prompt_idx"] == 42
    assert r["env"] == "opencodeinstruct"
    assert r["sigma"] == pytest.approx(0.4841229, abs=1e-4)  # k=3/8


def test_in_zone_matches_the_validator_threshold(tmp_path, monkeypatch):
    """in_zone doit refléter le vrai seuil (σ >= 0.43), pas une approximation :
    c'est la CIBLE que le prédicteur apprend."""
    out = tmp_path / "s.jsonl"
    monkeypatch.setenv("RELIQUARY_SAMPLE_DUMP", str(out))

    dump_group_sample(prompt="p", prompt_idx=1,          # k=3/8 → sigma 0.484
                      rewards=[1.0] * 3 + [0.0] * 5, env_name="e")
    dump_group_sample(prompt="p", prompt_idx=2,          # unanime → sigma 0
                      rewards=[0.0] * 8, env_name="e")
    dump_group_sample(prompt="p", prompt_idx=3,          # k=1/8 → sigma 0.331
                      rewards=[1.0] + [0.0] * 7, env_name="e")

    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert [r["in_zone"] for r in rows] == [True, False, False]


def test_appends_across_calls(tmp_path, monkeypatch):
    out = tmp_path / "s.jsonl"
    monkeypatch.setenv("RELIQUARY_SAMPLE_DUMP", str(out))
    for i in range(3):
        dump_group_sample(prompt=f"p{i}", prompt_idx=i,
                          rewards=[0.0] * 8, env_name="e")
    assert len(out.read_text().splitlines()) == 3


def test_disabled_when_env_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("RELIQUARY_SAMPLE_DUMP", raising=False)
    # ne doit rien écrire ni lever
    dump_group_sample(prompt="p", prompt_idx=1, rewards=[0.0] * 8, env_name="e")
    assert list(tmp_path.iterdir()) == []


def test_never_raises_on_write_failure(tmp_path, monkeypatch):
    """Un dump cassé ne doit jamais tuer un bake."""
    monkeypatch.setenv("RELIQUARY_SAMPLE_DUMP", str(tmp_path / "nope" / "s.jsonl"))
    dump_group_sample(prompt="p", prompt_idx=1, rewards=[0.0] * 8, env_name="e")


def test_non_serialisable_rewards_do_not_raise(tmp_path, monkeypatch):
    out = tmp_path / "s.jsonl"
    monkeypatch.setenv("RELIQUARY_SAMPLE_DUMP", str(out))
    dump_group_sample(prompt="p", prompt_idx=1, rewards=None, env_name="e")


def test_records_truncation_count_so_training_can_exclude_artifacts(
    tmp_path, monkeypatch,
):
    """Un rollout coupé au plafond vaut 0 → moyenne basse → std*(1-mean) élevé.

    Le groupe ressemble alors à un k=2 payant sans en être un. Mesuré le
    2026-08-06 sur 8049 groupes : 71% des « k=2 » collectés étaient de tels
    artefacts, et 99,6% des groupes tronqués passaient la porte d'enchère.
    Sans cette étiquette, l'entraînement apprend à repérer les prompts LONGS.
    """
    out = tmp_path / "samples.jsonl"
    monkeypatch.setenv("RELIQUARY_SAMPLE_DUMP", str(out))

    dump_group_sample(
        prompt="longest palindromic substring", prompt_idx=7,
        rewards=[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        env_name="opencodeinstruct", n_truncated=6,
    )
    dump_group_sample(
        prompt="def add(a, b):", prompt_idx=8,
        rewards=[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        env_name="opencodeinstruct",
    )

    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert rows[0]["n_truncated"] == 6      # artefact: à exclure
    assert rows[1]["n_truncated"] == 0      # vrai k=2: à garder
