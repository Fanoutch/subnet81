"""Cache torch.compile FIGÉ entre checkpoints (rechargement de checkpoint).

vLLM dérive le répertoire de cache torch.compile de son ``config_hash``, qui
inclut le CHEMIN du modèle. Chaque checkpoint étant un snapshot HF distinct,
le hash change à chaque avancée et vLLM recompile intégralement, alors que le
graphe est identique (mesuré : ``computation_graph.py`` byte-identique entre
deux caches, seul ``config_hash`` diffère).

Coût mesuré sur la box le 25/08, deux démarrages du même jour :
  cache MANQUÉ : torch.compile 22,41 s / init engine 36,90 s
  cache TOUCHÉ : torch.compile  4,23 s / init engine 11,32 s

Le repli est la variable elle-même : non définie => None => rien n'est passé
à vLLM => comportement d'aujourd'hui.
"""
from __future__ import annotations

import json
import os

from reliquary.miner.vllm_backend import (
    vllm_compile_cache_dir,
    vllm_compile_cache_key,
)


CFG_A = b'{"architectures": ["Qwen3ForCausalLM"], "hidden_size": 2560}'
CFG_B = b'{"architectures": ["Qwen3ForCausalLM"], "hidden_size": 4096}'

KW = {
    "max_model_len": 9216,
    "max_num_seqs": 256,
    "dtype": "bfloat16",
    "enforce_eager": False,
    "kv_cache_dtype": "auto",
    "trust_remote_code": True,
}


def _snapshot(tmp_path, name: str, cfg: bytes) -> str:
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_bytes(cfg)
    return str(d)


def test_unset_env_means_no_override(tmp_path):
    """Défaut = None : _build_llm ne passe rien, vLLM garde son propre hash."""
    p = _snapshot(tmp_path, "snap", CFG_A)
    assert vllm_compile_cache_dir(p, KW, env={}) is None


def test_two_checkpoints_share_one_cache_dir(tmp_path):
    """LE POINT DU PATCH : deux snapshots distincts, même architecture =>
    MÊME répertoire, donc le second démarrage touche le cache."""
    a = _snapshot(tmp_path, "snapshots_34a7311d", CFG_A)
    b = _snapshot(tmp_path, "snapshots_38e46181", CFG_A)
    env = {"RELIQUARY_VLLM_COMPILE_CACHE_DIR": "/workspace/vllm_compile_cache"}
    assert vllm_compile_cache_dir(a, KW, env=env) == vllm_compile_cache_dir(
        b, KW, env=env
    )


def test_different_architecture_gets_a_different_dir(tmp_path):
    """SÛRETÉ : si le validateur publie une autre architecture, le
    config.json change => nouveau répertoire => aucune réutilisation d'un
    graphe compilé pour un autre modèle."""
    a = _snapshot(tmp_path, "snap_a", CFG_A)
    b = _snapshot(tmp_path, "snap_b", CFG_B)
    env = {"RELIQUARY_VLLM_COMPILE_CACHE_DIR": "/workspace/vllm_compile_cache"}
    assert vllm_compile_cache_dir(a, KW, env=env) != vllm_compile_cache_dir(
        b, KW, env=env
    )


def test_engine_shape_kwargs_change_the_dir(tmp_path):
    """Les kwargs qui changent les formes/kernels compilés doivent séparer
    les caches : sinon un changement de max_model_len ou d'enforce_eager
    réutiliserait des artefacts invalides."""
    p = _snapshot(tmp_path, "snap", CFG_A)
    base = vllm_compile_cache_key(p, KW)
    for key, value in (
        ("max_model_len", 16384),
        ("max_num_seqs", 512),
        ("enforce_eager", True),
        ("dtype", "float16"),
    ):
        variante = dict(KW, **{key: value})
        assert vllm_compile_cache_key(p, variante) != base, key


def test_dir_carries_the_vllm_version(tmp_path):
    """Une montée de version de vLLM ne doit jamais réutiliser les artefacts
    de la précédente."""
    p = _snapshot(tmp_path, "snap", CFG_A)
    env = {"RELIQUARY_VLLM_COMPILE_CACHE_DIR": "/root/cache"}
    out = vllm_compile_cache_dir(p, KW, env=env)
    assert out.startswith("/root/cache" + os.sep + "vllm-")


def test_unreadable_config_falls_back_to_the_path(tmp_path):
    """Pas de config.json lisible => on repart du chemin (comportement
    actuel : une clé par checkpoint), jamais d'exception."""
    a = str(tmp_path / "absent_a")
    b = str(tmp_path / "absent_b")
    assert vllm_compile_cache_key(a, KW) != vllm_compile_cache_key(b, KW)


# --- câblage réel dans _build_llm -------------------------------------------

def test_build_llm_passes_cache_dir_when_armed(tmp_path, monkeypatch):
    """Le helper ne sert à rien s'il n'est pas passé à vLLM : on vérifie les
    kwargs réellement remis à LLM()."""
    import sys
    import types

    captured = {}

    fake_vllm = types.ModuleType("vllm")

    class _FakeLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_vllm.LLM = _FakeLLM
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    p = _snapshot(tmp_path, "snap", CFG_A)
    monkeypatch.setenv(
        "RELIQUARY_VLLM_COMPILE_CACHE_DIR", str(tmp_path / "cache")
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    from reliquary.miner import vllm_backend

    vllm_backend._build_llm(
        model_path=p, gpu_id=0, gpu_memory_utilization=0.76,
        max_model_len=9216, dtype="bfloat16", forced_seed=False,
    )

    assert "compilation_config" in captured
    assert set(captured["compilation_config"]) == {"cache_dir"}
    assert captured["compilation_config"]["cache_dir"].startswith(
        str(tmp_path / "cache")
    )


def test_build_llm_passes_nothing_when_unarmed(tmp_path, monkeypatch):
    """Repli : variable absente => aucune clé compilation_config => vLLM
    se comporte exactement comme aujourd'hui."""
    import sys
    import types

    captured = {}

    fake_vllm = types.ModuleType("vllm")

    class _FakeLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_vllm.LLM = _FakeLLM
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    p = _snapshot(tmp_path, "snap", CFG_A)
    monkeypatch.delenv("RELIQUARY_VLLM_COMPILE_CACHE_DIR", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    from reliquary.miner import vllm_backend

    vllm_backend._build_llm(
        model_path=p, gpu_id=0, gpu_memory_utilization=0.76,
        max_model_len=9216, dtype="bfloat16", forced_seed=False,
    )

    assert "compilation_config" not in captured
