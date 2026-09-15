"""vLLM doit charger les MÊMES poids que le validateur (15/09).

Mesuré sur la fenêtre 45969 : les snapshots de
``ReliquaryForge/qwen3-4b-base-dapo-v4`` portent à la fois ``model.safetensors``
(réécrit à chaque checkpoint) et des shards ``model-0000x-of-00003`` + index,
figés depuis le 14/09 08:31. transformers (validateur, preuve, réplique) charge
``model.safetensors`` ; vLLM filtre sur l'index et charge les shards périmés.
Nos tokens étaient tirés d'un autre modèle que celui qui les vérifie : 1er token
exact 2-3/16 (autres mineurs 84-97 %), seed_mismatch, garde dure, q10.

Correctif : vLLM reçoit une vue du snapshot (liens symboliques) sans l'index ni
les shards dès que ``model.safetensors`` est présent.
"""
from __future__ import annotations

import json
import os
import types

from reliquary.miner import vllm_backend
from reliquary.miner.vllm_weights_view import vllm_weights_path

SHARDS = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]


def _snapshot(tmp_path, *, single=True, index=True, name="6df4677bd9fa"):
    d = tmp_path / "snapshots" / name
    d.mkdir(parents=True)
    (d / "config.json").write_text('{"architectures": ["Qwen3ForCausalLM"]}')
    (d / "generation_config.json").write_text("{}")
    (d / "tokenizer.json").write_text("{}")
    if single:
        (d / "model.safetensors").write_bytes(b"NEW")
    if index:
        for s in SHARDS:
            (d / s).write_bytes(b"STALE")
        (d / "model.safetensors.index.json").write_text(json.dumps(
            {"metadata": {}, "weight_map": {"a": SHARDS[0], "b": SHARDS[1]}}))
    return str(d)


def test_snapshot_mixte_vue_sans_index_ni_shards(tmp_path):
    snap = _snapshot(tmp_path)
    view = vllm_weights_path(snap, view_root=str(tmp_path / "views"), env={})
    assert view != snap
    names = sorted(os.listdir(view))
    assert names == ["config.json", "generation_config.json", "model.safetensors",
                     "tokenizer.json"]
    assert open(os.path.join(view, "model.safetensors"), "rb").read() == b"NEW"
    assert os.path.realpath(os.path.join(view, "config.json")) == \
        os.path.realpath(os.path.join(snap, "config.json"))


def test_snapshot_shards_seuls_inchange(tmp_path):
    snap = _snapshot(tmp_path, single=False)
    assert vllm_weights_path(snap, view_root=str(tmp_path / "views"), env={}) == snap


def test_snapshot_fichier_unique_seul_inchange(tmp_path):
    snap = _snapshot(tmp_path, index=False)
    assert vllm_weights_path(snap, view_root=str(tmp_path / "views"), env={}) == snap


def test_kill_switch(tmp_path):
    snap = _snapshot(tmp_path)
    env = {"RELIQUARY_VLLM_WEIGHTS_VIEW": "0"}
    assert vllm_weights_path(snap, view_root=str(tmp_path / "views"), env=env) == snap


def test_vue_idempotente_et_distincte_par_revision(tmp_path):
    root = str(tmp_path / "views")
    a = _snapshot(tmp_path, name="rev_a")
    b = _snapshot(tmp_path, name="rev_b")
    va1 = vllm_weights_path(a, view_root=root, env={})
    va2 = vllm_weights_path(a, view_root=root, env={})
    vb = vllm_weights_path(b, view_root=root, env={})
    assert va1 == va2 and va1 != vb
    assert os.path.realpath(os.path.join(vb, "model.safetensors")) == \
        os.path.realpath(os.path.join(b, "model.safetensors"))


def test_vue_reconstruite_si_snapshot_change(tmp_path):
    """Un fichier ajouté au snapshot après coup apparaît dans la vue."""
    root = str(tmp_path / "views")
    snap = _snapshot(tmp_path)
    vllm_weights_path(snap, view_root=root, env={})
    open(os.path.join(snap, "reliquary_protocol_profile.json"), "w").write("{}")
    view = vllm_weights_path(snap, view_root=root, env={})
    assert "reliquary_protocol_profile.json" in os.listdir(view)
    assert "model.safetensors.index.json" not in os.listdir(view)


def test_vues_des_snapshots_purges_supprimees(tmp_path):
    import shutil
    root = str(tmp_path / "views")
    old = _snapshot(tmp_path, name="rev_old")
    v_old = vllm_weights_path(old, view_root=root, env={})
    shutil.rmtree(old)                       # purge HF de l'ancienne révision
    new = _snapshot(tmp_path, name="rev_new")
    v_new = vllm_weights_path(new, view_root=root, env={})
    assert os.path.isdir(v_new) and not os.path.exists(v_old)


def test_chemin_non_repertoire_inchange(tmp_path):
    assert vllm_weights_path("Qwen/Qwen3-4B-Base", view_root=str(tmp_path), env={}) == \
        "Qwen/Qwen3-4B-Base"


def test_echec_de_construction_retombe_sur_le_snapshot(tmp_path):
    snap = _snapshot(tmp_path)
    blocker = tmp_path / "views"
    blocker.write_text("pas un répertoire")
    assert vllm_weights_path(snap, view_root=str(blocker), env={}) == snap


# ------------------------------------------------------------- câblage moteur
def test_build_llm_charge_la_vue(tmp_path, monkeypatch):
    import sys
    snap = _snapshot(tmp_path)
    seen = {}

    class _LLM:
        def __init__(self, **kw):
            seen.update(kw)

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=_LLM))
    monkeypatch.setenv("RELIQUARY_VLLM_WEIGHTS_VIEW_ROOT", str(tmp_path / "views"))
    monkeypatch.delenv("RELIQUARY_VLLM_WEIGHTS_VIEW", raising=False)
    monkeypatch.delenv("RELIQUARY_VLLM_COMPILE_CACHE_DIR", raising=False)
    vllm_backend._build_llm(model_path=snap, gpu_id=0, gpu_memory_utilization=0.5,
                            max_model_len=1024, dtype="bfloat16",
                            tokenizer_path="Qwen/Qwen3-4B-Base")
    assert seen["model"] != snap
    assert "model.safetensors.index.json" not in os.listdir(seen["model"])
    assert seen["tokenizer"] == "Qwen/Qwen3-4B-Base"


def test_echange_a_chaud_charge_la_vue(tmp_path, monkeypatch):
    snap = _snapshot(tmp_path)
    monkeypatch.setenv("RELIQUARY_VLLM_WEIGHTS_VIEW_ROOT", str(tmp_path / "views"))
    monkeypatch.setattr("reliquary.miner.hot_swap_policy.hot_swap_mode", lambda: "on")
    calls = []

    class _LLM:
        def collective_rpc(self, name, kwargs):
            calls.append(kwargs["weights_path"])

        def reset_prefix_cache(self):
            pass

    import threading
    b = vllm_backend.VLLMBackend.__new__(vllm_backend.VLLMBackend)
    b._llm = _LLM()
    b._interrupt = threading.Event()
    b._abort_live_continuations = lambda: None
    assert b.reload_weights_inplace(snap) is True
    assert calls and calls[0] != snap
    assert "model.safetensors.index.json" not in os.listdir(calls[0])
    assert b._model_path == snap
