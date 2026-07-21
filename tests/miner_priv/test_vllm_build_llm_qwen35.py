"""Task 1: _build_llm must load the qwen3_5 hybrid-GDN model correctly under
forced-seed (trust_remote_code, text-only, triton GDN backend, eager, and NO
ngram speculative_config — spec-decoded tokens would fail seed-consistency).

The vLLM import is stubbed so this runs on the dev box without vllm/GPU.
"""
import sys
import types
from unittest import mock


def _stub_vllm(monkeypatch, captured):
    def _LLM(**kwargs):
        captured.update(kwargs)
        return mock.MagicMock()
    mod = types.ModuleType("vllm")
    mod.LLM = _LLM
    monkeypatch.setitem(sys.modules, "vllm", mod)


def test_build_llm_forced_seed_kwargs(monkeypatch):
    captured = {}
    _stub_vllm(monkeypatch, captured)
    from reliquary.miner import vllm_backend

    vllm_backend._build_llm(
        model_path="/m", tokenizer_path="/t", gpu_id=0,
        gpu_memory_utilization=0.55, max_model_len=16384, dtype="bfloat16",
        forced_seed=True,
    )
    assert captured["trust_remote_code"] is True
    assert captured["limit_mm_per_prompt"] == {"image": 0, "video": 0}
    assert captured["additional_config"] == {"gdn_prefill_backend": "triton"}
    assert captured["enforce_eager"] is True
    assert "speculative_config" not in captured  # no spec-decode under forced-seed


def test_build_llm_legacy_path_unchanged(monkeypatch):
    # forced_seed=False (default) keeps the legacy ngram speculative path and
    # does NOT inject the qwen3_5-only kwargs, so non-enforcement behaviour is
    # byte-identical to today.
    captured = {}
    _stub_vllm(monkeypatch, captured)
    monkeypatch.delenv("RELIQUARY_DISABLE_SPECULATIVE", raising=False)
    from reliquary.miner import vllm_backend

    vllm_backend._build_llm(
        model_path="/m", gpu_id=0, gpu_memory_utilization=0.85,
        max_model_len=16384, dtype="bfloat16",
    )
    assert captured.get("speculative_config", {}).get("method") == "ngram"
    assert "additional_config" not in captured
    assert "trust_remote_code" not in captured
