"""Unit tests for VLLMBackend with vllm.LLM mocked.

These tests do NOT require a GPU — they patch the vllm LLM class so the
backend's lazy-init path can be exercised on CPU.
"""
from unittest.mock import patch, MagicMock
import pytest

from reliquary.miner.vllm_backend import VLLMBackend


def test_init_does_not_load_engine():
    """Construction is lazy — no LLM instance built until first generate()."""
    backend = VLLMBackend(model_path="/fake/path", gpu_id=0)
    assert backend._llm is None
    assert backend._model_path == "/fake/path"
    assert backend._gpu_id == 0


def test_default_config_values():
    backend = VLLMBackend(model_path="/fake/path", gpu_id=0)
    assert backend._gpu_memory_utilization == 0.85
    assert backend._max_model_len == 4096
    assert backend._dtype == "bfloat16"


@patch("reliquary.miner.vllm_backend._build_sampling_params")
@patch("reliquary.miner.vllm_backend._build_llm")
def test_generate_returns_n_token_id_lists(mock_build, mock_build_sp):
    """generate(n=8) returns 8 list-of-int completions."""
    fake_llm = MagicMock()
    fake_outputs_list = [
        MagicMock(token_ids=[10, 20, 30]),
        MagicMock(token_ids=[40, 50]),
        MagicMock(token_ids=[60]),
        MagicMock(token_ids=[70, 80, 90, 100]),
        MagicMock(token_ids=[110]),
        MagicMock(token_ids=[120, 130]),
        MagicMock(token_ids=[140]),
        MagicMock(token_ids=[150, 160, 170]),
    ]
    fake_request_output = MagicMock(outputs=fake_outputs_list)
    fake_llm.generate.return_value = [fake_request_output]
    mock_build.return_value = fake_llm
    mock_build_sp.return_value = MagicMock(name="sampling_params_sentinel")

    backend = VLLMBackend(model_path="/fake", gpu_id=0)
    result = backend.generate(
        prompt_token_ids=[1, 2, 3],
        n=8,
        temperature=0.9,
        top_p=1.0,
        top_k=-1,
        max_tokens=1500,
    )

    assert len(result) == 8
    assert result[0] == [10, 20, 30]
    assert result[3] == [70, 80, 90, 100]


@patch("reliquary.miner.vllm_backend._build_sampling_params")
@patch("reliquary.miner.vllm_backend._build_llm")
def test_generate_lazy_inits_engine_once(mock_build, mock_build_sp):
    """Multiple generate() calls reuse the same LLM instance."""
    fake_llm = MagicMock()
    fake_request_output = MagicMock(outputs=[MagicMock(token_ids=[1]) for _ in range(8)])
    fake_llm.generate.return_value = [fake_request_output]
    mock_build.return_value = fake_llm
    mock_build_sp.return_value = MagicMock(name="sampling_params_sentinel")

    backend = VLLMBackend(model_path="/fake", gpu_id=0)
    backend.generate(prompt_token_ids=[1], n=8)
    backend.generate(prompt_token_ids=[2], n=8)

    assert mock_build.call_count == 1


@patch("reliquary.miner.vllm_backend._build_sampling_params")
@patch("reliquary.miner.vllm_backend._build_llm")
def test_generate_passes_sampling_params(mock_build_llm, mock_build_sp):
    """The kwargs passed to _build_sampling_params reflect what the caller asked for.

    `_build_sampling_params` is the seam where the actual vLLM `SamplingParams`
    is constructed; mocking it lets us assert what would be passed without
    requiring vllm to be installed.
    """
    fake_llm = MagicMock()
    fake_request_output = MagicMock(outputs=[MagicMock(token_ids=[1]) for _ in range(4)])
    fake_llm.generate.return_value = [fake_request_output]
    mock_build_llm.return_value = fake_llm
    mock_build_sp.return_value = MagicMock(name="sampling_params_sentinel")

    backend = VLLMBackend(model_path="/fake", gpu_id=0)
    backend.generate(
        prompt_token_ids=[5, 6, 7],
        n=4,
        temperature=0.7,
        top_p=0.95,
        top_k=50,
        max_tokens=512,
    )

    mock_build_sp.assert_called_once_with(
        n=4, temperature=0.7, top_p=0.95, top_k=50, max_tokens=512,
    )
    fake_llm.generate.assert_called_once()
    call_kwargs = fake_llm.generate.call_args.kwargs
    assert call_kwargs["sampling_params"] is mock_build_sp.return_value


@patch("reliquary.miner.vllm_backend._build_sampling_params")
@patch("reliquary.miner.vllm_backend._build_llm")
def test_reload_swaps_model_path_and_clears_engine(mock_build, mock_build_sp):
    """After reload, the next generate() rebuilds with the new path."""
    fake_llm_a = MagicMock()
    fake_llm_b = MagicMock()
    fake_request_output = MagicMock(outputs=[MagicMock(token_ids=[1]) for _ in range(8)])
    fake_llm_a.generate.return_value = [fake_request_output]
    fake_llm_b.generate.return_value = [fake_request_output]
    mock_build.side_effect = [fake_llm_a, fake_llm_b]
    mock_build_sp.return_value = MagicMock()

    backend = VLLMBackend(model_path="/old", gpu_id=0)
    backend.generate(prompt_token_ids=[1], n=8)   # builds fake_llm_a

    backend.reload(new_model_path="/new")
    assert backend._model_path == "/new"
    assert backend._llm is None

    backend.generate(prompt_token_ids=[1], n=8)   # rebuilds → fake_llm_b
    assert mock_build.call_count == 2
    assert mock_build.call_args.kwargs["model_path"] == "/new"


@patch("reliquary.miner.vllm_backend._build_llm")
def test_reload_before_first_generate_is_safe(mock_build):
    """reload() before any generate() just updates the path."""
    backend = VLLMBackend(model_path="/old", gpu_id=0)
    backend.reload(new_model_path="/new")
    assert backend._model_path == "/new"
    assert backend._llm is None
    # mock_build NOT called yet
    assert mock_build.call_count == 0
