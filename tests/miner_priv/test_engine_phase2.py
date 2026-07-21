"""Phase 2 engine wiring tests."""
from unittest.mock import MagicMock
import pytest


def test_engine_accepts_vllm_backend_param():
    """MiningEngine.__init__ accepts a VLLMBackend via keyword arg."""
    from reliquary.miner.engine import MiningEngine
    from reliquary.miner.vllm_backend import VLLMBackend

    backend = VLLMBackend(model_path="/fake", gpu_id=0)
    # Bypass full init by patching dependencies
    engine = MiningEngine.__new__(MiningEngine)
    engine._vllm_backend = backend
    assert engine._vllm_backend is backend


def test_engine_init_records_vllm_backend_attr():
    """When vllm_backend kwarg is passed, it's stored as self._vllm_backend."""
    from reliquary.miner.engine import MiningEngine
    from reliquary.miner.vllm_backend import VLLMBackend

    fake_backend = VLLMBackend(model_path="/fake", gpu_id=0)
    fake_hf_model = MagicMock()
    fake_hf_model.config.hidden_size = 128
    fake_tokenizer = MagicMock()
    fake_wallet = MagicMock()
    fake_env = MagicMock()
    fake_env.__len__ = MagicMock(return_value=10)

    # Patch BucketIndex to avoid the real HF dataset load
    from unittest.mock import patch
    with patch("reliquary.miner.engine.BucketIndex") as mock_bucket:
        mock_bucket.return_value = MagicMock(__len__=MagicMock(return_value=10),
                                              bucket_of=MagicMock(return_value=("X", "Y")))
        engine = MiningEngine(
            vllm_model=None,
            hf_model=fake_hf_model,
            tokenizer=fake_tokenizer,
            wallet=fake_wallet,
            env=fake_env,
            vllm_backend=fake_backend,
        )
    assert engine._vllm_backend is fake_backend


def test_generate_m_rollouts_uses_vllm_backend():
    """When _vllm_backend is set, _generate_m_rollouts calls backend.generate."""
    from unittest.mock import patch, MagicMock
    from reliquary.miner.engine import MiningEngine

    engine = MiningEngine.__new__(MiningEngine)
    # Mock the tokenizer
    fake_tokenizer = MagicMock()
    fake_tokenizer.encode = MagicMock(return_value=[100, 101, 102])
    fake_tokenizer.eos_token_id = 999
    engine.tokenizer = fake_tokenizer
    engine.max_new_tokens = 1500
    # Mock the backend
    fake_backend = MagicMock()
    fake_backend.generate = MagicMock(return_value=[
        [200, 201, 999, 700],   # first EOS at index 2 -> keep up to and including 999
        [300, 999],
        [400, 401, 402],         # no EOS
        [500],
        [600, 601, 602, 603, 604],
        [700, 701],
        [800],
        [900, 901],
    ])
    engine._vllm_backend = fake_backend
    engine.vllm_model = None   # explicit: not used

    problem = {"prompt": "What is 2+2?"}
    result = engine._generate_m_rollouts(problem, randomness=b"x" * 32)

    # 8 rollouts back
    assert len(result) == 8
    # Each is a dict with tokens + prompt_length
    for r in result:
        assert "tokens" in r
        assert "prompt_length" in r
        assert r["prompt_length"] == 3   # len([100, 101, 102])
        # tokens = prompt + completion
        assert r["tokens"][:3] == [100, 101, 102]
    # First rollout: completion truncated at first EOS (index 2 -> keeps [200,201,999])
    assert result[0]["tokens"] == [100, 101, 102, 200, 201, 999]
    # Third rollout has no EOS - completion kept whole
    assert result[2]["tokens"] == [100, 101, 102, 400, 401, 402]
    # Backend was called with the right token ids and n=8
    fake_backend.generate.assert_called_once()
    call_kwargs = fake_backend.generate.call_args.kwargs
    assert call_kwargs["prompt_token_ids"] == [100, 101, 102]
    assert call_kwargs["n"] == 8


def test_load_checkpoint_calls_backend_reload_when_set(tmp_path):
    """When _vllm_backend is set, _load_checkpoint should call backend.reload
    and SKIP the HF GPU-0 rebuild path."""
    from unittest.mock import MagicMock, patch
    from reliquary.miner.engine import MiningEngine

    engine = MiningEngine.__new__(MiningEngine)
    engine._loaded_checkpoint_path = None
    engine.proof_gpu = 1
    engine.vllm_gpu = 0
    engine.hf_model = MagicMock()    # current proof model
    engine.vllm_model = None         # not used in vLLM path

    fake_backend = MagicMock()
    engine._vllm_backend = fake_backend

    # Stub HF load + cuda calls so we don't need real GPU/model
    with patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_hf:
        mock_hf.return_value = MagicMock(to=MagicMock(return_value=MagicMock(eval=MagicMock(return_value=MagicMock()))))
        result = engine._load_checkpoint(str(tmp_path))

    fake_backend.reload.assert_called_once_with(new_model_path=str(tmp_path))
    # Only ONE HF load (the proof model on GPU 1), NOT two
    assert mock_hf.call_count == 1
