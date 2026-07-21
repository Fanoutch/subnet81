import pydantic
import pytest

from reliquary.protocol.submission import RolloutMetadata, CommitModel


def test_rollout_metadata_bft_fields():
    m = RolloutMetadata(prompt_length=1, completion_length=2, success=True,
                        total_reward=0.0, advantage=0.0, token_logprobs=[0.0],
                        forced=True, force_span=[10, 13])
    assert m.forced is True and m.force_span == [10, 13] and m.truncated is False


def test_commit_requires_v7():
    with pytest.raises(pydantic.ValidationError):
        CommitModel.model_validate({
            "tokens": [0] * 40, "commitments": [], "proof_version": "v6",
            "model": {"name": "x", "layer_index": -1}, "signature": "ab",
            "beacon": {"randomness": "r"},
            "rollout": {"prompt_length": 1, "completion_length": 1, "success": True,
                        "total_reward": 0.0, "advantage": 0.0, "token_logprobs": [0.0]}})
