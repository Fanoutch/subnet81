"""Production wiring for max_num_seqs (bench: mns512 = 4934->5859 tok/s on 4B).

The bench proved the batched processor reopened concurrency scaling: capping
vLLM's scheduler at 512 running seqs beats the default. This wires
RELIQUARY_VLLM_MAX_NUM_SEQS into the miner's engine args (both sync and async
builders read _build_llm's kwargs helper). Unset -> no override (vLLM default),
malformed/non-positive -> ignored.
"""
from __future__ import annotations

from reliquary.miner.vllm_backend import vllm_max_num_seqs


def test_unset_means_no_override():
    assert vllm_max_num_seqs({}) is None


def test_reads_env_value():
    assert vllm_max_num_seqs({"RELIQUARY_VLLM_MAX_NUM_SEQS": "512"}) == 512


def test_malformed_or_nonpositive_is_ignored():
    assert vllm_max_num_seqs({"RELIQUARY_VLLM_MAX_NUM_SEQS": "abc"}) is None
    assert vllm_max_num_seqs({"RELIQUARY_VLLM_MAX_NUM_SEQS": "0"}) is None
