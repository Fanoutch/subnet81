"""The backend generate signatures must not default max_tokens below the BFT
thinking budget: a forced rollout's phase-1 must reach exactly
prompt_len + BFT_THINKING_BUDGET or the validator rejects it (TOKEN_TAMPERED).
A caller that omits max_tokens must therefore not silently cap phase-1 short."""
import inspect

from reliquary.constants import BFT_THINKING_BUDGET
from reliquary.miner import vllm_backend


def test_generate_default_max_tokens_not_below_thinking_budget():
    for meth in ("generate", "generate_multi"):
        default = inspect.signature(
            getattr(vllm_backend.VLLMBackend, meth)
        ).parameters["max_tokens"].default
        assert default >= BFT_THINKING_BUDGET, (
            f"VLLMBackend.{meth} defaults max_tokens={default} < "
            f"BFT_THINKING_BUDGET={BFT_THINKING_BUDGET}"
        )
