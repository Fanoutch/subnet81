"""_generate_m_rollouts must, for the math env, cap phase-1 at the thinking
budget and route completions through BFT (forced/force_span tagging); for a
non-math env it must NOT cap and NOT force-terminate."""
import types

import torch

from reliquary.constants import BFT_THINKING_BUDGET, M_ROLLOUTS
from reliquary.miner import engine as engine_mod
from reliquary.miner.engine import MiningEngine


class _Backend:
    """Fake vLLM backend: records max_tokens, returns M maxed (no-EOS) gens."""

    def __init__(self, gen_len):
        self.gen_len = gen_len
        self.max_tokens_seen = None

    def generate(self, *, prompt_token_ids, n, max_tokens, **kw):
        self.max_tokens_seen = max_tokens
        # every gen is `gen_len` long with no </think>(7) and no EOS(2) → forced
        return [[5] * self.gen_len for _ in range(n)]


class _HF:
    device = "cpu"

    def generate(self, rows, attention_mask=None, max_new_tokens=0, **kw):
        return torch.tensor([r.tolist() + [50, 2] for r in rows])


class _Tok:
    pad_token_id = 0

    def convert_tokens_to_ids(self, s):
        return 7  # </think>

    def encode(self, s, add_special_tokens=False):
        return [8]  # FORCE tail → force_ids = [7, 8]


def _engine(env_name):
    e = MiningEngine.__new__(MiningEngine)
    e._vllm_backend = _Backend(gen_len=4)
    e.vllm_model = None
    e.tokenizer = _Tok()
    e.max_new_tokens = 8192
    e._eos_ids = [2]
    e.hf_model = _HF()
    e.env = types.SimpleNamespace(name=env_name)
    e.wallet = types.SimpleNamespace(
        hotkey=types.SimpleNamespace(ss58_address="hk"))
    e._local_hash = "abc"
    return e


def test_math_caps_phase1_and_forces(monkeypatch):
    # Legacy (non-forced-seed) semantics: the vLLM backend path. The enforced
    # path (backend bypassed → HF) is covered by test_forced_seed_wiring_fixes.
    import reliquary.constants as constants
    monkeypatch.setattr(constants, "FORCED_SEED_ENFORCE", False)
    monkeypatch.setattr(engine_mod, "encode_prompt", lambda tok, p: [1, 1, 1])
    e = _engine("openmathinstruct")
    out = e._generate_m_rollouts({"prompt": "q"}, "")
    # phase-1 was capped at the thinking budget, not 8192
    assert e._vllm_backend.max_tokens_seen == BFT_THINKING_BUDGET
    assert len(out) == M_ROLLOUTS
    # every rollout was force-terminated (no </think>, no EOS)
    assert all(r["forced"] is True and "force_span" in r for r in out)


def test_code_env_no_cap_no_force(monkeypatch):
    import reliquary.constants as constants
    monkeypatch.setattr(constants, "FORCED_SEED_ENFORCE", False)
    monkeypatch.setattr(engine_mod, "encode_prompt", lambda tok, p: [1, 1, 1])
    e = _engine("opencodeinstruct")
    out = e._generate_m_rollouts({"prompt": "q"}, "", env=e.env)
    # NOT capped, and no BFT tagging on the code path
    assert e._vllm_backend.max_tokens_seen == 8192
    assert all("forced" not in r for r in out)
