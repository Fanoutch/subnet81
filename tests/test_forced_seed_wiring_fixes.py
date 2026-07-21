"""Production-wiring fixes for FORCED_SEED_ENFORCE (audit b790e42):
1) the enforced HF generation path must fall back to self.hf_model when
   self.vllm_model is None (prod CLI wires a backend, not a raw HF twin);
2) the fire model must be fire-as-ready under enforcement (pool is flushed at
   every randomness flip → always empty at the first OPEN tick → the legacy
   single-burst would forfeit every window);
3) cross-window pool disk persistence must be off under enforcement (entries
   never survive a window)."""
import types

import torch

import reliquary.constants as constants
from reliquary.constants import M_ROLLOUTS
from reliquary.miner import engine as engine_mod
from reliquary.miner.engine import MiningEngine, pool_persist_enabled

_SENTINEL = 2 ** 63 - 1


class _HF:
    device = "cpu"

    def __init__(self):
        self.calls = 0

    def generate(self, rows, **kw):
        self.calls += 1
        # phase-1: extend each row with 4 non-EOS/non-close tokens;
        # phase-2: append a boxed answer then EOS.
        if self.calls == 1:
            return torch.tensor([r.tolist() + [9, 9, 9, 9] for r in rows])
        return torch.tensor([r.tolist() + [50, 2] for r in rows])


class _ExplodingBackend:
    def generate(self, **kw):
        raise AssertionError("vLLM backend must NOT be used under FORCED_SEED_ENFORCE")


class _Tok:
    pad_token_id = 0

    def convert_tokens_to_ids(self, s):
        return 7

    def encode(self, s, add_special_tokens=False):
        return [8]


def _engine():
    e = MiningEngine.__new__(MiningEngine)
    e.vllm_model = None            # prod: CLI wires a backend, no raw HF twin
    e.hf_model = _HF()
    e._vllm_backend = _ExplodingBackend()
    e.tokenizer = _Tok()
    e.max_new_tokens = 8192
    e._eos_ids = [2]
    e.wallet = types.SimpleNamespace(
        hotkey=types.SimpleNamespace(ss58_address="hk"))
    e._local_hash = "abc"
    e.env = types.SimpleNamespace(name="openmathinstruct")
    return e


def test_enforced_hf_path_falls_back_to_hf_model(monkeypatch):
    monkeypatch.setattr(constants, "FORCED_SEED_ENFORCE", True)
    monkeypatch.setattr(engine_mod, "encode_prompt", lambda tok, p: [1, 1, 1])
    e = _engine()
    out = e._generate_m_rollouts({"prompt": "q"}, "rand")
    assert len(out) == M_ROLLOUTS
    assert e.hf_model.calls >= 2  # phase-1 + phase-2 both on hf_model
    assert all(r["forced"] is True for r in out)


def test_fire_as_ready_under_enforcement(monkeypatch):
    e = MiningEngine.__new__(MiningEngine)
    e._prompt_range_from_window = _SENTINEL
    e._active_prompt_range = lambda *a, **k: None  # range dormant
    monkeypatch.setattr(constants, "FORCED_SEED_ENFORCE", True)
    assert e._fire_as_ready(1, "rand") is True
    monkeypatch.setattr(constants, "FORCED_SEED_ENFORCE", False)
    assert e._fire_as_ready(1, "rand") is False  # legacy stays legacy


def test_drop_pool_on_ckpt_forced_under_enforcement(monkeypatch):
    # checkpoint_hash is a u_at seed input: an entry baked under the OLD hash
    # and fired under the NEW one is a guaranteed SEED_MISMATCH (not a
    # recoverable GRAIL bet) → the drop must not depend on a launch env var.
    from reliquary.miner.engine import drop_pool_on_ckpt_advance
    monkeypatch.delenv("RELIQUARY_DROP_POOL_ON_CKPT", raising=False)
    monkeypatch.setattr(constants, "FORCED_SEED_ENFORCE", True)
    assert drop_pool_on_ckpt_advance() is True   # forced even without env var
    monkeypatch.setattr(constants, "FORCED_SEED_ENFORCE", False)
    assert drop_pool_on_ckpt_advance() is False  # legacy default: optimistic
    monkeypatch.setenv("RELIQUARY_DROP_POOL_ON_CKPT", "1")
    assert drop_pool_on_ckpt_advance() is True   # env var still honoured


def test_pool_persistence_off_under_enforcement(monkeypatch):
    monkeypatch.setattr(constants, "FORCED_SEED_ENFORCE", True)
    assert pool_persist_enabled(_SENTINEL) is False
    monkeypatch.setattr(constants, "FORCED_SEED_ENFORCE", False)
    assert pool_persist_enabled(_SENTINEL) is True
    assert pool_persist_enabled(12345) is False  # range armed → still off
