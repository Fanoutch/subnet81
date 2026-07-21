"""Batch phase-1 across prompts via a prefetch cache.

Measured 2026-07-21: under forced-seed the bake loop calls
`generate_forced_phase1` once per prompt (~40s), so a 6-prompt bake costs ~264s
against a 100s collection window — the pool is always flushed stale at the
randomness flip and nothing is ever submitted.

`_prefetch_phase1` runs ONE batched vLLM call for all prompts and parks the
completions; `_generate_m_rollouts` then consumes them instead of calling the
backend. Phase-2, the GRAIL proof forwards and grading are untouched.

The dangerous failure here is silent: a cached completion reused under a
different randomness/checkpoint still *generates fine*, but every token is
forced off the wrong stream and the validator rejects it as TOKEN_TAMPERED.
So the cache key and its single-use discipline are what these tests pin.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


class _Backend:
    """Records batched vs per-prompt calls."""

    def __init__(self):
        self.multi_calls = []
        self.single_calls = []

    def generate_forced_phase1_multi(self, prompts_token_ids, *, prompt_indices,
                                     randomness, checkpoint_hash, m_rollouts,
                                     max_tokens, stop_token_ids=None):
        self.multi_calls.append(
            {"prompt_indices": list(prompt_indices), "randomness": randomness}
        )
        return [[[900 + i] for _ in range(m_rollouts)]
                for i, _ in enumerate(prompts_token_ids)]

    def generate_forced_phase1(self, prompt_token_ids, *, randomness, prompt_idx,
                               checkpoint_hash, m_rollouts, max_tokens,
                               stop_token_ids=None):
        self.single_calls.append(prompt_idx)
        return [[1] for _ in range(m_rollouts)]


def _engine(backend, randomness="ab" * 32, ckpt="ckpt-1"):
    from reliquary.miner.engine import MiningEngine

    e = MiningEngine.__new__(MiningEngine)
    e._vllm_backend = backend
    e._local_hash = ckpt
    e._cached_randomness = randomness
    e._eos_ids = set()
    e.max_new_tokens = 8192
    e.tokenizer = SimpleNamespace(pad_token_id=0)
    e.wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="hk"))
    e._phase1_cache = {}
    return e


PROBLEMS = [{"prompt": "p one"}, {"prompt": "p two"}, {"prompt": "p three"}]
IDXS = [7, 8, 9]


@pytest.fixture(autouse=True)
def _vllm_forced_seed_on(monkeypatch):
    """The prefetch only applies on the vLLM forced-seed path (flag=1)."""
    monkeypatch.setenv("RELIQUARY_VLLM_FORCED_SEED", "1")


def _prefetch(e, backend, randomness=None, encode=None):
    import reliquary.miner.engine as eng

    # encode_prompt is module-level in engine; stub it so no tokenizer is needed
    orig = eng.encode_prompt
    eng.encode_prompt = encode or (lambda tok, text: [1, 2, 3])
    try:
        return e._prefetch_phase1(
            PROBLEMS, IDXS,
            randomness=randomness or e._cached_randomness,
            env=SimpleNamespace(name="openmathinstruct"),
        )
    finally:
        eng.encode_prompt = orig


def test_prefetch_issues_exactly_one_batched_call_for_all_prompts():
    """This is the whole win: 1 call instead of len(PROMPTS)."""
    b = _Backend()
    e = _engine(b)
    _prefetch(e, b)
    assert len(b.multi_calls) == 1
    assert b.multi_calls[0]["prompt_indices"] == IDXS
    assert b.single_calls == []


def test_cached_completions_are_keyed_by_randomness_and_checkpoint():
    """Reuse under a different window seed = every token forced off-stream."""
    b = _Backend()
    e = _engine(b, randomness="aa" * 32, ckpt="ckpt-1")
    _prefetch(e, b)
    keys = list(e._phase1_cache)
    assert all(k[1] == "aa" * 32 and k[2] == "ckpt-1" for k in keys), keys
    assert sorted(k[0] for k in keys) == sorted(IDXS)


def test_a_stale_randomness_never_hits_the_cache():
    """After a flip, the cached entries must be ignored, not silently reused."""
    b = _Backend()
    e = _engine(b, randomness="aa" * 32)
    _prefetch(e, b)
    assert e._take_prefetched_phase1(7, "ff" * 32, "ckpt-1") is None
    assert e._take_prefetched_phase1(7, "aa" * 32, "ckpt-1") is not None


def test_a_stale_checkpoint_never_hits_the_cache():
    """Checkpoint advance invalidates generation just like a randomness flip."""
    b = _Backend()
    e = _engine(b, ckpt="ckpt-1")
    _prefetch(e, b)
    assert e._take_prefetched_phase1(7, e._cached_randomness, "ckpt-2") is None


def test_a_cached_entry_is_consumed_once():
    """Single-use: a leftover must never be served to a later window."""
    b = _Backend()
    e = _engine(b)
    _prefetch(e, b)
    first = e._take_prefetched_phase1(8, e._cached_randomness, "ckpt-1")
    assert first is not None
    assert e._take_prefetched_phase1(8, e._cached_randomness, "ckpt-1") is None


def test_prefetch_is_a_noop_without_a_vllm_forced_seed_backend():
    """No backend => per-prompt path must stay exactly as it is today."""
    e = _engine(None)
    e._vllm_backend = None
    out = _prefetch(e, None)
    assert out == 0
    assert e._phase1_cache == {}


def test_generate_m_rollouts_consumes_the_prefetch_instead_of_calling_vllm():
    """The win only materialises if the per-prompt path actually uses the cache."""
    import reliquary.miner.engine as eng

    b = _Backend()
    e = _engine(b)
    _prefetch(e, b)
    e._bft_from_seqs = lambda seqs, prompt_tokens, **kw: [
        {"tokens": s, "prompt_length": len(prompt_tokens)} for s in seqs
    ]
    orig = eng.encode_prompt
    eng.encode_prompt = lambda tok, text: [1, 2, 3]
    try:
        out = e._generate_m_rollouts(
            PROBLEMS[0], e._cached_randomness,
            env=SimpleNamespace(name="openmathinstruct"), prompt_idx=IDXS[0],
        )
    finally:
        eng.encode_prompt = orig

    assert b.single_calls == [], "fell back to a per-prompt vLLM call"
    assert len(out) == 8


def test_generate_m_rollouts_falls_back_when_nothing_was_prefetched():
    """Cache miss must behave exactly as today — no silent empty generation."""
    import reliquary.miner.engine as eng

    b = _Backend()
    e = _engine(b)                      # no prefetch performed
    e._bft_from_seqs = lambda seqs, prompt_tokens, **kw: [
        {"tokens": s, "prompt_length": len(prompt_tokens)} for s in seqs
    ]
    orig = eng.encode_prompt
    eng.encode_prompt = lambda tok, text: [1, 2, 3]
    try:
        e._generate_m_rollouts(
            PROBLEMS[0], e._cached_randomness,
            env=SimpleNamespace(name="openmathinstruct"), prompt_idx=IDXS[0],
        )
    finally:
        eng.encode_prompt = orig

    assert b.single_calls == [IDXS[0]]


def test_backend_failure_leaves_the_cache_empty_rather_than_partial():
    """A partial cache would pair some prompts with another prompt's stream."""
    class _Boom(_Backend):
        def generate_forced_phase1_multi(self, *a, **k):
            raise RuntimeError("engine died")

    b = _Boom()
    e = _engine(b)
    _prefetch(e, b)          # must swallow: bake falls back to per-prompt
    assert e._phase1_cache == {}
