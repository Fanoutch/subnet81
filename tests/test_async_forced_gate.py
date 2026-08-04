"""§3.5 rolling batch — the async loop is now allowed under forced-seed.

The July blocker ("AsyncLLMEngine cannot apply the forced-seed HF
LogitsProcessor") died with the native batched processor: the async backend
registers the SAME engine-level class (selector) as the sync path, and
AsyncVLLMBackend.generate_forced_phase1 threads the per-request payload. The
async loop may therefore run under FORCED_SEED_ENFORCE — but ONLY when the vLLM
forced path is explicitly enabled (the processor must be registered)."""
from reliquary.miner.engine import should_use_async_loop


def test_free_mode_keeps_legacy_behaviour():
    assert should_use_async_loop(True, False, False) is True
    assert should_use_async_loop(False, False, False) is False


def test_enforced_requires_vllm_forced_processor():
    # enforce + processor registered → async allowed (the §3.5 unlock)
    assert should_use_async_loop(True, True, True) is True
    # enforce without the registered processor → HF sync loop (unforced tokens
    # from the async engine would be SEED_MISMATCH)
    assert should_use_async_loop(True, True, False) is False


def test_sync_backend_never_async():
    assert should_use_async_loop(False, True, True) is False
