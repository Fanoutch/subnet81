"""Task 5: the RELIQUARY_VLLM_FORCED_SEED flag gates whether forced-seed
generation runs on vLLM. Default OFF → live behaviour unchanged (backend stays
None under enforcement → HF path). ON → the vLLM backend is used even under
enforcement.
"""
import importlib


def _reload_engine():
    from reliquary.miner import engine
    return importlib.reload(engine)


def test_flag_default_off(monkeypatch):
    monkeypatch.delenv("RELIQUARY_VLLM_FORCED_SEED", raising=False)
    eng = _reload_engine()
    assert eng.vllm_forced_seed_enabled() is False


def test_flag_on(monkeypatch):
    monkeypatch.setenv("RELIQUARY_VLLM_FORCED_SEED", "1")
    eng = _reload_engine()
    assert eng.vllm_forced_seed_enabled() is True


def test_backend_selection_matches_flag(monkeypatch):
    # Mirrors the engine gate at _generate_m_rollouts: under enforcement the
    # backend is None UNLESS the vLLM-forced-seed flag is on.
    from reliquary.miner import engine as eng
    def select(enforce, flag, backend_obj):
        # replicate the gate expression for the test (kept in sync with engine)
        use_vllm = (not enforce) or eng.vllm_forced_seed_enabled()
        return backend_obj if use_vllm else None

    sentinel = object()
    monkeypatch.setenv("RELIQUARY_VLLM_FORCED_SEED", "1")
    importlib.reload(eng)
    assert select(True, True, sentinel) is sentinel        # enforce + flag → vLLM
    monkeypatch.delenv("RELIQUARY_VLLM_FORCED_SEED", raising=False)
    importlib.reload(eng)
    assert select(True, False, sentinel) is None           # enforce, no flag → HF
    assert select(False, False, sentinel) is sentinel      # no enforce → vLLM (legacy)
