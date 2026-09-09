"""Re-warm post-checkpoint (chantier 2026-08-12).

Défaut : ``VLLMBackend.reload()`` ne fait que lâcher l'ancien moteur ; le
premier ``generate()`` de la fenêtre suivante paie rebuild + capture CUDA
graphs + JIT Triton (130-400 s mesurés) → fenêtre perdue à chaque avancée de
checkpoint (28569 le 2026-08-12).

Fix : ``VLLMBackend.warmup()`` paie ce coût immédiatement au reload, pendant
le temps mort post-flush, et l'engine l'appelle sur le chemin de swap réussi
(``_rewarm_after_reload``), env-débrayable via RELIQUARY_REWARM_ON_RELOAD=0.
"""
import sys
import types
import pytest


def _fake_vllm_modules(monkeypatch, generate_calls):
    """Injecte des modules vllm factices (la dev box n'a pas vLLM)."""
    vllm_mod = types.ModuleType("vllm")

    class _SP:
        def __init__(self, **kw): self.kw = kw
    vllm_mod.SamplingParams = _SP

    inputs_mod = types.ModuleType("vllm.inputs")

    class _TP:
        def __init__(self, prompt_token_ids): self.prompt_token_ids = prompt_token_ids
    inputs_mod.TokensPrompt = _TP

    monkeypatch.setitem(sys.modules, "vllm", vllm_mod)
    monkeypatch.setitem(sys.modules, "vllm.inputs", inputs_mod)

    class _FakeLLM:
        def generate(self, prompts, sp):
            generate_calls.append((prompts, sp))
            return []
    return _FakeLLM()


def test_warmup_builds_engine_and_generates(monkeypatch):
    from reliquary.miner.vllm_backend import VLLMBackend
    b = VLLMBackend("dummy-path", forced_seed=True)
    calls = []
    fake_llm = _fake_vllm_modules(monkeypatch, calls)
    ensured = []

    def _ensure(self):
        ensured.append(True)
        self._llm = fake_llm
    monkeypatch.setattr(VLLMBackend, "_ensure_loaded", _ensure)

    dt = b.warmup()
    assert ensured, "warmup doit construire le moteur (_ensure_loaded)"
    assert len(calls) == 1, "warmup doit faire exactement une mini-génération"
    prompts, sp = calls[0]
    assert sp.kw["max_tokens"] <= 32, "le warmup doit rester minuscule"
    assert "extra_args" in sp.kw, "le warmup doit passer par le chemin forced-seed (JIT du processeur)"
    assert dt is not None and dt >= 0.0


def test_warmup_never_raises(monkeypatch):
    from reliquary.miner.vllm_backend import VLLMBackend
    b = VLLMBackend("dummy-path", forced_seed=True)

    def _boom(self):
        raise RuntimeError("engine build exploded")
    monkeypatch.setattr(VLLMBackend, "_ensure_loaded", _boom)
    assert b.warmup() is None  # échec avalé, jamais levé


def test_engine_rewarm_helper_calls_backend(monkeypatch):
    from reliquary.miner import engine as eng
    monkeypatch.delenv("RELIQUARY_REWARM_ON_RELOAD", raising=False)
    called = []

    class _B:
        def warmup(self): called.append(True); return 1.0
    eng._rewarm_after_reload(_B())
    assert called, "le swap réussi doit chauffer le moteur"


def test_engine_rewarm_helper_env_off(monkeypatch):
    from reliquary.miner import engine as eng
    monkeypatch.setenv("RELIQUARY_REWARM_ON_RELOAD", "0")
    called = []

    class _B:
        def warmup(self): called.append(True)
    eng._rewarm_after_reload(_B())
    assert not called, "RELIQUARY_REWARM_ON_RELOAD=0 doit débrayer"


def test_engine_rewarm_helper_swallows_and_tolerates_missing(monkeypatch):
    from reliquary.miner import engine as eng
    monkeypatch.delenv("RELIQUARY_REWARM_ON_RELOAD", raising=False)

    class _B:
        def warmup(self): raise RuntimeError("boom")
    eng._rewarm_after_reload(_B())          # ne doit pas lever
    eng._rewarm_after_reload(object())      # backend sans warmup : no-op


def test_reload_vram_wait_stops_on_plateau(monkeypatch):
    """Le wait VRAM doit s'arrêter dès que la mémoire libre PLAFONNE (ancien
    moteur mort) au lieu de viser `uti*total+5` — cible inatteignable quand le
    modèle de preuve HF réside sur le GPU (30 s brûlées à CHAQUE reload,
    mesuré 13:28:12 le 2026-08-12 : free plafonné à 112.5 pour cible 114)."""
    import types as _types
    from reliquary.miner.vllm_backend import VLLMBackend

    b = VLLMBackend("dummy", gpu_memory_utilization=0.78)
    b._llm = object()  # force le chemin de teardown

    total = 140 * (1024 ** 3)
    frees = iter([100, 108, 112.5, 112.5, 112.5, 112.5, 112.5, 112.5, 112.5,
                  112.5, 112.5, 112.5, 112.5, 112.5, 112.5, 112.5])
    calls = {"n": 0}

    def mem_get_info():
        calls["n"] += 1
        try: f = next(frees)
        except StopIteration: f = 112.5
        return int(f * (1024 ** 3)), total

    fake_cuda = _types.SimpleNamespace(
        is_available=lambda: True,
        empty_cache=lambda: None,
        mem_get_info=mem_get_info,
    )
    fake_torch = _types.SimpleNamespace(cuda=fake_cuda)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    slept = []
    import time as _time
    monkeypatch.setattr(_time, "sleep", lambda s: slept.append(s))

    b.reload("new-path")
    # cible 0.78*140+5=114.2 jamais atteinte, mais plateau à 112.5 après ~3
    # lectures stables -> on doit sortir bien avant les 30 itérations du timeout
    assert sum(slept) <= 10, f"attente {sum(slept)}s : le plateau doit couper court (avant: 30s)"
    assert b._model_path == "new-path"


def test_reload_vram_wait_ignores_low_plateau(monkeypatch):
    """Incident 2026-08-12 22:22 : free stable à 10 GiB (l'ancien moteur n'a
    PAS ENCORE commencé à libérer) a été pris pour le plateau final →
    'proceeding' à 10 GiB → OOM d'init (rattrapé par le retry 1/5, mais 5
    échecs = génération MORTE). Un plateau ne vaut que si la libération a été
    observée (free remonté d'au moins 10 GiB depuis la 1re lecture) ou si on
    est près de la cible."""
    import types as _types
    from reliquary.miner.vllm_backend import VLLMBackend

    b = VLLMBackend("dummy", gpu_memory_utilization=0.78)
    b._llm = object()

    total = 140 * (1024 ** 3)
    seq = [10.0] * 8 + [40.0, 80.0, 112.5, 112.5, 112.5, 112.5, 112.5, 112.5,
                        112.5, 112.5, 112.5, 112.5, 112.5, 112.5]
    frees = iter(seq)

    def mem_get_info():
        try: f = next(frees)
        except StopIteration: f = 112.5
        return int(f * (1024 ** 3)), total

    fake_cuda = _types.SimpleNamespace(
        is_available=lambda: True, empty_cache=lambda: None,
        mem_get_info=mem_get_info,
    )
    monkeypatch.setitem(sys.modules, "torch", _types.SimpleNamespace(cuda=fake_cuda))
    slept = []
    import time as _time
    monkeypatch.setattr(_time, "sleep", lambda s: slept.append(s))

    b.reload("new-path")
    # il faut avoir SURVÉCU au faux plateau des 8 premières lectures (>=8s
    # d'attente) puis coupé sur le vrai plateau post-libération (<30s)
    assert sum(slept) >= 8, f"sorti à {sum(slept)}s : a pris le plateau bas pour la fin de libération"
    assert sum(slept) < 25, f"attente {sum(slept)}s : le vrai plateau post-libération doit couper court"
