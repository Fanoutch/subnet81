"""Fixes 19/08 (rapport croisé 2 agents) : fork grading, tir à l'append,
timeouts de poll, dé-sérialisation stream_fire, décodage unique."""
import asyncio
import resource

import pytest

from reliquary.environment.code_grader import (
    _CHILD_PRELUDE, _MEM_LIMIT_BYTES, grade_completion,
)


def test_grade_completion_no_preexec_same_semantics():
    # même sémantique de correction (P/F par assertion), sans preexec_fn
    comp = "def add(a, b):\n    return a + b\n"
    cases = ["assert add(1, 2) == 3", "assert add(0, 0) == 0",
             "assert add(1, 1) == 3"]
    assert grade_completion(comp, cases) == pytest.approx(2 / 3)
    assert grade_completion("", []) == 0.0


def test_child_memory_limit_still_enforced():
    # la limite mémoire est posée PAR L'ENFANT (prélude) : une allocation
    # au-delà de 512 Mo doit échouer DANS le code gradé → cas F, pas crash.
    comp = "def big():\n    x = bytearray(%d)\n    return len(x)\n" % (
        _MEM_LIMIT_BYTES * 2)
    cases = ["assert big() > 0"]
    assert grade_completion(comp, cases) == 0.0


def test_prelude_sets_rlimit():
    assert "setrlimit" in _CHILD_PRELUDE and str(_MEM_LIMIT_BYTES) in _CHILD_PRELUDE


def test_driver_sets_rlimit_first():
    src = open("reliquary/environment/code_grader_driver.py").read()
    body = src.split('"""', 2)[2]  # après le docstring
    assert "setrlimit" in body.split("def ")[0]  # avant toute fonction


def test_maybe_fire_on_append_guards():
    from reliquary.miner.engine import MiningEngine
    from reliquary.protocol.submission import WindowState

    e = MiningEngine.__new__(MiningEngine)
    # aucun contexte → no-op sûr
    assert e._maybe_fire_on_append() is False

    class _St:
        state = WindowState.OPEN
        randomness = "r1"
        window_n = 100
    e._last_state = _St()
    e._fire_ctx = ("url", None, [])
    e._cached_randomness = "r1"
    e._cached_window_n = 100
    e._inflight_fire_tasks = set()
    e._submitted_count = {}
    e._sealed_window = None
    e._fire_as_ready = lambda w, r: True
    fired = {}

    async def _fake_fire(st, url, client, results, budget=0):
        fired["budget"] = budget
    e._fire_for_window = _fake_fire
    e._pool = []

    async def _run():
        assert e._maybe_fire_on_append() is True
        assert len(e._inflight_fire_tasks) == 1
        # un 2e append pendant le vol : gardé (un seul fire en vol)
        assert e._maybe_fire_on_append() is False
        await asyncio.gather(*e._inflight_fire_tasks)
    asyncio.run(_run())
    assert fired["budget"] > 0

    # fenêtre scellée → refus (anti-boucle du re-tir)
    e._sealed_window = 100
    async def _run2():
        assert e._maybe_fire_on_append() is False
    asyncio.run(_run2())

    # randomness périmée → refus
    e._sealed_window = None
    e._cached_randomness = "r2"
    async def _run3():
        assert e._maybe_fire_on_append() is False
    asyncio.run(_run3())


def test_proof_rollouts_accepts_texts_param():
    import inspect
    from reliquary.miner.engine import MiningEngine
    sig = inspect.signature(MiningEngine._proof_rollouts)
    assert "texts" in sig.parameters


def test_fused_argmax_out_parity():
    import torch
    from reliquary.miner.engine import _chunked_chosen_logprobs_fused
    g = torch.Generator().manual_seed(11)
    h = torch.randn(80, 32, generator=g)
    lm = torch.nn.Linear(32, 300, bias=False)
    with torch.no_grad():
        lm.weight.copy_(torch.randn(300, 32, generator=g))
    toks = torch.randint(0, 300, (80,), generator=g).tolist()
    amx = []
    lps = _chunked_chosen_logprobs_fused(h, lm, toks, 20, chunk=16,
                                         argmax_out=amx)
    assert len(amx) == len(lps) == 60
    with torch.no_grad():
        ref = torch.log_softmax(lm(h).float(), dim=-1)[19:79]
    for i, a in enumerate(amx):
        assert abs(a - float(ref[i].max().exp())) < 1e-6
        # argmax >= chosen toujours
        import math
        assert a >= math.exp(lps[i]) - 1e-9


def test_local_verif_screen(monkeypatch):
    import math
    from reliquary.miner import engine
    monkeypatch.setattr(engine, "MIN_LOCAL_Q10", 5e-4)
    monkeypatch.setattr(engine, "MIN_LOCAL_MEDIAN", 0.08)
    ok_lps = [math.log(0.5)] * 40
    assert engine.local_verif_screen(ok_lps, [0.6] * 40) is None
    # q10 effondré (4+ positions très basses sur 40)
    bad = [math.log(1e-5)] * 6 + [math.log(0.5)] * 34
    assert engine.local_verif_screen(bad, None) == "local_q10"
    # token-auth : chosen minuscule + argmax ultra-confiant
    lps = [math.log(0.5)] * 29 + [math.log(5e-5)]
    amx = [0.6] * 29 + [0.995]
    assert engine.local_verif_screen(lps, amx) == "local_token_auth"
    # même chosen minuscule mais argmax pas confiant → sain (queue légitime)
    amx2 = [0.6] * 29 + [0.5]
    assert engine.local_verif_screen(lps, amx2) is None
    # rollout court (<30) : pas de screen distribution, token-auth seul
    assert engine.local_verif_screen([math.log(1e-5)] * 10, [0.5] * 10) is None
