"""Continuations forced-seed injectées dans le moteur vLLM (réparation de l'EOS).

Contexte (14/09) : aucun réglage vLLM ne rend l'EOS final exact vis-à-vis du
forward du validateur (plafond ~90 %). La réparation prolonge un rollout refusé
à partir du token que le validateur tire, via vLLM lui-même. Le moteur est tenu
par le bake en streaming (``_VLLM_CALL_LOCK`` pendant tout le lot) : les
continuations doivent donc pouvoir être INJECTÉES dans le lot en cours, et
servies directement quand aucun lot ne tourne.

Invariants :
- extra_args forced-seed : ``rollout_index`` du rollout, ``base_offset`` =
  offset de complétion du premier token généré, ``start_len`` = longueur du
  préfixe — un écart = un flux différent = SEED_MISMATCH ;
- chemin direct (aucun bake) et chemin injecté (bake en cours) rendent le même
  résultat ;
- un abandon (flip de fenêtre / interruption) libère l'appelant avec ``None``.
"""
from __future__ import annotations

import threading
import time
import types

from reliquary.miner import vllm_backend as vb
from reliquary.miner.vllm_backend import VLLMBackend
from reliquary.miner.vllm_forced_seed import FORCED_SEED_EXTRA_KEY


class _Out:
    def __init__(self, rid, token_ids):
        self.request_id = rid
        self.finished = True
        self.outputs = [types.SimpleNamespace(
            token_ids=list(token_ids), stop_reason=None, finish_reason="stop")]


class _DynEngine:
    """Chaque requête finit après ``steps`` pas ; sortie = [base_offset, 99]."""

    def __init__(self, steps=2):
        self.steps = steps
        self.active: dict[str, list] = {}
        self.added: list = []
        self.aborted: list = []
        self.lock = threading.Lock()

    def add_request(self, rid, prompt, params):
        with self.lock:
            self.added.append((rid, prompt, params))
            self.active[rid] = [self.steps, params]

    def has_unfinished_requests(self):
        with self.lock:
            return bool(self.active)

    def step(self):
        time.sleep(0.005)
        outs = []
        with self.lock:
            for rid in list(self.active):
                self.active[rid][0] -= 1
                if self.active[rid][0] <= 0:
                    params = self.active.pop(rid)[1]
                    fs = params.extra_args[FORCED_SEED_EXTRA_KEY]
                    outs.append(_Out(rid, [fs["base_offset"], 99]))
        return outs

    def abort_request(self, rids):
        with self.lock:
            for rid in (rids if isinstance(rids, (list, tuple)) else [rids]):
                self.active.pop(rid, None)
                self.aborted.append(rid)


import pytest


@pytest.fixture(autouse=True)
def _fake_vllm(monkeypatch):
    import sys

    class _SP:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class _TP:
        def __init__(self, prompt_token_ids):
            self.prompt_token_ids = prompt_token_ids

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(SamplingParams=_SP))
    monkeypatch.setitem(sys.modules, "vllm.inputs", types.SimpleNamespace(TokensPrompt=_TP))
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", types.SimpleNamespace(
        RequestOutputKind=types.SimpleNamespace(FINAL_ONLY="final_only")))


def _backend(engine):
    b = VLLMBackend.__new__(VLLMBackend)
    b._llm = types.SimpleNamespace(llm_engine=engine)
    b._interrupt = threading.Event()
    b._ensure_loaded = lambda: None
    return b


ITEMS = [
    {"prefix_tokens": [1, 2, 3, 40], "prompt_idx": 11, "rollout_index": 5,
     "prompt_len": 2},
    {"prefix_tokens": [1, 2, 7, 8, 41], "prompt_idx": 11, "rollout_index": 9,
     "prompt_len": 2},
]


def test_chemin_direct_sans_bake():
    eng = _DynEngine()
    b = _backend(eng)
    res = b.run_forced_continuations(
        ITEMS, randomness="ab", checkpoint_hash="c", max_tokens=100,
        stop_token_ids=[99], primary_eos_id=99, timeout=5.0)
    # base_offset = len(prefix) - prompt_len (offset de complétion du 1er token)
    assert res == [[2, 99], [3, 99]]
    fs = [p.extra_args[FORCED_SEED_EXTRA_KEY] for _, _, p in eng.added]
    assert [(f["rollout_index"], f["base_offset"], f["start_len"], f["prompt_idx"])
            for f in fs] == [(5, 2, 4, 11), (9, 3, 5, 11)]
    assert [p.max_tokens for _, _, p in eng.added] == [98, 97]


def test_injection_pendant_un_bake_en_cours():
    eng = _DynEngine(steps=3)
    b = _backend(eng)
    started = threading.Event()
    results = {}

    def bake():
        # simule le driver du stream : tient le verrou, une requête longue
        with vb._VLLM_CALL_LOCK:
            b._stream_active.set()
            eng.add_request("bake-0", None, types.SimpleNamespace(
                extra_args={FORCED_SEED_EXTRA_KEY: {"base_offset": 0}}))
            eng.active["bake-0"][0] = 200
            started.set()
            cont = {}
            while eng.has_unfinished_requests():
                b._drain_continuation_queue(eng, cont)
                for out in eng.step():
                    b._route_continuation_output(out, cont)
            b._stream_active.clear()

    th = threading.Thread(target=bake)
    th.start()
    started.wait(2)
    t0 = time.monotonic()
    results["res"] = b.run_forced_continuations(
        ITEMS, randomness="ab", checkpoint_hash="c", max_tokens=100,
        stop_token_ids=[99], primary_eos_id=99, timeout=5.0)
    dt = time.monotonic() - t0
    eng.abort_request(["bake-0"])
    th.join(5)
    assert results["res"] == [[2, 99], [3, 99]]
    assert dt < 0.9          # servie pendant le bake, sans attendre sa fin


def test_interruption_libere_lappelant():
    eng = _DynEngine(steps=10_000)
    b = _backend(eng)
    threading.Timer(0.2, b._interrupt.set).start()
    res = b.run_forced_continuations(
        ITEMS, randomness="ab", checkpoint_hash="c", max_tokens=100,
        stop_token_ids=[99], primary_eos_id=99, timeout=5.0)
    assert res is None
    assert set(eng.aborted) >= {rid for rid, _, _ in eng.added}


def test_timeout_libere_lappelant():
    eng = _DynEngine(steps=10_000)
    b = _backend(eng)
    t0 = time.monotonic()
    res = b.run_forced_continuations(
        ITEMS, randomness="ab", checkpoint_hash="c", max_tokens=100,
        stop_token_ids=[99], primary_eos_id=99, timeout=0.3)
    assert res is None
    assert time.monotonic() - t0 < 2.0
