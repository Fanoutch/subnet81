"""La réparation de l'EOS ne doit plus retenir le bake (mesuré 14/09, fen 45898).

Les continuations de réparation sont injectées dans le bake en streaming. La
boucle du bake tournait tant que le moteur avait des requêtes — continuations
comprises : dernier groupe prêt à 24,6 s, bake rendu à 57,7 s (une continuation
filait vers le plafond de 8 192 tokens), donc le bake suivant démarrait 33 s
trop tard. Sur 8 bakes : +5 à +33 s.

Invariants :
- le bake rend la main dès que SES groupes sont livrés, continuations en vol
  ou non ;
- une continuation laissée en vol n'est pas perdue : l'appelant qui l'attend
  pilote le moteur, ou le bake suivant l'adopte ;
- l'appelant qui pilote cède le moteur dès qu'un bake le réclame ;
- un abandon (flip) ou un rechargement libère les appelants avec ``None`` ;
- chaque continuation est plafonnée par ``max_new_tokens``.
"""
from __future__ import annotations

import threading
import time
import types

import pytest

from reliquary.miner import vllm_backend as vb
from reliquary.miner.vllm_backend import VLLMBackend
from reliquary.miner.vllm_forced_seed import FORCED_SEED_EXTRA_KEY


class _Out:
    def __init__(self, rid, token_ids):
        self.request_id = rid
        self.finished = True
        self.outputs = [types.SimpleNamespace(
            token_ids=list(token_ids), stop_reason=None, finish_reason="stop")]


class _Engine:
    """Bake : fini après ``bake_steps`` ; continuation : ``cont_steps``.
    Sortie = [base_offset, 99]."""

    def __init__(self, bake_steps=20, cont_steps=400):
        self.bake_steps, self.cont_steps = bake_steps, cont_steps
        self.active: dict[str, list] = {}
        self.added: list = []
        self.aborted: list = []
        self.lock = threading.Lock()
        self.steppers: set = set()

    def add_request(self, rid, prompt, params):
        with self.lock:
            self.added.append((rid, prompt, params))
            steps = self.cont_steps if rid.startswith("cont-") else self.bake_steps
            self.active[rid] = [steps, params]

    def has_unfinished_requests(self):
        with self.lock:
            return bool(self.active)

    def step(self):
        me = threading.get_ident()
        with self.lock:
            self.steppers.add(me)
            assert len(self.steppers) == 1, "deux fils pilotent le moteur"
        time.sleep(0.002)
        outs = []
        with self.lock:
            for rid in list(self.active):
                self.active[rid][0] -= 1
                if self.active[rid][0] <= 0:
                    params = self.active.pop(rid)[1]
                    fs = params.extra_args[FORCED_SEED_EXTRA_KEY]
                    outs.append(_Out(rid, [fs["base_offset"], 99]))
            self.steppers.discard(me)
        return outs

    def abort_request(self, rids):
        with self.lock:
            for rid in (rids if isinstance(rids, (list, tuple)) else [rids]):
                self.active.pop(rid, None)
                self.aborted.append(rid)


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


ITEM = [{"prefix_tokens": [1, 2, 3, 40], "prompt_idx": 11, "rollout_index": 5,
         "prompt_len": 2}]


def _bake(b, should_abort=None, on_group=None):
    return b.generate_forced_phase1_multi_stream(
        [[1, 2, 3]], prompt_indices=[7], randomness="ab", checkpoint_hash="c",
        m_rollouts=2, max_tokens=64, stop_token_ids=[99], primary_eos_id=99,
        should_abort=should_abort, on_group=on_group)


def _cont(b, res, timeout=10.0, **kw):
    res["out"] = b.run_forced_continuations(
        ITEM, randomness="ab", checkpoint_hash="c", max_tokens=100,
        stop_token_ids=[99], primary_eos_id=99, timeout=timeout, **kw)
    res["t"] = time.monotonic()


def test_continuation_en_vol_au_retour_du_bake_est_servie():
    eng = _Engine(bake_steps=50, cont_steps=600)
    b = _backend(eng)
    res = {}
    fired = threading.Event()

    def on_group(pos, pidx, group):
        pass

    def inject():
        # attendre que le bake tienne le moteur, puis injecter
        while not b._stream_active.is_set():
            time.sleep(0.001)
        fired.set()
        _cont(b, res)

    th = threading.Thread(target=inject)
    th.start()
    t0 = time.monotonic()
    out = _bake(b, on_group=on_group)
    t_bake = time.monotonic() - t0
    th.join(10)
    assert fired.is_set()
    assert out == [[[0, 99], [0, 99]]]
    # le bake (50 steps ≈ 0,1 s) ne doit PAS attendre la continuation (600 steps)
    assert t_bake < 0.8, f"bake retenu par la continuation : {t_bake:.2f}s"
    assert res["out"] == [[2, 99]], "continuation perdue au retour du bake"
    assert not eng.aborted


def test_bake_suivant_adopte_la_continuation_et_nattend_pas():
    eng = _Engine(bake_steps=50, cont_steps=700)
    b = _backend(eng)
    res = {}

    def inject():
        while not b._stream_active.is_set():
            time.sleep(0.001)
        _cont(b, res)

    th = threading.Thread(target=inject)
    th.start()
    _bake(b)
    # bake suivant immédiatement : il doit obtenir le moteur malgré l'appelant
    # qui pilote la continuation, et rendre la main à la fin de SES groupes
    t0 = time.monotonic()
    out2 = _bake(b)
    t_bake2 = time.monotonic() - t0
    th.join(10)
    assert out2 == [[[0, 99], [0, 99]]]
    assert t_bake2 < 0.8, f"bake suivant bloqué : {t_bake2:.2f}s"
    assert res["out"] == [[2, 99]]
    assert not eng.aborted


def test_flip_pendant_le_bake_libere_lappelant():
    eng = _Engine(bake_steps=10_000, cont_steps=10_000)
    b = _backend(eng)
    res = {}
    stop = threading.Event()

    def inject():
        while not b._stream_active.is_set():
            time.sleep(0.001)
        _cont(b, res)
        stop.set()

    th = threading.Thread(target=inject)
    th.start()
    time.sleep(0.05)
    t_flip = time.monotonic() + 0.2
    _bake(b, should_abort=lambda: time.monotonic() > t_flip)
    th.join(5)
    assert stop.is_set()
    assert res["out"] is None
    assert any(r.startswith("cont-") for r in eng.aborted)


def test_rechargement_libere_les_continuations_en_vol(monkeypatch):
    eng = _Engine(bake_steps=50, cont_steps=100_000)
    b = _backend(eng)
    res = {}

    def inject():
        while not b._stream_active.is_set():
            time.sleep(0.001)
        _cont(b, res, timeout=30.0)

    th = threading.Thread(target=inject)
    th.start()
    t_b = time.monotonic()
    _bake(b)
    assert time.monotonic() - t_b < 0.8, "bake retenu par la continuation"
    time.sleep(0.05)
    monkeypatch.setattr(b, "_reload_locked", lambda path: None, raising=False)
    t0 = time.monotonic()
    b.reload("/new/ckpt")
    th.join(5)
    assert res.get("out", "absent") is None
    assert res["t"] - t0 < 2.0, "l'appelant a attendu son délai au lieu d'être libéré"


def test_plafond_de_tokens_par_continuation():
    eng = _Engine(bake_steps=1, cont_steps=1)
    b = _backend(eng)
    b.run_forced_continuations(
        [{"prefix_tokens": [1, 2, 3, 40], "prompt_idx": 11, "rollout_index": 5,
          "prompt_len": 2},
         {"prefix_tokens": [1, 2] + [7] * 97, "prompt_idx": 11,
          "rollout_index": 6, "prompt_len": 2}],
        randomness="ab", checkpoint_hash="c", max_tokens=100,
        stop_token_ids=[99], primary_eos_id=99, timeout=5.0, max_new_tokens=8)
    # min(plafond de complétion restant, budget de réparation)
    assert [p.max_tokens for _, _, p in eng.added] == [8, 3]
