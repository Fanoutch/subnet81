"""Boucles du mineur SFT avec faux validateur et faux vLLM (upstream requis :
``reliquary.miner.corpus_miner`` pour build_submission / _retry)."""
from collections import Counter
import threading

from reliquary.miner.corpus_miner import Generation

from sft_miner.book import CursorBook
from sft_miner.miner import gpu_loop, submit_loop


class _Job:
    job_id = "job"
    checkpoint_sha256 = "sha"
    eos_token_id = 2
    prompt_count = 100


class _Tok:
    def decode(self, tokens, **kw):
        return " ".join(map(str, tokens))


def _gens():
    return [Generation([5, 2], ["p"]) for _ in range(4)]


class _Validator:
    """Curseur attendu côté validateur ; ``plan`` force une raison par curseur."""

    def __init__(self, start=0, plan=None):
        self.expected = start
        self.plan = dict(plan or {})
        self.bodies = []

    def cursor(self, hk):
        return self.expected

    def submit(self, body):
        self.bodies.append(body["cursor"])
        c = body["cursor"]
        if c != self.expected:
            return {"accepted": False, "reason": "bad_cursor"}
        reason = self.plan.pop(c, "accepted")
        if reason == "accepted":
            self.expected += 1
            return {"accepted": True, "reason": "accepted"}
        if reason == "prompt_full":
            self.expected += 1                    # le curseur avance quand même
        return {"accepted": False, "reason": reason}


def _run_submit(book, validator, stop_after=None):
    stats = Counter()
    t = threading.Thread(target=submit_loop, kwargs=dict(
        book=book, client=validator, job=_Job(), hotkey="hk", tokenizer=_Tok(),
        sign=lambda b: "sig", stats=stats, sleep=lambda s: None))
    t.start()
    return t, stats


def _fill(book, cursors):
    for c in cursors:
        assert book.next_admission() == c
    for c in cursors:
        book.on_generated(c, (c, f"prompt {c}", _gens()))


def test_submits_in_order_and_advances():
    book = CursorBook(start=0, max_in_flight=4, max_ahead=8)
    v = _Validator()
    _fill(book, [0, 1, 2, 3])
    t, stats = _run_submit(book, v)
    while v.expected < 4:
        pass
    book.stop("fin")
    t.join(5)
    assert v.bodies == [0, 1, 2, 3]
    assert stats["accepted"] == 4


def test_prompt_full_realigns_and_keeps_going():
    book = CursorBook(start=0, max_in_flight=4, max_ahead=8)
    v = _Validator(plan={1: "prompt_full"})
    _fill(book, [0, 1, 2, 3])
    t, stats = _run_submit(book, v)
    while v.expected < 4:
        pass
    book.stop("fin")
    t.join(5)
    assert v.bodies == [0, 1, 2, 3]            # 2 et 3 ne sont pas perdus
    assert stats["prompt_full"] == 1


def test_job_complete_stops_everything():
    book = CursorBook(start=0, max_in_flight=2, max_ahead=8)
    v = _Validator(plan={0: "job_complete"})
    _fill(book, [0, 1])
    t, _ = _run_submit(book, v)
    t.join(5)
    assert book.stopped == "job_complete"
    assert v.bodies == [0]


def test_banned_hotkey_halts():
    book = CursorBook(start=0, max_in_flight=2, max_ahead=8)
    v = _Validator(plan={0: "miner_banned"})
    _fill(book, [0, 1])
    t, _ = _run_submit(book, v)
    t.join(5)
    assert book.stopped == "miner_banned"


class _FakeStream:
    """Chaque ``step`` finit le plus ancien prompt ; ``fail_once`` simule une
    préemption au premier passage du curseur."""

    def __init__(self, fail_once=()):
        self.live = []
        self.fail = set(fail_once)
        self.added = []
        self.tokens_done = 0

    def add(self, cursor, ids):
        self.live.append(cursor)
        self.added.append(cursor)

    def abort(self, cursor):
        self.live.remove(cursor)

    def busy(self):
        return bool(self.live)

    def step(self):
        c = self.live.pop(0)
        if c in self.fail:
            self.fail.discard(c)
            return [(c, None)]
        self.tokens_done += 8
        return [(c, _gens())]


def test_gpu_loop_regenerates_a_preempted_prompt_and_keeps_order():
    book = CursorBook(start=0, max_in_flight=3, max_ahead=5)
    stream = _FakeStream(fail_once={1})
    stats = Counter()
    t = threading.Thread(target=gpu_loop, kwargs=dict(
        book=book, stream=stream, prompt_for=lambda c: (c, f"p{c}", [1, 2]), stats=stats,
        deadline=None, sleep=lambda s: None))
    t.start()
    got = []
    while len(got) < 5:
        item = book.take_submittable()
        if item:
            got.append(item[0])
            book.on_accepted(item[0])
    book.stop("fin")
    t.join(5)
    assert got == [0, 1, 2, 3, 4]
    assert stream.added.count(1) == 2              # régénéré une fois
    assert stats["regenerated"] == 1


class _Out:
    def __init__(self, rid, tokens):
        self.request_id, self.finished = rid, True
        self.outputs = [type("O", (), {"token_ids": tokens})()]


class _RenamingEngine:
    """Comme vLLM 0.30 : ``add_request`` renomme la requête en ``<id>-<8 car.>``
    (c'est ce nom que voit la capture), mais ``step`` rend l'id EXTERNE."""

    def __init__(self, capture):
        self.capture = capture
        self.pending = []

    def add_request(self, rid, prompt, params):
        import torch
        internal = f"{rid}-deadbeef"
        tokens = [7] * 40 + [2]
        n = len(prompt["prompt_token_ids"])
        # la capture couvre prompt + réponse (préremplissage compris)
        self.capture.rows[internal] = torch.randn(n + len(tokens) - 1, 256, dtype=torch.bfloat16)
        self.pending.append((rid, tokens))
        return internal

    def step(self):
        out, self.pending = [_Out(r, t) for r, t in self.pending], []
        return out

    def has_unfinished_requests(self):
        return bool(self.pending)

    def abort_request(self, ids):
        pass


class _PrefixCapture:
    """Même règle de correspondance que ``HiddenStateCapture.pop`` d'upstream."""

    def __init__(self):
        self.rows = {}

    def pop(self, rid):
        m = [r for r in self.rows if r == rid or r.startswith(rid + "-")]
        if len(m) != 1:
            raise KeyError(rid)
        return self.rows.pop(m[0])


def test_stream_matches_outputs_by_external_request_id():
    from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS as PROOF
    from sft_miner.miner import VllmStream

    capture = _PrefixCapture()
    gen = type("G", (), {})()
    gen._llm = type("L", (), {"llm_engine": _RenamingEngine(capture)})()
    gen._capture, gen._proof = capture, PROOF
    gen._params = type("P", (), {"clone": lambda self: type("Q", (), {})()})()
    stream = VllmStream(gen, n=4)
    stream.add(5, [1, 2, 3])
    done = stream.step()
    assert [c for c, g in done] == [5]
    assert done[0][1] is not None and len(done[0][1]) == 4
    assert stream.tokens_done == 4 * 41
    assert capture.rows == {}                       # lignes libérées


def test_gpu_loop_does_not_admit_while_kv_cache_is_full():
    """Admission pilotée par l'occupation du cache : au-dessus du seuil, aucun
    nouveau prompt ne démarre (sinon vLLM évince et recalcule des séquences,
    dont la preuve devient inutilisable)."""
    book = CursorBook(start=0, max_in_flight=8, max_ahead=64)
    stream = _FakeStream()
    usage = {"v": 0.0}
    stream.can_admit = lambda: usage["v"] < 0.6
    admitted_while_full = []

    def add(cursor, ids):
        if usage["v"] >= 0.6:
            admitted_while_full.append(cursor)
        stream.live.append(cursor)
        stream.added.append(cursor)
        usage["v"] += 0.25                           # chaque prompt occupe du cache

    stream.add = add
    base_step = stream.step

    def step():
        out = base_step()
        usage["v"] = max(0.0, usage["v"] - 0.25)
        return out

    stream.step = step
    stats = Counter()
    t = threading.Thread(target=gpu_loop, kwargs=dict(
        book=book, stream=stream, prompt_for=lambda c: (c, f"p{c}", [1]), stats=stats,
        deadline=None, sleep=lambda s: None))
    t.start()
    got = []
    while len(got) < 6:
        item = book.take_submittable()
        if item:
            got.append(item[0])
            book.on_accepted(item[0])
    book.stop("fin")
    t.join(5)
    assert got == [0, 1, 2, 3, 4, 5]
    assert admitted_while_full == []


def test_gpu_loop_admits_at_most_one_new_prompt_per_step():
    """La jauge du cache ne voit un prompt qu'après le pas qui le préremplit :
    en admettre plusieurs d'un coup sous le seuil fait déborder le cache."""
    book = CursorBook(start=0, max_in_flight=8, max_ahead=64)
    stream = _FakeStream()
    stream.can_admit = lambda: True
    per_step, count = [], {"n": 0}
    base_add, base_step = stream.add, stream.step

    def add(cursor, ids):
        count["n"] += 1
        base_add(cursor, ids)

    def step():
        per_step.append(count["n"])
        count["n"] = 0
        return base_step()

    stream.add, stream.step = add, step
    t = threading.Thread(target=gpu_loop, kwargs=dict(
        book=book, stream=stream, prompt_for=lambda c: (c, f"p{c}", [1]), stats=Counter(),
        deadline=None, sleep=lambda s: None))
    t.start()
    got = []
    while len(got) < 6:
        item = book.take_submittable()
        if item:
            got.append(item[0])
            book.on_accepted(item[0])
    book.stop("fin")
    t.join(5)
    assert max(per_step) <= 1
