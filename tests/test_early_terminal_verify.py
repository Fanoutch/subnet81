"""Vérification de l'EOS final DÈS la sortie de chaque rollout (14/09 soir).

Mesuré (fen 45914-45917, 185 premiers tirs) : prêt → précommit reçu 14,0 s
p50, dont 9,4 s de réparation EOS. Le groupe attendait ses 16 rollouts, puis
les faisait vérifier d'un bloc par la réplique, derrière les autres groupes du
bake. Désormais chaque rollout part en vérification à l'instant où vLLM le
termine ; au livrage du groupe, le 1er tour de la réparation lit ces verdicts.

Invariants :
- le verdict n'est réutilisé que pour EXACTEMENT les mêmes tokens, sous le même
  contexte (randomness, checkpoint, modèle) ;
- rollout sans stop final : pas de vérification ;
- verdict manquant ou en erreur : la réparation interroge la réplique comme
  avant, pour ces seuls rollouts.
"""
from __future__ import annotations

import threading
import time
import types

from reliquary.miner import engine, replica_client
from reliquary.miner.early_terminal import EarlyTerminalVerifier

EOS = 99
CTX = ("ab" * 32, "c" * 40, "/snap/rev")


def _verify_fn(calls, delay=0.0, fail=False):
    def fn(ctx, prompt_idx, items):
        calls.append((ctx, prompt_idx, [it["rollout"] for it in items]))
        if delay:
            time.sleep(delay)
        if fail:
            return None
        return [{"ok": it["rollout"] != 2, "pick": 55 if it["rollout"] == 2 else EOS,
                 "cdf_miss": 0.0} for it in items]
    return fn


def test_verdict_reutilise_pour_les_memes_tokens():
    calls = []
    v = EarlyTerminalVerifier(_verify_fn(calls), eos_ids=[EOS])
    v.submit(CTX, 7, 0, [1, 2, 10, EOS], prompt_len=2)
    v.submit(CTX, 7, 2, [1, 2, 12, EOS], prompt_len=2)
    assert v.lookup(CTX, 7, 0, [1, 2, 10, EOS], timeout=2.0)["ok"] is True
    assert v.lookup(CTX, 7, 2, [1, 2, 12, EOS], timeout=2.0) == {
        "ok": False, "pick": 55, "cdf_miss": 0.0}
    v.close()


def test_tokens_differents_ou_autre_contexte_pas_de_verdict():
    v = EarlyTerminalVerifier(_verify_fn([]), eos_ids=[EOS])
    v.submit(CTX, 7, 0, [1, 2, 10, EOS], prompt_len=2)
    assert v.lookup(CTX, 7, 0, [1, 2, 11, EOS], timeout=2.0) is None
    other = ("ff" * 32,) + CTX[1:]
    assert v.lookup(other, 7, 0, [1, 2, 10, EOS], timeout=2.0) is None
    v.close()


def test_rollout_sans_stop_final_non_soumis():
    calls = []
    v = EarlyTerminalVerifier(_verify_fn(calls), eos_ids=[EOS])
    v.submit(CTX, 7, 0, [1, 2, 10, 11], prompt_len=2)
    time.sleep(0.05)
    assert calls == []
    assert v.lookup(CTX, 7, 0, [1, 2, 10, 11], timeout=0.1) is None
    v.close()


def test_echec_de_verification_pas_de_verdict():
    v = EarlyTerminalVerifier(_verify_fn([], fail=True), eos_ids=[EOS])
    v.submit(CTX, 7, 0, [1, 2, 10, EOS], prompt_len=2)
    assert v.lookup(CTX, 7, 0, [1, 2, 10, EOS], timeout=2.0) is None
    v.close()


def test_lookup_attend_un_verdict_en_cours():
    v = EarlyTerminalVerifier(_verify_fn([], delay=0.2), eos_ids=[EOS])
    v.submit(CTX, 7, 0, [1, 2, 10, EOS], prompt_len=2)
    t0 = time.monotonic()
    assert v.lookup(CTX, 7, 0, [1, 2, 10, EOS], timeout=2.0)["ok"] is True
    assert time.monotonic() - t0 >= 0.15
    v.close()


def test_anciens_contextes_purges():
    v = EarlyTerminalVerifier(_verify_fn([]), eos_ids=[EOS], keep_contexts=2)
    for k in range(4):
        ctx = (f"{k:02d}" * 32,) + CTX[1:]
        v.submit(ctx, 7, 0, [1, 2, 10, EOS], prompt_len=2)
    assert v.lookup(("00" * 32,) + CTX[1:], 7, 0, [1, 2, 10, EOS], timeout=0.2) is None
    assert v.lookup(("03" * 32,) + CTX[1:], 7, 0, [1, 2, 10, EOS], timeout=2.0) is not None
    v.close()


# ------------------------------------------------------ réparation : 1er tour
def _eng(monkeypatch, early):
    eng = engine.MiningEngine.__new__(engine.MiningEngine)
    eng._eos_ids = [EOS]
    eng._cached_randomness = CTX[0]
    eng._local_hash = CTX[1]
    eng._loaded_checkpoint_path = CTX[2]
    eng.max_new_tokens = 64
    eng._early_terminal = early
    calls = []

    def fake_verdicts(sock, **kw):
        calls.append([it["rollout"] for it in kw["items"]])
        return [{"ok": True, "pick": EOS, "cdf_miss": 0.0} for _ in kw["items"]]

    monkeypatch.setattr(replica_client, "terminal_verdicts", fake_verdicts)
    monkeypatch.setenv("RELIQUARY_REPLICA_SOCKET", "/tmp/replica.sock")
    eng._vllm_backend = types.SimpleNamespace(
        run_forced_continuations=lambda items, **kw: [[66, EOS] for _ in items])
    return eng, calls


def _gens(n=4):
    return [{"tokens": [1, 2, 10 + r, EOS], "prompt_length": 2} for r in range(n)]


def test_reparation_lit_les_verdicts_precoces(monkeypatch):
    early = EarlyTerminalVerifier(_verify_fn([]), eos_ids=[EOS])
    for r, g in enumerate(_gens()):
        early.submit(CTX, 7, r, g["tokens"], prompt_len=2)
    eng, calls = _eng(monkeypatch, early)
    out = eng._repair_terminal_eos(_gens(), prompt_idx=7, env=None)
    # rollout 2 refusé par le verdict précoce → prolongé puis revérifié seul
    assert calls == [[2]]
    assert out[2]["tokens"] == [1, 2, 12, 55, 66, EOS]
    assert all(g.get("terminal_ok") for g in out)
    early.close()


def test_reparation_complete_les_verdicts_manquants(monkeypatch):
    early = EarlyTerminalVerifier(_verify_fn([]), eos_ids=[EOS])
    early.submit(CTX, 7, 0, [1, 2, 10, EOS], prompt_len=2)     # 1 seul connu
    eng, calls = _eng(monkeypatch, early)
    eng._repair_terminal_eos(_gens(), prompt_idx=7, env=None)
    assert calls[0] == [1, 2, 3]
    early.close()


def test_sans_verificateur_precoce_inchange(monkeypatch):
    eng, calls = _eng(monkeypatch, None)
    eng._repair_terminal_eos(_gens(), prompt_idx=7, env=None)
    assert calls == [[0, 1, 2, 3]]


# ------------------------------------------------------ boucle du bake
def test_stream_appelle_on_rollout_a_chaque_rollout(monkeypatch):
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
    from reliquary.miner.vllm_backend import VLLMBackend

    class _Out:
        def __init__(self, rid, toks):
            self.request_id, self.finished = rid, True
            self.outputs = [types.SimpleNamespace(token_ids=toks, stop_reason=None,
                                                  finish_reason="stop")]

    class _Eng:
        def __init__(self):
            self.plan = [["p0-r1"], ["p0-r0"]]

        def add_request(self, *a):
            pass

        def has_unfinished_requests(self):
            return bool(self.plan)

        def step(self):
            rids = self.plan.pop(0)
            return [_Out(r, [5, EOS]) for r in rids]

    b = VLLMBackend.__new__(VLLMBackend)
    b._llm = types.SimpleNamespace(llm_engine=_Eng())
    b._interrupt = threading.Event()
    b._ensure_loaded = lambda: None
    b._stream_request_id = lambda pos, r: f"p{pos}-r{r}"
    seen = []
    b.generate_forced_phase1_multi_stream(
        [[1, 2]], prompt_indices=[7], randomness="ab", checkpoint_hash="c",
        m_rollouts=2, max_tokens=8, stop_token_ids=[EOS], primary_eos_id=EOS,
        on_group=lambda pos, pidx, g: seen.append(("group", pidx)),
        on_rollout=lambda pos, pidx, r, toks: seen.append(("rollout", pidx, r, list(toks))))
    assert seen == [("rollout", 7, 1, [5, EOS]), ("rollout", 7, 0, [5, EOS]),
                    ("group", 7)]


def test_bake_soumet_les_memes_tokens_que_la_reparation(monkeypatch):
    """Cohérence : le rollout vérifié à la sortie de vLLM a les tokens exacts
    que ``_generate_m_rollouts`` passera à la réparation (prompt encodé +
    complétion coupée au premier EOS)."""
    import asyncio

    submitted = []

    class _Early:
        def submit(self, ctx, prompt_idx, r, tokens, *, prompt_len):
            submitted.append((ctx, prompt_idx, r, list(tokens), prompt_len))

    class _Backend:
        def generate_forced_phase1_multi_stream(
                self, prompts_tokens, *, prompt_indices, randomness,
                checkpoint_hash, m_rollouts, max_tokens, stop_token_ids,
                primary_eos_id, on_group=None, should_abort=None, on_rollout=None):
            for r in range(m_rollouts):
                on_rollout(0, prompt_indices[0], r, [5, EOS, 6, EOS])
            return [[[5, EOS]] * m_rollouts]

    eng = engine.MiningEngine.__new__(engine.MiningEngine)
    eng._vllm_backend = _Backend()
    eng._cached_randomness = CTX[0]
    eng._local_hash = CTX[1]
    eng._loaded_checkpoint_path = CTX[2]
    eng.max_new_tokens = 64
    eng.tokenizer = types.SimpleNamespace()
    eng._eos_ids = [EOS]
    eng._primary_eos_id = lambda: EOS
    eng._early_terminal_verifier = lambda: _Early()
    eng._note_ready = lambda *a: None

    async def _grade(pairs, entries, *, expected_ckpt_n, env):
        return None

    eng._grade_chunk_streaming = _grade
    monkeypatch.setattr(engine, "encode_prompt", lambda tok, p: [1, 2])
    monkeypatch.setattr(engine, "vllm_forced_seed_enabled", lambda: True)
    asyncio.run(eng._bake_stream_fire([{"prompt": "p"}], [7],
                                      expected_ckpt_n=1, env=None))
    assert len(submitted) == engine.M_ROLLOUTS
    ctx, pidx, r, toks, plen = submitted[0]
    assert ctx == CTX and pidx == 7 and plen == 2
    assert toks == [1, 2, 5, EOS]
