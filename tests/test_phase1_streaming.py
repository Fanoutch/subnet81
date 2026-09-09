"""Phase-1 forcée en STREAMING : livrer chaque groupe dès que ses 8 finissent.

Pourquoi (mesuré 2026-08-06) : le départage d'enchère du validateur est
``min(tokens, cap) / (round_arrivée - round_ouverture)`` — chaque seconde entre
l'ouverture de fenêtre et l'arrivée divise le rang. Le peloton soumet ses k=2
en 20-45 s ; nous en 60-110 s, car ``generate`` (appel monolithique) ne rend
RIEN tant que les 128 séquences ne sont pas finies : un k=2 dont les 8 rollouts
sont prêts à ~25 s attend les traînards du plafond (~50 s), puis le grading.
Notre k=2 de la fenêtre 27872 a fini rang 26 pour cette raison exacte.

Le correctif pilote le MÊME moteur sync pas à pas (``add_request``/``step()`` —
c'est littéralement ce que ``generate`` fait en interne) et invoque un callback
par prompt dès que ses ``m_rollouts`` séquences sont terminées, pendant que les
autres continuent à décoder. Mêmes kernels, même processeur forced-seed
engine-level : la parité certifiée par la gate 4B (PASS 0.9793, 2026-08-06)
transfère telle quelle.

Invariants testés ici :
- les requêtes construites sont IDENTIQUES à celles du chemin batché (mêmes
  ``extra_args`` forced-seed, même ordre prompt-major) — un écart = un flux
  forcé différent = SEED_MISMATCH ;
- le callback part à la COMPLÉTION du groupe, pas à la fin du lot ;
- le regroupement rollout->prompt survit à un ordre d'achèvement arbitraire ;
- ``should_abort`` (flip de fenêtre) interrompt et avorte le reste ;
- le retour agrégé reste celui du chemin batché (drop-in).
"""
from __future__ import annotations

import types

import pytest

from reliquary.miner.vllm_backend import VLLMBackend


class _Out:
    """RequestOutput minimal : .finished, .request_id, .outputs[0].token_ids."""

    def __init__(self, request_id, token_ids, finished=True):
        self.request_id = request_id
        self.finished = finished
        inner = types.SimpleNamespace(
            token_ids=list(token_ids), stop_reason=None, finish_reason="stop",
        )
        self.outputs = [inner]


class _FakeEngine:
    """Moteur pas-à-pas scriptable : ``plan`` = liste d'étapes, chaque étape
    étant la liste des request_id qui FINISSENT à ce step."""

    def __init__(self, plan, tokens_by_rid):
        self._plan = list(plan)
        self._tokens = tokens_by_rid
        self.added = []            # (request_id, prompt, params) dans l'ordre
        self.aborted = []

    def add_request(self, request_id, prompt, params):
        self.added.append((request_id, prompt, params))

    def has_unfinished_requests(self):
        return bool(self._plan)

    def step(self):
        if not self._plan:
            return []
        rids = self._plan.pop(0)
        return [_Out(rid, self._tokens[rid]) for rid in rids]

    def abort_request(self, request_ids):
        self.aborted.extend(
            request_ids if isinstance(request_ids, (list, tuple, set))
            else [request_ids]
        )


def _backend_with(engine):
    b = VLLMBackend.__new__(VLLMBackend)
    b._llm = types.SimpleNamespace(llm_engine=engine)
    b._loaded = True
    b._request_counter = 0
    b._ensure_loaded = lambda: None
    return b


def _run(backend, engine, n_prompts=2, m=2, on_group=None, should_abort=None):
    return backend.generate_forced_phase1_multi_stream(
        [[1, 2, 3]] * n_prompts,
        prompt_indices=list(range(100, 100 + n_prompts)),
        randomness="ab" * 32,
        checkpoint_hash="ck",
        m_rollouts=m,
        max_tokens=64,
        stop_token_ids=[9],
        primary_eos_id=9,
        on_group=on_group,
        should_abort=should_abort,
    )


def test_requests_are_built_exactly_like_the_batched_path(monkeypatch):
    """Même extra_args forced-seed, même ordre prompt-major — sinon les flux
    forcés divergent du chemin certifié par la gate."""
    import reliquary.miner.vllm_backend as vb

    captured_batched = []

    class _SP:
        def __init__(self, **kw):
            self.kw = kw

    class _TP:
        def __init__(self, prompt_token_ids):
            self.prompt_token_ids = prompt_token_ids

    fake_vllm = types.SimpleNamespace(SamplingParams=_SP)
    fake_inputs = types.SimpleNamespace(TokensPrompt=_TP)
    monkeypatch.setitem(__import__("sys").modules, "vllm", fake_vllm)
    monkeypatch.setitem(__import__("sys").modules, "vllm.inputs", fake_inputs)

    # plan : tout finit au premier step, tokens finissant par l'EOS 9
    rids_tokens = {}
    engine = _FakeEngine([], rids_tokens)
    b = _backend_with(engine)

    # 2 prompts x 2 rollouts -> 4 requêtes ; on ne fait qu'un dry-run de
    # CONSTRUCTION (plan vide => zéro step), puis on compare aux extra_args
    # qu'aurait produits le chemin batché.
    _run(b, engine, n_prompts=2, m=2)
    assert len(engine.added) == 4
    from reliquary.miner.vllm_forced_seed import (
        FORCED_SEED_EXTRA_KEY, forced_seed_extra_args,
    )
    expect = []
    for pos, pidx in enumerate((100, 101)):
        for r in range(2):
            expect.append(forced_seed_extra_args(
                randomness="ab" * 32, prompt_idx=pidx, checkpoint_hash="ck",
                rollout_index=r, base_offset=0, start_len=3))
    got = [params.kw["extra_args"][FORCED_SEED_EXTRA_KEY]
           for _, _, params in engine.added]
    assert got == expect


def test_group_callback_fires_at_group_completion_not_batch_end(monkeypatch):
    _install_fake_vllm(monkeypatch)
    # 2 prompts x 2 rollouts. Le prompt 1 (rid p1) finit AVANT le prompt 0.
    tokens = {"p0-r0": [5, 9], "p0-r1": [6, 9], "p1-r0": [7, 9], "p1-r1": [8, 9]}
    plan = [["p1-r0"], ["p1-r1"], ["p0-r0"], ["p0-r1"]]
    engine = _FakeEngine(plan, tokens)
    b = _backend_with(engine)
    order = []
    _patch_rids(b)

    _run(b, engine, on_group=lambda pos, pidx, group: order.append((pos, pidx)))
    # prompt 1 complet au step 2, prompt 0 au step 4 : le callback doit suivre
    # l'ordre de COMPLÉTION, pas l'ordre d'entrée.
    assert order == [(1, 101), (0, 100)]


def test_regrouping_survives_interleaved_completion(monkeypatch):
    _install_fake_vllm(monkeypatch)
    tokens = {"p0-r0": [10, 9], "p0-r1": [11, 9], "p1-r0": [20, 9], "p1-r1": [21, 9]}
    plan = [["p1-r1", "p0-r0"], ["p0-r1", "p1-r0"]]
    engine = _FakeEngine(plan, tokens)
    b = _backend_with(engine)
    _patch_rids(b)
    got = _run(b, engine)
    # retour agrégé = contrat du chemin batché, ordre d'entrée, rollouts triés
    assert got == [[[10, 9], [11, 9]], [[20, 9], [21, 9]]]


def test_abort_stops_the_stream_and_aborts_remaining(monkeypatch):
    """Flip de fenêtre : les groupes restants seraient jetés hors-tranche de
    toute façon — les avorter libère le GPU pour la nouvelle tranche."""
    _install_fake_vllm(monkeypatch)
    tokens = {"p0-r0": [5, 9], "p0-r1": [6, 9], "p1-r0": [7, 9], "p1-r1": [8, 9]}
    plan = [["p0-r0"], ["p0-r1"], ["p1-r0"], ["p1-r1"]]
    engine = _FakeEngine(plan, tokens)
    b = _backend_with(engine)
    _patch_rids(b)
    steps = {"n": 0}

    def abort_after_two():
        steps["n"] += 1
        return steps["n"] > 2

    done = []
    _run(b, engine, on_group=lambda pos, pidx, g: done.append(pos),
         should_abort=abort_after_two)
    assert done == [0], "le groupe complet avant l'abandon doit être livré"
    assert set(engine.aborted) == {"p1-r0", "p1-r1"}, (
        "les requêtes du groupe incomplet doivent être avortées"
    )


def _install_fake_vllm(monkeypatch):
    class _SP:
        def __init__(self, **kw):
            self.kw = kw

    class _TP:
        def __init__(self, prompt_token_ids):
            self.prompt_token_ids = prompt_token_ids

    monkeypatch.setitem(
        __import__("sys").modules, "vllm",
        types.SimpleNamespace(SamplingParams=_SP))
    monkeypatch.setitem(
        __import__("sys").modules, "vllm.inputs",
        types.SimpleNamespace(TokensPrompt=_TP))


def _patch_rids(backend):
    """Force des request_id prévisibles p{pos}-r{r} pour scripter le plan."""
    backend._stream_request_id = lambda pos, r: f"p{pos}-r{r}"
