"""Voie prioritaire FIFO de la tête de bake (RELIQUARY_HEAD_FIFO).

Constat du 28/08 (fenêtre 35144) : g1 prêt à 6,1 s et g2 à 6,3 s partent en
grade CONCURRENT ; l'ordre d'envoi devient l'ordre de fin de grade — g2 a
doublé g1, et g1 (prêt le premier) n'est parti qu'à ~17 s, rang 21+. La tête
doit traverser grade→pool→tir dans l'ordre de livraison, un groupe à la fois.

Invariants :
- HEAD_FIFO=K : le grade du (n≤K)-ième groupe livré ne DÉMARRE qu'après la fin
  du grade du précédent (série stricte, ordre de livraison) ;
- au-delà de K : chemin concurrent historique inchangé ;
- HEAD_FIFO absent/0 : comportement actuel byte-identique (aucune attente) ;
- un grade de tête qui dépasse le garde-fou ne bloque pas la file : la boucle
  repasse en concurrent et le bake continue.
"""
from __future__ import annotations

import asyncio
import types

import pytest

from reliquary.miner.engine import MiningEngine


class _StreamBackend:
    def __init__(self, order):
        self._order = order

    def generate_forced_phase1_multi_stream(
        self, prompts_tokens, *, prompt_indices, randomness, checkpoint_hash,
        m_rollouts, max_tokens, stop_token_ids, primary_eos_id,
        on_group=None, should_abort=None,
    ):
        groups = [[[7, 9]] * m_rollouts for _ in prompt_indices]
        for pos in self._order:
            if should_abort is not None and should_abort():
                break
            if on_group is not None:
                on_group(pos, prompt_indices[pos], groups[pos])
        return groups


def _engine(backend, events, grade_s=0.05):
    e = MiningEngine.__new__(MiningEngine)
    e._vllm_backend = backend
    e._cached_randomness = "ab" * 32
    e._local_hash = "ck"
    e.max_new_tokens = 64
    e.tokenizer = types.SimpleNamespace()
    e._eos_ids = [9]
    e._primary_eos_id = lambda: 9

    async def _fake_grade(chunk_pairs, entries, *, expected_ckpt_n, env):
        for prompt_idx, _problem in chunk_pairs:
            events.append(("start", prompt_idx))
            await asyncio.sleep(grade_s)
            events.append(("end", prompt_idx))
            entries.append({"prompt_idx": prompt_idx})

    e._grade_chunk_streaming = _fake_grade
    return e


def _run(engine, n=4):
    problems = [{"prompt": f"p{i}"} for i in range(n)]
    idx = list(range(100, 100 + n))
    import reliquary.miner.engine as eng

    orig = eng.encode_prompt
    eng.encode_prompt = lambda tok, p: [1, 2]
    try:
        return asyncio.run(engine._bake_stream_fire(
            problems, idx, expected_ckpt_n=1, env=None,
        ))
    finally:
        eng.encode_prompt = orig


def _enable_stream(monkeypatch):
    monkeypatch.setattr("reliquary.constants.FORCED_SEED_ENFORCE", True)
    monkeypatch.setenv("RELIQUARY_VLLM_FORCED_SEED", "1")


def test_head_fifo_serialise_la_tete(monkeypatch):
    _enable_stream(monkeypatch)
    monkeypatch.setenv("RELIQUARY_HEAD_FIFO", "2")
    events: list = []
    backend = _StreamBackend([0, 1, 2, 3])
    entries = _run(_engine(backend, events))
    assert len(entries) == 4
    # tête : le grade de 101 ne démarre qu'après la FIN de celui de 100
    assert events.index(("end", 100)) < events.index(("start", 101))
    # et l'ordre de tête suit l'ordre de livraison
    assert events.index(("start", 100)) < events.index(("start", 101))


def test_head_fifo_ne_serialise_pas_le_balayage(monkeypatch):
    _enable_stream(monkeypatch)
    monkeypatch.setenv("RELIQUARY_HEAD_FIFO", "2")
    events: list = []
    backend = _StreamBackend([0, 1, 2, 3])
    _run(_engine(backend, events, grade_s=0.15))
    # balayage (3e/4e livrés) : démarrés sans attendre la fin l'un de l'autre
    s2, s3 = events.index(("start", 102)), events.index(("start", 103))
    e2 = events.index(("end", 102))
    assert s3 < e2, "le balayage doit rester concurrent"


def test_defaut_zero_comportement_actuel(monkeypatch):
    _enable_stream(monkeypatch)
    monkeypatch.delenv("RELIQUARY_HEAD_FIFO", raising=False)
    events: list = []
    backend = _StreamBackend([0, 1, 2, 3])
    _run(_engine(backend, events, grade_s=0.15))
    # aucun sérialisme imposé : le grade de 101 démarre AVANT la fin de 100
    assert events.index(("start", 101)) < events.index(("end", 100))


def test_garde_fou_grade_lent(monkeypatch):
    _enable_stream(monkeypatch)
    monkeypatch.setenv("RELIQUARY_HEAD_FIFO", "2")
    monkeypatch.setenv("RELIQUARY_HEAD_FIFO_WAIT_S", "0.1")
    events: list = []
    backend = _StreamBackend([0, 1, 2, 3])
    entries = _run(_engine(backend, events, grade_s=0.4))
    # le grade de tête dépasse le garde-fou -> la boucle continue quand même
    # et TOUS les groupes finissent gradés (le gather final les attend)
    assert len(entries) == 4
