"""Câblage engine du streaming par groupe (_bake_stream_fire).

Invariants :
- chaque groupe part en grade/preuve à SA complétion (ordre d'achèvement,
  pas ordre d'entrée) ;
- le groupe est servi au chemin per-prompt historique via le cache phase-1
  (mêmes fonctions, mêmes preuves — seul l'ordonnancement change) ;
- backend sans support stream -> None (le caller retombe sur le chunké) ;
- flag RELIQUARY_STREAM_FIRE=0 -> chemin historique inchangé ;
- le cache phase-1 est purgé en fin de bake (rien ne survit à la randomness
  suivante).
"""
from __future__ import annotations

import asyncio
import types

import pytest

from reliquary.miner.engine import MiningEngine, stream_fire_enabled


class _StreamBackend:
    """Backend factice : livre les groupes dans un ordre scripté."""

    def __init__(self, order):
        self._order = order          # liste de positions, ordre de complétion

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


def _engine(backend, graded):
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
            # prouve que la complétion streamée est bien dans le cache au
            # moment du grading (c'est elle que _generate_m_rollouts pop())
            key = (prompt_idx, e._cached_randomness, e._local_hash)
            assert key in e._phase1_cache, "groupe absent du cache au grading"
            graded.append(prompt_idx)
            entries.append({"prompt_idx": prompt_idx})

    e._grade_chunk_streaming = _fake_grade
    return e


def _problems(n):
    return [{"prompt": f"p{i}"} for i in range(n)]


def test_groups_are_graded_in_completion_order(monkeypatch):
    monkeypatch.setattr(
        "reliquary.miner.engine.encode_prompt", lambda tok, p: [1, 2, 3])
    monkeypatch.setattr("reliquary.constants.FORCED_SEED_ENFORCE", True)
    monkeypatch.setenv("RELIQUARY_VLLM_FORCED_SEED", "1")
    graded = []
    e = _engine(_StreamBackend(order=[2, 0, 1]), graded)
    entries = asyncio.run(e._bake_stream_fire(
        _problems(3), [100, 101, 102], expected_ckpt_n=1, env=None))
    assert graded == [102, 100, 101], "le grading doit suivre la complétion"
    assert [x["prompt_idx"] for x in entries] == [102, 100, 101]
    assert e._phase1_cache == {}, "le cache doit être purgé en fin de bake"


def test_backend_without_stream_support_returns_none(monkeypatch):
    monkeypatch.setattr("reliquary.constants.FORCED_SEED_ENFORCE", True)
    monkeypatch.setenv("RELIQUARY_VLLM_FORCED_SEED", "1")
    e = _engine(types.SimpleNamespace(), [])   # pas de méthode stream
    e._vllm_backend = types.SimpleNamespace()
    got = asyncio.run(e._bake_stream_fire(
        _problems(2), [1, 2], expected_ckpt_n=1, env=None))
    assert got is None, "sans support stream, le caller doit retomber en chunké"


def test_flag_off_disables_streaming(monkeypatch):
    monkeypatch.setenv("RELIQUARY_STREAM_FIRE", "0")
    assert stream_fire_enabled() is False
    monkeypatch.delenv("RELIQUARY_STREAM_FIRE", raising=False)
    assert stream_fire_enabled() is True, "défaut = ON (mesuré, cf. docstring)"


def test_driver_failure_does_not_hang_or_raise(monkeypatch):
    """Un backend qui explose au milieu ne doit ni bloquer ni tuer le bake."""
    monkeypatch.setattr(
        "reliquary.miner.engine.encode_prompt", lambda tok, p: [1, 2, 3])
    monkeypatch.setattr("reliquary.constants.FORCED_SEED_ENFORCE", True)
    monkeypatch.setenv("RELIQUARY_VLLM_FORCED_SEED", "1")

    class _Boom:
        def generate_forced_phase1_multi_stream(self, *a, **kw):
            kw["on_group"](0, 100, [[7, 9]])
            raise RuntimeError("moteur mort")

    graded = []
    e = _engine(_Boom(), graded)
    entries = asyncio.run(asyncio.wait_for(e._bake_stream_fire(
        _problems(1), [100], expected_ckpt_n=1, env=None), timeout=10))
    assert graded == [100], "le groupe livré avant le crash doit être gradé"
    assert [x["prompt_idx"] for x in entries] == [100]
