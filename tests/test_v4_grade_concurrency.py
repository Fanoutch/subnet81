"""Fix 18/08 : grading concurrent entre groupes + fenêtre scellée.

Contrefactuel mesuré : ~5 slots/fenêtre perdus post-seal (seal à 10-40 s),
la queue venait du grading séquentiel groupe par groupe.
"""
import asyncio
import time
import types

import pytest

from reliquary.miner.engine import MiningEngine


def _engine():
    e = object.__new__(MiningEngine)
    e._pool = []
    e._pool_lock = asyncio.Lock()
    e._local_n = 0
    return e


def test_groups_grade_concurrently(monkeypatch):
    e = _engine()
    calls = []

    def slow_entry(prompt_idx, problem, ckpt, env):
        time.sleep(0.25)
        calls.append(prompt_idx)
        return None  # court-circuite pool/tranche

    e._pre_bake_entry = slow_entry
    pairs = [(i, {}) for i in range(4)]
    t0 = time.monotonic()
    asyncio.run(e._grade_chunk_streaming(pairs, [], expected_ckpt_n=0, env=None))
    wall = time.monotonic() - t0
    assert sorted(calls) == [0, 1, 2, 3]
    # séquentiel = 1.0 s ; concurrence 3 → ~0.5 s
    assert wall < 0.85, f"pas concurrent: {wall:.2f}s"


def test_concurrency_env_bound(monkeypatch):
    monkeypatch.setenv("RELIQUARY_GRADE_CONCURRENCY", "1")
    e = _engine()

    def slow_entry(prompt_idx, problem, ckpt, env):
        time.sleep(0.15)
        return None

    e._pre_bake_entry = slow_entry
    t0 = time.monotonic()
    asyncio.run(e._grade_chunk_streaming([(i, {}) for i in range(3)], [],
                                         expected_ckpt_n=0, env=None))
    assert time.monotonic() - t0 >= 0.42  # borné à 1 = séquentiel
    monkeypatch.delenv("RELIQUARY_GRADE_CONCURRENCY")


def test_entry_reaches_pool(monkeypatch):
    e = _engine()
    entry = {"checkpoint_n": 0}
    e._pre_bake_entry = lambda *a: dict(entry)
    entries = []
    asyncio.run(e._grade_chunk_streaming([(7, {})], entries,
                                         expected_ckpt_n=0, env=None))
    assert len(entries) == 1 and len(e._pool) == 1


def test_sealed_window_defers_fire():
    e = object.__new__(MiningEngine)
    e._sealed_window = 123
    state = types.SimpleNamespace(window_n=123)
    # _fire_for_window return immédiat (aucun attribut requis au-delà)
    asyncio.run(e._fire_for_window(state, "http://x", None, []))
