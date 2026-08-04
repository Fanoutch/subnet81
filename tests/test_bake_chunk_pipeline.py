"""_bake_streaming en chunks double-bufferés : le prefetch GPU du chunk N+1
doit recouvrir le grading CPU du chunk N.

Mesuré 2026-08-03 (H100, 4B, cap 2600) : GPU actif ~50% seulement — un seul
prefetch batché au début du bake de 40, puis la randomness flippe, le cache
phase-1 devient stale et chaque prompt régénère en appel 8-séquences avec des
trous CPU entre les appels. Le pipeline chunké répare les deux : appels vLLM
amortis (chunk×8 séquences) TOUJOURS sous la randomness courante, et le GPU
préfetche pendant que le CPU grade.

Sûreté : mêmes fonctions (generate_forced_phase1_multi + _pre_bake_entry),
mêmes tokens, mêmes preuves — seul l'ordonnancement change.
"""
import asyncio
import time

import pytest

from reliquary.miner.engine import MiningEngine


class _Env:
    name = "opencodeinstruct"


def _mk_engine(events, *, prefetch_s=0.08, grade_s=0.08):
    """MiningEngine squelette : seuls les attributs consommés par
    _bake_streaming existent ; prefetch/grade sont instrumentés."""
    e = MiningEngine.__new__(MiningEngine)
    e._pool = []
    e._pool_lock = asyncio.Lock()
    e._phase1_cache = {}
    e._cached_randomness = "aa" * 32
    e._local_n = 7

    def prefetch(problems, prompt_indices, *, randomness, env):
        events.append(("prefetch_start", tuple(prompt_indices), time.monotonic()))
        time.sleep(prefetch_s)
        events.append(("prefetch_end", tuple(prompt_indices), time.monotonic()))
        return len(prompt_indices)

    def pre_bake(idx, problem, expected_ckpt_n, env):
        events.append(("grade_start", idx, time.monotonic()))
        time.sleep(grade_s)
        events.append(("grade_end", idx, time.monotonic()))
        return {"prompt_idx": idx, "checkpoint_n": expected_ckpt_n}

    e._prefetch_phase1 = prefetch
    e._pre_bake_entry = pre_bake
    return e


def _run(engine, n, monkeypatch, chunk):
    monkeypatch.setenv("RELIQUARY_BAKE_CHUNK", str(chunk))
    problems = [{"prompt": f"p{i}"} for i in range(n)]
    return asyncio.run(engine._bake_streaming(
        problems, list(range(n)), expected_ckpt_n=7, env=_Env(),
    ))


def test_chunked_prefetch_covers_all_prompts(monkeypatch):
    events = []
    entries = _run(_mk_engine(events), 6, monkeypatch, chunk=2)
    prefetched = [i for ev, idxs, _ in events if ev == "prefetch_start"
                  for i in idxs]
    assert sorted(prefetched) == list(range(6)), "chaque prompt préfetché une fois"
    assert len([e for e in events if e[0] == "prefetch_start"]) == 3, \
        "6 prompts / chunk 2 = 3 appels préfetch batchés"
    assert [e["prompt_idx"] for e in entries] == list(range(6))


def test_prefetch_next_chunk_overlaps_grading(monkeypatch):
    events = []
    _run(_mk_engine(events), 6, monkeypatch, chunk=2)
    # Le prefetch du chunk suivant doit DÉMARRER avant la fin du grading du
    # chunk courant (recouvrement GPU/CPU) — sinon on a le comportement
    # d'aujourd'hui (séquentiel, GPU idle pendant le grade).
    t_pf2_start = next(t for ev, idxs, t in events
                       if ev == "prefetch_start" and idxs == (2, 3))
    t_grade1_end = next(t for ev, i, t in events
                        if ev == "grade_end" and i == 1)
    assert t_pf2_start < t_grade1_end, (
        "prefetch(chunk 2) doit tourner PENDANT le grading du chunk 1 "
        f"(start={t_pf2_start:.3f} vs grade_end={t_grade1_end:.3f})"
    )


def test_entries_land_in_pool_incrementally(monkeypatch):
    events = []
    engine = _mk_engine(events)
    entries = _run(engine, 4, monkeypatch, chunk=2)
    assert len(engine._pool) == 4 and len(entries) == 4
    assert engine._phase1_cache == {}, "cache purgé en fin de bake"


def test_chunk_default_matches_full_batch_when_large(monkeypatch):
    """chunk >= n prompts → un seul préfetch (comportement historique)."""
    events = []
    _run(_mk_engine(events), 4, monkeypatch, chunk=64)
    assert len([e for e in events if e[0] == "prefetch_start"]) == 1
