"""Restart B (17/09) : détection du flip par ``/miner-state`` pendant le trou 503.

flip_diag (restart A, 11 fen) : dans 4 fenêtres sur 11 le GET ``/state`` prend
2,8-3,5 s (cache validateur froid) et le poll du cooldown 1,5 s, soit un signal
de flip à ~5 s au lieu de ~1,5 s. ``/miner-state`` (5 ko) porte randomness,
fenêtre, checkpoint, ouverture et cooldown de la tranche.
"""
from __future__ import annotations

import asyncio
import base64
import inspect
from types import SimpleNamespace

import pytest

from reliquary.miner import engine

CODE = "opencodeinstruct"


def _encode(prompts, rng):
    start, end = rng
    bm = bytearray((end - start + 7) // 8)
    for p in prompts:
        o = p - start
        if 0 <= o < end - start:
            bm[o // 8] |= 1 << (o % 8)
    return base64.b64encode(bm).decode()


def _ms(**kw):
    rng = kw.pop("rng", (1000, 6000))
    cd = kw.pop("cd", {1000, 1007, 1008, 5999})
    base = {
        "schema_version": 1, "state": "open", "window_n": 46200, "anchor_block": 46200,
        "window_opened_at": 1789643748.85, "checkpoint_n": 2360,
        "checkpoint_repo_id": "ReliquaryForge/qwen3-4b-base-dapo-v4",
        "checkpoint_revision": "31557b7f", "randomness": "aa" * 32,
        "environments": {CODE: {"encoding": "bitset-v1", "prompt_range": list(rng),
                                "cooldown_bitmap": _encode(cd, rng),
                                "cooldown_count": len(cd),
                                "accepting_submissions": True, "admission_remaining": 10}},
    }
    base.update(kw)
    return base


def test_decode_bitmap_aller_retour_et_bits_hors_tranche():
    rng = (1000, 1013)
    cd = {1000, 1005, 1012}
    assert engine.decode_cooldown_bitmap(_encode(cd, rng), rng) == cd
    bad = bytearray(base64.b64decode(_encode(cd, rng)))
    bad[-1] |= 0x80                       # bit au-delà de la tranche
    with pytest.raises(ValueError):
        engine.decode_cooldown_bitmap(base64.b64encode(bad).decode(), rng)
    with pytest.raises(ValueError):
        engine.decode_cooldown_bitmap(_encode(cd, (1000, 1100)), rng)


def test_vue_complete():
    v = engine.miner_state_flip_view(_ms(), [CODE])
    assert v.window_n == 46200 and v.randomness == "aa" * 32
    assert v.cooldowns[CODE] == {1000, 1007, 1008, 5999}
    assert v.checkpoint_n == 2360 and v.window_opened_at == 1789643748.85


@pytest.mark.parametrize("mut", [
    {"state": "closed"}, {"randomness": ""}, {"checkpoint_revision": None},
    {"checkpoint_n": None}, {"environments": {}},
])
def test_vue_refusee_si_incomplete(mut):
    assert engine.miner_state_flip_view(_ms(**mut), [CODE]) is None


def test_vue_refusee_si_compte_incoherent():
    ms = _ms()
    ms["environments"][CODE]["cooldown_count"] = 99
    assert engine.miner_state_flip_view(ms, [CODE]) is None


def test_flag_off_par_defaut(monkeypatch):
    monkeypatch.delenv("RELIQUARY_MINER_STATE_FLIP", raising=False)
    assert engine.miner_state_flip_enabled() is False
    monkeypatch.setenv("RELIQUARY_MINER_STATE_FLIP", "1")
    assert engine.miner_state_flip_enabled() is True


def _eng():
    eng = engine.MiningEngine.__new__(engine.MiningEngine)
    eng._pool_lock = asyncio.Lock()
    eng._pool = [object(), object()]
    eng._retry_by_env = {CODE: [1, 2]}
    eng._cooldowns = {CODE: {1}}
    eng._cached_randomness = "bb" * 32
    eng._submitted_this_window = {5}
    eng._active_prompt_range = lambda w, r: (0, 1)
    calls = []

    async def pull(state):
        calls.append(("pull", state.checkpoint_revision))
        return False

    eng._apply_checkpoint_pull = pull
    eng._signal_flip = lambda: calls.append(("signal",))
    return eng, calls


def test_flip_applique_tout_ce_que_le_bake_lit(monkeypatch):
    monkeypatch.setattr(engine, "study_dump", lambda *a, **k: None)
    eng, calls = _eng()
    v = engine.miner_state_flip_view(_ms(), [CODE])
    ok = asyncio.run(eng._flip_from_miner_state(v, t_send=1.0, t_recv=2.0))
    assert ok
    assert eng._pool == [] and eng._retry_by_env[CODE] == []
    assert eng._submitted_this_window == set()
    assert eng._cooldowns[CODE] == {1000, 1007, 1008, 5999}
    assert eng._cached_randomness == "aa" * 32 and eng._cached_window_n == 46200
    assert eng._window_open_ts == 1789643748.85
    assert calls == [("pull", "31557b7f"), ("signal",)]     # réveil APRÈS le pull


def test_flip_ignore_si_randomness_deja_connue(monkeypatch):
    eng, calls = _eng()
    eng._cached_randomness = "aa" * 32
    v = engine.miner_state_flip_view(_ms(), [CODE])
    assert asyncio.run(eng._flip_from_miner_state(v, t_send=1.0, t_recv=2.0)) is False
    assert calls == [] and len(eng._pool) == 2
    assert asyncio.run(eng._flip_from_miner_state(None, t_send=1.0, t_recv=2.0)) is False


def test_boucle_sonde_miner_state_seulement_pendant_le_trou():
    src = inspect.getsource(engine.MiningEngine._trigger_loop)
    i = src.index("miner_state_flip_enabled()")
    j = src.index("get_window_state_v2_with_resp(")
    assert i < j                                   # avant le GET /state lourd
    assert '_state_gap_since", None) is not None' in src[i - 10:i + 200]
