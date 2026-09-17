"""Restart A (17/09) : correctifs de temps perdu + mesures pures.

Audit multi-agents du 17/09 :
- le réveil au flip (16/09) signalait AVANT le pull du checkpoint ; le
  générateur trouvait des poids préchargés en avance sur le hash, consommait
  l'événement et redormait 1 s pleine ;
- ``_hf_download`` attendait la purge du cache HF (~8 Go) sur ce même chemin ;
- les réessais ``batch_filled`` partent toutes les ~6 s et continuent après
  ~75 s, alors que le 112e payé code arrive à 71,5 s p50 → A/B par parité ;
- mesures : chronologie du flip, fin de preuve / fin EOS séparées, timeouts de
  grading, attente des notations entre bakes, négatifs hors zone.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import threading
import time as _time
import types
from types import SimpleNamespace

from reliquary.miner import engine

CODE = "opencodeinstruct"


def _fc(admitted_code=60, open_ago=20.0):
    now = _time.time()
    return SimpleNamespace(
        phase="collecting", precommit_cutoff_ts=now - open_ago + 1407.0,
        precommit_seconds=1407.0, picks_target=7,
        admission_budgets={CODE: 224}, admitted={CODE: admitted_code},
        proven={CODE: 0}, picks_by_environment={CODE: 0},
        remaining={CODE: 224 - admitted_code},
    )


def _st(window_n, open_ago=20.0):
    return SimpleNamespace(fill_closed=_fc(open_ago=open_ago), window_n=window_n,
                           state="open", randomness="ab" * 16, cooldown_prompts=[])


# ── A/B des réessais ─────────────────────────────────────────────────────────
def test_retry_arm_inactif_par_defaut(monkeypatch):
    monkeypatch.delenv("RELIQUARY_RETRY_AB", raising=False)
    assert engine.retry_arm(46181) is None


def test_retry_arm_parite(monkeypatch):
    monkeypatch.setenv("RELIQUARY_RETRY_AB", "1")
    assert engine.retry_arm(46181) == "rapide"
    assert engine.retry_arm(46182) == "temoin"
    assert engine.retry_arm(None) is None


def test_bras_temoin_identique_a_l_historique(monkeypatch):
    monkeypatch.delenv("RELIQUARY_BATCH_FILLED_MAX_RETRIES", raising=False)
    now = _time.time()
    monkeypatch.setenv("RELIQUARY_RETRY_AB", "1")
    ab = engine.fire_retry_decision(_st(46182), "batch_filled", 3, CODE, now=now)
    monkeypatch.setenv("RELIQUARY_RETRY_AB", "0")
    ref = engine.fire_retry_decision(_st(46182), "batch_filled", 3, CODE, now=now)
    assert ab == ref


def test_bras_rapide_pause_courte(monkeypatch):
    monkeypatch.setenv("RELIQUARY_RETRY_AB", "1")
    now = _time.time()
    ok, nb = engine.fire_retry_decision(_st(46181), "batch_filled", 6, CODE, now=now)
    assert ok is True and 0 < nb - now <= 2.0 + 1e-9
    _, nb_ref = engine.fire_retry_decision(_st(46182), "batch_filled", 6, CODE, now=now)
    assert nb_ref - now > nb - now


def test_bras_rapide_arret_apres_stop(monkeypatch):
    monkeypatch.setenv("RELIQUARY_RETRY_AB", "1")
    now = _time.time()
    assert engine.fire_retry_decision(_st(46181, open_ago=95.0), "batch_filled", 2,
                                      CODE, now=now) == (False, None)
    assert engine.fire_retry_decision(_st(46181, open_ago=60.0), "batch_filled", 2,
                                      CODE, now=now)[0] is True


def test_bras_rapide_plafond(monkeypatch):
    monkeypatch.setenv("RELIQUARY_RETRY_AB", "1")
    monkeypatch.setenv("RELIQUARY_RETRY_FAST_MAX_RETRIES", "5")
    now = _time.time()
    assert engine.fire_retry_decision(_st(46181), "batch_filled", 4, CODE, now=now)[0]
    assert not engine.fire_retry_decision(_st(46181), "batch_filled", 5, CODE, now=now)[0]


def test_bras_rapide_motifs_non_batch_filled_inchanges(monkeypatch):
    monkeypatch.setenv("RELIQUARY_RETRY_AB", "1")
    now = _time.time()
    assert engine.fire_retry_decision(_st(46181), "stale_round", 2, CODE, now=now) == (False, None)
    assert engine.fire_retry_decision(_st(46181), "out_of_zone", 1, CODE, now=now) == (False, None)


# ── réveil APRÈS le pull ─────────────────────────────────────────────────────
def test_reveil_apres_le_pull_et_avant_le_saut_de_tir():
    src = inspect.getsource(engine.MiningEngine._trigger_loop)
    i_pull = src.index("await self._apply_checkpoint_pull(state)")
    i_sig = src.index("self._signal_flip()")
    i_skip = src.index("if ckpt_advanced_this_iter:")
    assert i_pull < i_sig < i_skip
    assert src.count("self._signal_flip()") == 1


def test_hf_download_n_attend_pas_la_purge(monkeypatch):
    import huggingface_hub

    done = threading.Event()
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        lambda **kw: "/tmp/snap")

    def slow_prune(repo, rev):
        _time.sleep(0.6)
        done.set()
    monkeypatch.setattr(engine, "_prune_hf_revisions", slow_prune)

    async def go():
        t0 = _time.monotonic()
        path = await engine._hf_download("r/r", "abc")
        dt = _time.monotonic() - t0
        await asyncio.sleep(0.9)                      # la purge finit en fond
        return path, dt
    path, dt = asyncio.new_event_loop().run_until_complete(go())
    assert path == "/tmp/snap" and dt < 0.3 and done.is_set()


# ── mesures pures ────────────────────────────────────────────────────────────
def test_flip_diag_log(tmp_path, monkeypatch):
    out = tmp_path / "flip.jsonl"
    monkeypatch.setenv("RELIQUARY_FLIP_DUMP", str(out))
    row = engine.flip_diag_log(window_n=7, t_send=100.0, t_recv=100.4,
                               cooldown_s=0.7, pull_s=0.02, t_signal=101.5,
                               open_ts=99.0, ckpt_advanced=True)
    assert row["get_state_s"] == 0.4 and row["recv_off"] == 1.4
    assert row["signal_off"] == 2.5 and row["ckpt_advanced"] is True
    assert json.loads(out.read_text().splitlines()[0])["window_n"] == 7


def test_flip_diag_sans_ouverture_connue():
    row = engine.flip_diag_log(window_n=7, t_send=None, t_recv=None, cooldown_s=0,
                               pull_s=0, t_signal=5.0, open_ts=None)
    assert row["recv_off"] is None and row["get_state_s"] is None


EOS = 99


def _proof_eng():
    eng = engine.MiningEngine.__new__(engine.MiningEngine)
    eng._eos_ids = [EOS]
    eng._cached_randomness = "ab" * 32
    eng._local_hash = "c" * 40
    eng.tokenizer = types.SimpleNamespace(decode=lambda toks: "t")

    def repair(gens, prompt_idx, env=None):
        _time.sleep(0.2)
        out = [dict(g) for g in gens]
        for g in out:
            g["terminal_ok"] = True
        return out
    eng._repair_terminal_eos = repair
    eng._proof_rollouts = lambda gens, texts=None, *, device=None, terminal_ctx=None: [
        {"all_tokens": list(g["tokens"]), "local_screen": None} for g in gens]
    return eng


def test_preuve_et_eos_horodatees_separement():
    eng = _proof_eng()
    gens = [{"tokens": [1, 2, 10 + r, EOS], "prompt_length": 2} for r in range(4)]
    tl = {}
    out, cache = eng._repair_with_early_proof(gens, ["a"] * 4, prompt_idx=7,
                                              env=None, timing=tl)
    assert cache is not None
    assert "t_eos_end" in tl and "t_proof_only_end" in tl
    assert tl["t_proof_only_end"] <= tl["t_eos_end"]


def test_preuve_sans_timing_inchangee():
    eng = _proof_eng()
    gens = [{"tokens": [1, 2, 10 + r, EOS], "prompt_length": 2} for r in range(4)]
    out, cache = eng._repair_with_early_proof(gens, ["a"] * 4, prompt_idx=7, env=None)
    assert cache is not None


def test_hors_zone_ecrit_dans_un_dump_separe(tmp_path, monkeypatch):
    from reliquary.miner import replica_client  # noqa: F401

    ooz = tmp_path / "ooz.jsonl"
    samples = []
    monkeypatch.setenv("RELIQUARY_OOZ_DUMP", str(ooz))
    monkeypatch.setenv("RELIQUARY_REPLICA_SOCKET", "/tmp/replica.sock")
    monkeypatch.setenv("RELIQUARY_MIN_ROLLOUT_LEN", "0")
    eng = _proof_eng()
    eng._cached_window_n = 46181
    gens = [{"tokens": [1, 2] + [5] * (3 + r) + [EOS], "prompt_length": 2}
            for r in range(engine.M_ROLLOUTS)]
    eng._generate_m_rollouts = lambda problem, rand, env, prompt_idx: gens
    monkeypatch.setattr(engine, "grade_group_parallel_ex",
                        lambda env, pairs, max_workers=8: (
                            [1.0] * len(pairs), [False] * (len(pairs) - 1) + [True]))
    monkeypatch.setattr(engine, "dump_group_sample", lambda **k: samples.append(k))
    eng._record_drop = lambda dropped, reason=None: None
    eng._sz_note = lambda *a, **k: None
    r = eng._pre_bake_entry(7, {"prompt": "p"}, 1,
                            env=types.SimpleNamespace(name=CODE))
    assert r is None and samples == []
    row = json.loads(ooz.read_text().splitlines()[0])
    assert row["prompt_idx"] == 7 and row["window_n"] == 46181
    assert row["prompt"] == "p" and sum(row["timeouts"]) == 1
    assert len(row["completion_lens"]) == engine.M_ROLLOUTS
