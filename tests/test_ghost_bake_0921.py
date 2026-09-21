"""21/09 : BAKES FANTÔMES — étiqueter des prompts pendant le temps mort GPU.

Spec : docs/superpowers/specs/2026-09-21-bakes-fantomes-design.md.
Sous V1 le cycle dure ~687 s et le mineur ne produit que ~263 s : le GPU dort
pendant le trou 503. On y génère et note des prompts de la bande p95-p99 du
prior, avec une randomness SYNTHÉTIQUE, pour fabriquer la ressource rare —
« ce prompt a été mesuré récemment, et avec quel verdict ».

Ce que ces tests verrouillent, par ordre d'importance :
1. SÉCURITÉ : aucun lot hors de la fenêtre de tir (330-537 s après l'ouverture,
   cycle jamais < 627 s sur 349 fenêtres), aucun hors du trou 503, et
   interruption au flip (le moteur vérifie ``should_abort`` à chaque pas).
2. ISOLEMENT : un groupe fantôme n'entre JAMAIS dans le pool d'envoi ni dans le
   cache phase-1, et ne touche pas la randomness courante.
3. ÉTAGE A : avec ``GHOST_FEED=0`` rien n'alimente le mémo ni la liste noire.
"""
from __future__ import annotations

import asyncio
import inspect
import random
import types

import numpy as np

from reliquary.miner import engine
from reliquary.miner import ghost_bake as gb

REAL_RND = "ab" * 32


# ───────────────────────────── drapeaux et bornes ────────────────────────────
def test_desactive_par_defaut(monkeypatch):
    monkeypatch.delenv("RELIQUARY_GHOST_BAKE", raising=False)
    monkeypatch.delenv("RELIQUARY_GHOST_FEED", raising=False)
    assert gb.ghost_enabled() is False
    assert gb.ghost_feed_enabled() is False


def test_activation_par_variables(monkeypatch):
    monkeypatch.setenv("RELIQUARY_GHOST_BAKE", "1")
    monkeypatch.setenv("RELIQUARY_GHOST_FEED", "1")
    assert gb.ghost_enabled() and gb.ghost_feed_enabled()


def test_bornes_par_defaut_et_surcharge(monkeypatch):
    monkeypatch.delenv("RELIQUARY_GHOST_T_MIN", raising=False)
    monkeypatch.delenv("RELIQUARY_GHOST_T_MAX", raising=False)
    assert gb.ghost_bounds() == (330.0, 537.0)
    monkeypatch.setenv("RELIQUARY_GHOST_T_MIN", "300")
    monkeypatch.setenv("RELIQUARY_GHOST_T_MAX", "500")
    assert gb.ghost_bounds() == (300.0, 500.0)
    monkeypatch.setenv("RELIQUARY_GHOST_T_MAX", "bruit")
    assert gb.ghost_bounds() == (300.0, 537.0)


def test_fenetre_de_tir():
    ok = lambda e: gb.ghost_window_ok(e, t_min=330.0, t_max=537.0)
    assert ok(None) is False              # ouverture inconnue => jamais
    assert ok(329.9) is False
    assert ok(330.0) and ok(537.0)
    assert ok(537.1) is False


# ─────────────────────────── randomness synthétique ──────────────────────────
def test_randomness_synthetique_deterministe_et_distincte():
    a = gb.synthetic_randomness(46610, 1)
    assert a == gb.synthetic_randomness(46610, 1)
    assert a != gb.synthetic_randomness(46610, 2)
    assert a != gb.synthetic_randomness(46611, 1)
    assert len(a) == 64 and int(a, 16) >= 0
    assert a != REAL_RND


# ───────────────────────────── cibles de la bande ────────────────────────────
def test_bande_p95_p99():
    band = gb.band_indices(np.arange(100, dtype=float), lo_pct=95, hi_pct=99)
    assert list(band) == [95, 96, 97, 98]


def test_cibles_exclusions_et_rafraichissement():
    t = gb.GhostTargets(list(range(10)), rng=random.Random(0))
    got = t.pick(10, exclude={0, 1}, seen={2: 900, 3: 500}, now_window=1000,
                 refresh_after=250)
    assert 0 not in got and 1 not in got       # liste noire / mémo frais
    assert 2 not in got                        # étiqueté il y a 100 fenêtres
    assert 3 in got                            # étiqueté il y a 500 : à rafraîchir
    assert len(got) == len(set(got)) == 7


def test_cibles_bornees_par_la_taille_du_lot():
    t = gb.GhostTargets(list(range(100)), rng=random.Random(1))
    assert len(t.pick(8, exclude=set(), seen={}, now_window=0)) == 8


# ───────────────────────────── étiquette et troncature ───────────────────────
def test_etiquette_reutilise_le_filtre_de_production():
    assert gb.label_in_zone([0.0] * 16, [False] * 16) is False
    assert gb.label_in_zone([0.0, 1.0] * 8, [False] * 16) is True


def test_troncature():
    assert gb.is_truncated([5, 6, 7], eos_ids=(9,)) is True
    assert gb.is_truncated([5, 9], eos_ids=(9,)) is False


# ─────────────────────────────── intégration moteur ──────────────────────────
class _Tok:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 50 for c in text]

    def decode(self, toks):
        return " ".join(str(t) for t in toks)


class _Env:
    name = "opencodeinstruct"

    def get_problem(self, idx):
        return {"prompt": f"p{idx}", "idx": idx}

    def compute_reward(self, problem, text):
        # prompt pair : groupe mixte (en zone) ; impair : tout raté (hors zone)
        if problem["idx"] % 2 == 0:
            return 1.0 if text.endswith("1 9") else 0.0
        return 0.0


class _Backend:
    """Enregistre l'appel ; livre 16 complétions par prompt (EOS=9)."""

    def __init__(self):
        self.kwargs = None

    def generate_forced_phase1_multi_stream(self, prompts_tokens, **kw):
        self.kwargs = kw
        for pos, idx in enumerate(kw["prompt_indices"]):
            comps = [[3, 1 if r % 2 else 2, 9] for r in range(16)]
            kw["on_group"](pos, idx, comps)
        return None


class _Memo:
    def __init__(self):
        self.calls = []
        self._payable, self._last_w = {}, {}

    def update(self, idx, payable, window_n=None, volume=None):
        self.calls.append((idx, payable, window_n))


def _eng(monkeypatch, *, now=1000.0, elapsed=400.0, state_age=30.0):
    e = engine.MiningEngine.__new__(engine.MiningEngine)
    e._vllm_backend = _Backend()
    e.tokenizer = _Tok()
    e._eos_ids = (9,)
    e.max_new_tokens = 8192
    e._cached_randomness = REAL_RND
    e._cached_window_n = 46610
    e._local_hash = "ckpt"
    e._window_open_ts = now - elapsed
    e._last_state_ts = now - state_age
    e._pool, e._phase1_cache = [], {}
    e._sz_notes = []
    e._sz_note = lambda idx, rewards: e._sz_notes.append(idx)
    e._sz_active = lambda: set()
    e._weights_out_of_sync = lambda: False
    e._primary_eos_id = lambda: 9
    e._ghost_targets = gb.GhostTargets([10, 11, 12, 13], rng=random.Random(0))
    memo = _Memo()
    monkeypatch.setattr(engine, "_ghost_memo", lambda: memo)
    dumps = []
    monkeypatch.setattr(engine, "study_dump",
                        lambda var, row: dumps.append((var, row)))
    monkeypatch.setattr(engine.time, "time", lambda: now)
    return e, memo, dumps


def test_pret_uniquement_dans_la_fenetre_et_le_trou(monkeypatch):
    monkeypatch.setenv("RELIQUARY_GHOST_BAKE", "1")
    e, *_ = _eng(monkeypatch)
    assert e._ghost_ready(1000.0) is True
    e, *_ = _eng(monkeypatch, elapsed=200.0)            # encore en collecte
    assert e._ghost_ready(1000.0) is False
    e, *_ = _eng(monkeypatch, elapsed=600.0)            # trop près du flip
    assert e._ghost_ready(1000.0) is False
    e, *_ = _eng(monkeypatch, state_age=1.0)            # pas de trou 503
    assert e._ghost_ready(1000.0) is False


def test_pas_pret_si_desactive_ou_poids_desynchronises(monkeypatch):
    monkeypatch.delenv("RELIQUARY_GHOST_BAKE", raising=False)
    e, *_ = _eng(monkeypatch)
    assert e._ghost_ready(1000.0) is False
    monkeypatch.setenv("RELIQUARY_GHOST_BAKE", "1")
    e, *_ = _eng(monkeypatch)
    e._weights_out_of_sync = lambda: True
    assert e._ghost_ready(1000.0) is False
    e, *_ = _eng(monkeypatch)
    e._vllm_backend = None
    assert e._ghost_ready(1000.0) is False


def test_le_lot_utilise_une_randomness_synthetique(monkeypatch):
    e, *_ = _eng(monkeypatch)
    asyncio.run(e._ghost_lot(_Env()))
    kw = e._vllm_backend.kwargs
    assert kw["randomness"] != REAL_RND
    assert kw["checkpoint_hash"] == "ckpt"
    assert e._cached_randomness == REAL_RND            # jamais modifiée


def test_interruption_au_flip_et_apres_t_max(monkeypatch):
    e, *_ = _eng(monkeypatch)
    asyncio.run(e._ghost_lot(_Env()))
    abort = e._vllm_backend.kwargs["should_abort"]
    assert abort() is False
    e._cached_randomness = "cd" * 32                   # flip /miner-state
    assert abort() is True
    e._cached_randomness = REAL_RND
    monkeypatch.setattr(engine.time, "time", lambda: 1000.0 + 200.0)  # > T_MAX
    assert abort() is True


def test_jamais_dans_le_pool_ni_le_cache(monkeypatch):
    e, *_ = _eng(monkeypatch)
    asyncio.run(e._ghost_lot(_Env()))
    assert e._pool == [] and e._phase1_cache == {}


def test_etage_A_n_alimente_rien(monkeypatch):
    monkeypatch.delenv("RELIQUARY_GHOST_FEED", raising=False)
    e, memo, dumps = _eng(monkeypatch)
    asyncio.run(e._ghost_lot(_Env()))
    assert memo.calls == [] and e._sz_notes == []
    assert any(var == "RELIQUARY_GHOST_DUMP" for var, _ in dumps)


def test_etage_B_alimente_memo_et_liste_noire(monkeypatch):
    monkeypatch.setenv("RELIQUARY_GHOST_FEED", "1")
    e, memo, _ = _eng(monkeypatch)
    asyncio.run(e._ghost_lot(_Env()))
    fed = {idx for idx, payable, _ in memo.calls if payable}
    assert fed == {10, 12}                             # pairs = en zone
    assert set(e._sz_notes) == {11, 13}                # impairs = hors zone
    assert all(w == 46610 for _, _, w in memo.calls)


def test_les_prompts_etiquetes_sont_memorises(monkeypatch):
    e, *_ = _eng(monkeypatch)
    asyncio.run(e._ghost_lot(_Env()))
    assert set(e._ghost_seen) == {10, 11, 12, 13}
    assert set(e._ghost_seen.values()) == {46610}


def test_branche_en_pause_lance_le_fantome():
    src = inspect.getsource(engine.MiningEngine._generator_loop)
    head = src.split("self._bake_paused(env_name", 1)[1].split("cooldown = self._cooldowns", 1)[0]
    assert "self._ghost_ready(" in head
    assert "self._ghost_lot(" in head


def test_launcher_debit_fantome_0921_soir():
    src = open("ops/launch_miner_v4.sh", encoding="utf-8").read()
    assert "RELIQUARY_GHOST_LOT=${RELIQUARY_GHOST_LOT:-16}" in src
    assert "RELIQUARY_GHOST_T_MIN=${RELIQUARY_GHOST_T_MIN:-280}" in src


def test_lot_et_bornes_lus_par_le_module(monkeypatch):
    monkeypatch.setenv("RELIQUARY_GHOST_LOT", "16")
    monkeypatch.setenv("RELIQUARY_GHOST_T_MIN", "280")
    monkeypatch.delenv("RELIQUARY_GHOST_T_MAX", raising=False)
    assert gb.ghost_lot_size() == 16
    assert gb.ghost_bounds() == (280.0, 537.0)
