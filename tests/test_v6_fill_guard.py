"""Régime de fenêtre v6 « fill-closed » — gardes du moteur (agent E).

Chaque comportement nouveau est gaté par la présence de ``state.fill_closed``
(objet publié par ``/state`` sous v6). Sous v5 (``fill_closed`` absent ou
None) chaque helper doit rendre la décision historique : tests « v5
inchangé » à côté de chaque test v6. Les helpers sont PURS : les tests
construisent des ``SimpleNamespace`` pour ``state`` / ``fill_closed`` afin de
ne pas dépendre du modèle pydantic (port wire fait par un autre agent).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from reliquary.miner import engine

CAP = 32                                   # cap v5/v6, passé EXPLICITEMENT aux helpers purs
ENGINE_CAP = engine.MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW   # valeur liée à l'import (branchements)

CUTOFF = 1_789_000_000.0


def _fc(phase="collecting", cutoff=CUTOFF, remaining=None, seconds=1767.0):
    return SimpleNamespace(
        phase=phase,
        precommit_cutoff_ts=cutoff,
        precommit_seconds=seconds,
        remaining={"opencodeinstruct": 448} if remaining is None else remaining,
    )


def _state(fc=None, window_n=42, state="open"):
    return SimpleNamespace(fill_closed=fc, window_n=window_n, state=state,
                           randomness="ab" * 16)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("RELIQUARY_V6_FILL_CUTOFF_MARGIN_S", "RELIQUARY_STATE_RETRY_MAX_S"):
        monkeypatch.delenv(k, raising=False)


# ---------------------------------------------------------------- A. veto de tir
def test_veto_none_sous_v5():
    # fill_closed absent → jamais de veto, quelles que soient les autres valeurs
    for kw in (
        dict(now=CUTOFF + 10, submitted=CAP, env="opencodeinstruct", state_age_s=99),
        dict(now=0.0, submitted=0, env="x", state_age_s=0.0),
    ):
        assert engine.fill_closed_fire_veto(None, **kw) is None


def test_veto_phase():
    assert engine.fill_closed_fire_veto(
        _fc(phase="draining"), CUTOFF - 1000, 0, "opencodeinstruct",
    ) == "phase_draining"
    assert engine.fill_closed_fire_veto(
        _fc(phase="sealed"), CUTOFF - 1000, 0, "opencodeinstruct",
    ) == "phase_sealed"
    assert engine.fill_closed_fire_veto(
        _fc(), CUTOFF - 1000, 0, "opencodeinstruct",
    ) is None


def test_veto_cutoff(monkeypatch):
    assert engine.fill_closed_fire_veto(_fc(), CUTOFF - 39, 0, "opencodeinstruct") == "cutoff"
    assert engine.fill_closed_fire_veto(_fc(), CUTOFF - 41, 0, "opencodeinstruct") is None
    # marge surchargeable par l'env (launcher v6 : 40)
    monkeypatch.setenv("RELIQUARY_V6_FILL_CUTOFF_MARGIN_S", "10")
    assert engine.fill_closed_fire_veto(_fc(), CUTOFF - 39, 0, "opencodeinstruct") is None
    assert engine.fill_closed_fire_veto(_fc(), CUTOFF - 9, 0, "opencodeinstruct") == "cutoff"
    # cutoff inconnu (champ optionnel) → pas de veto cutoff
    assert engine.fill_closed_fire_veto(_fc(cutoff=None), CUTOFF, 0, "opencodeinstruct") is None


def test_veto_env_budget():
    assert engine.fill_closed_fire_veto(
        _fc(remaining={"opencodeinstruct": 0}), CUTOFF - 1000, 0, "opencodeinstruct",
    ) == "env_budget"
    # env absent du dict → pas de veto
    assert engine.fill_closed_fire_veto(
        _fc(remaining={"openmathinstruct": 0}), CUTOFF - 1000, 0, "opencodeinstruct",
    ) is None
    # env inconnue (None) → pas de veto env
    assert engine.fill_closed_fire_veto(
        _fc(remaining={"opencodeinstruct": 0}), CUTOFF - 1000, 0, None,
    ) is None
    assert engine.fill_closed_env_exhausted(_fc(remaining={"opencodeinstruct": 0}),
                                            "opencodeinstruct") is True
    assert engine.fill_closed_env_exhausted(None, "opencodeinstruct") is False


def test_veto_quota():
    assert engine.fill_closed_fire_veto(_fc(), CUTOFF - 1000, CAP, "opencodeinstruct", cap=CAP) == "quota"
    assert engine.fill_closed_fire_veto(_fc(), CUTOFF - 1000, CAP - 1, "opencodeinstruct", cap=CAP) is None


def test_veto_state_stale():
    assert engine.fill_closed_fire_veto(
        _fc(), CUTOFF - 1000, 0, "opencodeinstruct", state_age_s=6,
    ) == "state_stale"
    assert engine.fill_closed_fire_veto(
        _fc(), CUTOFF - 1000, 0, "opencodeinstruct", state_age_s=4,
    ) is None


def test_veto_ordre_phase_avant_le_reste():
    # la phase prime (un draining à quota plein rend phase_*, pas quota)
    assert engine.fill_closed_fire_veto(
        _fc(phase="draining"), CUTOFF, CAP, "opencodeinstruct", state_age_s=99, cap=CAP,
    ) == "phase_draining"


def test_maybe_fire_on_append_veto_compte_dans_fire_diag():
    """Branchement : sous v6 le veto est compté dans fire_diag et rien n'est tiré."""
    eng = engine.MiningEngine.__new__(engine.MiningEngine)
    from reliquary.protocol.submission import WindowState
    st = SimpleNamespace(
        state=WindowState.OPEN, randomness="ab" * 16, window_n=42,
        fill_closed=_fc(phase="draining"),
    )
    eng._last_state = st
    eng._last_state_ts = 1.0          # âge énorme (epoch+1 s), mais la phase prime
    eng._fire_ctx = ("http://v", object(), [])
    eng._cached_randomness = st.randomness
    eng._cached_window_n = 42
    eng._submitted_count = {}
    assert eng._maybe_fire_on_append() is False
    assert eng._fire_diag[42]["veto_phase_draining"] == 1


def test_maybe_fire_on_append_v5_inchange():
    """v5 : fill_closed absent → la garde suivante (fire-as-ready) est atteinte."""
    eng = engine.MiningEngine.__new__(engine.MiningEngine)
    from reliquary.protocol.submission import WindowState
    st = SimpleNamespace(
        state=WindowState.OPEN, randomness="ab" * 16, window_n=42,
    )
    eng._last_state = st
    eng._last_state_ts = 0.0
    eng._fire_ctx = ("http://v", object(), [])
    eng._cached_randomness = st.randomness
    eng._cached_window_n = 42
    eng._submitted_count = {}
    eng._fire_as_ready = lambda w, r: False
    assert eng._maybe_fire_on_append() is False
    assert eng._fire_diag[42]["not_fire_as_ready"] == 1
    assert not any(k.startswith("veto") for k in eng._fire_diag[42])


# ---------------------------------------------------------- B. garde pré-flip
def test_guard_elapsed_v5_inchange():
    # v5 : elapsed = now − open observé ; open inconnu (boot) → None → "full"
    assert engine.guard_elapsed(_state(None), 100.0, 130.0) == 30.0
    assert engine.guard_elapsed(None, 100.0, 130.0) == 30.0
    assert engine.guard_elapsed(_state(None), None, 130.0) is None
    assert engine.guard_elapsed(_state(None), 0.0, 130.0) is None


def test_guard_elapsed_under_v6():
    # v6 : pas de flip à l'horloge → None → bake_guard_decision(None) == "full"
    assert engine.guard_elapsed(_state(_fc()), 100.0, 100.0 + 999.0) is None
    assert engine.bake_guard_decision(engine.guard_elapsed(_state(_fc()), 100.0, 9e9)) == "full"


# ------------------------------------------------------- C. sceau batch_filled
def test_batch_filled_ne_scelle_pas_sous_v6():
    assert engine.batch_filled_seals_window(_state(_fc())) is False


def test_batch_filled_scelle_sous_v5():
    assert engine.batch_filled_seals_window(_state(None)) is True
    assert engine.batch_filled_seals_window(SimpleNamespace(window_n=1)) is True


# --------------------------------------------------------------- D. quota exact
REFUNDED = {"stale_round", "future_round", "batch_filled", "window_not_active",
            "window_mismatch", "wrong_checkpoint", "prompt_in_cooldown",
            "prompt_out_of_range", "prompt_full", "hash_duplicate", "rate_limited"}


def test_quota_refund_reasons():
    for r in REFUNDED:
        assert engine.quota_refund_for_reject(r) == -1, r
    for r in ("precommit_expired", "grail_fail", "seed_mismatch", "out_of_zone",
              "token_tampered", "bad_envelope_signature", None, ""):
        assert engine.quota_refund_for_reject(r) == 0, r


def test_apply_quota_after_fire_v6_rembourse_par_stade():
    # règle validateur : +1 SEULEMENT quand le precommit est enregistré →
    # tout rejet de stade precommit est remboursé, quel que soit le motif
    assert engine.apply_quota_after_fire(5, [("precommit", "stale_round")], fill_closed=True, cap=CAP) == 4
    assert engine.apply_quota_after_fire(5, [("precommit", "x")], fill_closed=True, cap=CAP) == 4
    # stade corps (precommit accepté) : jamais remboursé, même hash_duplicate
    assert engine.apply_quota_after_fire(5, [("body", "hash_duplicate")], fill_closed=True, cap=CAP) == 5
    # stade inconnu → repli sur la table des motifs
    assert engine.apply_quota_after_fire(5, [(None, "stale_round")], fill_closed=True, cap=CAP) == 4
    assert engine.apply_quota_after_fire(5, [(None, "grail_fail")], fill_closed=True, cap=CAP) == 5
    assert engine.apply_quota_after_fire(5, ["stale_round", "hash_duplicate", "grail_fail"],
                                         fill_closed=True, cap=CAP) == 3      # str = stade inconnu
    assert engine.apply_quota_after_fire(1, [("precommit", "stale_round")] * 3, fill_closed=True, cap=CAP) == 0


def test_rate_limited_marque_quota_epuise():
    assert engine.apply_quota_after_fire(
        7, [("precommit", "stale_round"), ("precommit", "rate_limited")], fill_closed=True, cap=CAP,
    ) == CAP
    assert engine.apply_quota_after_fire(7, [(None, "rate_limited")], fill_closed=True, cap=CAP) == CAP


def test_apply_quota_after_fire_v5_inchange():
    assert engine.apply_quota_after_fire(
        5, [("precommit", "stale_round"), (None, "hash_duplicate"), ("precommit", "rate_limited")],
        fill_closed=False, cap=CAP,
    ) == 5


def test_retryable_reasons_v6_sans_rate_limited():
    v5 = engine.fire_retryable_reasons(_state(None))
    v6 = engine.fire_retryable_reasons(_state(_fc()))
    assert "rate_limited" in v5
    assert v5 == {"stale_round", "batch_filled", "rate_limited", "future_round",
                  "registration_unavailable"}
    assert v6 == v5 - {"rate_limited"}


# ------------------------------------------------------------ E. pause du bake
def test_pause_bake_v5_jamais():
    assert engine.should_pause_bake(None, 999.0, CAP, "opencodeinstruct", cap=CAP) is False
    assert engine.should_pause_bake(_state(None), 999.0, CAP, "opencodeinstruct", cap=CAP) is False


def test_pause_bake_gap_503():
    assert engine.should_pause_bake(_state(_fc()), 6.0, 0, "opencodeinstruct") is True
    assert engine.should_pause_bake(_state(_fc()), 4.0, 0, "opencodeinstruct") is False


def test_pause_bake_quota():
    assert engine.should_pause_bake(_state(_fc()), 0.0, CAP, "opencodeinstruct", cap=CAP) is True
    assert engine.should_pause_bake(_state(_fc()), 0.0, CAP - 1, "opencodeinstruct", cap=CAP) is False


def test_pause_bake_env_budget():
    st = _state(_fc(remaining={"opencodeinstruct": 0}))
    assert engine.should_pause_bake(st, 0.0, 0, "opencodeinstruct") is True
    assert engine.should_pause_bake(st, 0.0, 0, "openmathinstruct") is False


def test_pause_bake_phase():
    assert engine.should_pause_bake(_state(_fc(phase="draining")), 0.0, 0, "opencodeinstruct") is True
    assert engine.should_pause_bake(_state(_fc(phase="sealed")), 0.0, 0, "opencodeinstruct") is True


def _engine_for_post_grade(st):
    import asyncio
    eng = engine.MiningEngine.__new__(engine.MiningEngine)
    eng._last_state = st
    eng._last_state_ts = 0.0
    eng._pool_lock = asyncio.Lock()
    eng._pool = []
    eng._submitted_count = {42: ENGINE_CAP}
    eng._cached_window_n = 42
    eng._cached_randomness = "ab" * 16
    eng.active_envs = ["opencodeinstruct"]
    eng._maybe_fire_on_append = lambda: False
    return eng


def test_post_grade_entry_v6_quota_atteint_hors_pool(monkeypatch):
    import asyncio
    monkeypatch.delenv("RELIQUARY_DROP_POOL_ON_CKPT", raising=False)
    eng = _engine_for_post_grade(_state(_fc()))
    entry = {"prompt_idx": 7, "env_name": "opencodeinstruct"}
    entries: list = []
    asyncio.run(eng._post_grade_entry(entry, 7, entries, object()))
    assert eng._pool == []                 # quota 32 atteint → pas en pool


def test_post_grade_entry_v5_inchange(monkeypatch):
    import asyncio
    monkeypatch.delenv("RELIQUARY_DROP_POOL_ON_CKPT", raising=False)
    eng = _engine_for_post_grade(_state(None))
    entry = {"prompt_idx": 7, "env_name": "opencodeinstruct"}
    entries: list = []
    asyncio.run(eng._post_grade_entry(entry, 7, entries, object()))
    assert eng._pool == [entry] and entries == [entry]


# ---------------------------------------------------------------- F. heartbeat
def test_heartbeat_due():
    assert engine.heartbeat_due(None, 1000.0) is True
    assert engine.heartbeat_due(1000.0, 1059.9) is False
    assert engine.heartbeat_due(1000.0, 1060.0) is True
    assert engine.heartbeat_due(1000.0, 1010.0, every=10) is True


def test_heartbeat_enabled_v5_off_v6_on():
    assert engine.heartbeat_enabled(_state(None), 5) is False
    assert engine.heartbeat_enabled(None, 5) is False
    assert engine.heartbeat_enabled(_state(_fc()), 5) is True      # fill_closed publié
    assert engine.heartbeat_enabled(None, 6) is True               # boot v6, gap 503


def test_heartbeat_line_format():
    line = engine.heartbeat_line(_state(_fc(phase="draining"), window_n=42), 7, 32, 12.4)
    assert line == "heartbeat window=42 state=open phase=draining quota=7/32 age=12s"
    # gap 503 : pas de state → window du dernier connu, state=503, phase=-
    line = engine.heartbeat_line(None, 0, 32, 90.0, state_label="503", window_n=41)
    assert line == "heartbeat window=41 state=503 phase=- quota=0/32 age=90s"


# ------------------------------------------------------------- G. backoff 503
def test_state_retry_delay_v5_inchange():
    # défaut RELIQUARY_STATE_RETRY_MAX_S = 0.05 → toujours 0.05 (v5)
    for n in (0, 1, 200, 201, 250, 10_000):
        assert engine.state_retry_delay(n) == 0.05


def test_state_retry_delay_v6(monkeypatch):
    monkeypatch.setenv("RELIQUARY_STATE_RETRY_MAX_S", "0.25")
    assert engine.state_retry_delay(0) == 0.05
    assert engine.state_retry_delay(200) == 0.05
    assert engine.state_retry_delay(201) == pytest.approx(0.10)
    assert engine.state_retry_delay(202) == pytest.approx(0.20)
    assert engine.state_retry_delay(203) == 0.25
    assert engine.state_retry_delay(100_000) == 0.25     # pas d'overflow


# --------------------------------------------------------------- H. open exact
def test_window_open_ts_v5_est_now():
    assert engine.window_open_ts(_state(None), 123.0) == 123.0
    assert engine.window_open_ts(None, 123.0) == 123.0


def test_window_open_ts_v6_exact():
    assert engine.window_open_ts(_state(_fc(cutoff=CUTOFF, seconds=1767.0)), 5.0) == CUTOFF - 1767.0
    # champs optionnels absents → now
    assert engine.window_open_ts(_state(_fc(cutoff=None)), 5.0) == 5.0
    assert engine.window_open_ts(_state(_fc(seconds=None)), 5.0) == 5.0


# ----------------------------------------------- J. libellé du log ombre (LTA)
def test_shadow_log_reports_configured_thresholds(monkeypatch, caplog):
    import logging
    monkeypatch.setenv("RELIQUARY_LTA_CHOSEN_MAX", "2e-5")
    monkeypatch.setenv("RELIQUARY_LTA_ARGMAX_MIN", "0.97")
    msg = engine.shadow_token_auth_message(7, 3, 16)
    assert "2e-05/0.97" in msg and "prompt=7" in msg and "3/16" in msg
    assert "1e-5/0.99" not in msg
    with caplog.at_level(logging.INFO, logger="reliquary.miner.engine"):
        engine.logger.info(msg)
    assert "2e-05/0.97" in caplog.text


def test_shadow_log_defaults_sont_ceux_du_code(monkeypatch):
    monkeypatch.delenv("RELIQUARY_LTA_CHOSEN_MAX", raising=False)
    monkeypatch.delenv("RELIQUARY_LTA_ARGMAX_MIN", raising=False)
    assert "0.0001/0.985" in engine.shadow_token_auth_message(1, 1, 16)


# ------------------------------------------- revue item 5 : stade marqué par le submitter
async def test_submitter_marque_le_stade_precommit(monkeypatch):
    from tests.test_precommit_port import BINDING_FIXTURE, _Recorder, _request
    from reliquary.miner.submitter import submit_batch_v2
    import reliquary.protocol.signatures as _sg
    # bittensor est cassé sur la dev box (baseline test_precommit_port : 5 failed)
    # → signature factice, le contrat testé ici est le MARQUAGE du stade.
    monkeypatch.setattr(_sg, "sign_precommit", lambda *, wallet, **f: b"\x00" * 64)
    # La fixture ``_request`` porte 8 rollouts (ère v3) ; ``M_ROLLOUTS`` est
    # lié à l'import (16 sous v5/v6) → on l'aligne sur la fixture, le contrat
    # testé ici est le marquage du stade, pas le nombre de rollouts.
    import reliquary.protocol.submission as _sub
    monkeypatch.setattr(_sub, "M_ROLLOUTS", 8)
    _wallet = lambda: SimpleNamespace(hotkey=SimpleNamespace(sign=lambda b: b"\x00" * 64))
    rec = _Recorder(precommit_body={"accepted": False, "reason": "stale_round"})
    async with rec.client as client:
        resp = await submit_batch_v2(
            "http://v", _request(), client=client,
            wallet=_wallet(), randomness=BINDING_FIXTURE["randomness"],
        )
    assert not resp.accepted and getattr(resp, "_stage", None) == "precommit"
    # un corps refusé (precommit accepté) n'est PAS marqué precommit
    rec = _Recorder()
    rec._submit_status = 200

    async def handler(request):
        import httpx
        rec.calls.append(request)
        if request.url.path.endswith("/submit/precommit"):
            return httpx.Response(200, json=rec._precommit_body)
        return httpx.Response(200, json={"accepted": False, "reason": "hash_duplicate"})
    rec.handler = handler
    async with rec.client as client:
        resp = await submit_batch_v2(
            "http://v", _request(), client=client,
            wallet=_wallet(), randomness=BINDING_FIXTURE["randomness"],
        )
    assert not resp.accepted and getattr(resp, "_stage", None) != "precommit"


# --------------------- revue item 12 : remboursement bout-en-bout dans _fire_for_window
def _engine_for_fire(st, entries):
    import asyncio, time
    eng = engine.MiningEngine.__new__(engine.MiningEngine)
    eng._pool, eng._pool_lock = list(entries), asyncio.Lock()
    eng._submitted_count = {}
    eng._last_state = st
    eng._last_state_ts = time.time()
    eng.active_envs = ["opencodeinstruct"]
    eng.envs = {"opencodeinstruct": object()}
    eng._active_prompt_range = lambda w, r, env=None: None
    return eng


def _fake_submit(verdicts):
    from reliquary.protocol.submission import BatchSubmissionResponse, RejectReason

    async def _submit_entry(entry, state, url, client, results):
        stage, reason = verdicts[entry["prompt_idx"]]
        resp = BatchSubmissionResponse(
            accepted=(reason == "accepted"), reason=RejectReason(reason))
        if stage:
            resp._stage = stage
        return entry, resp
    return _submit_entry


def _fire_state(fc):
    from reliquary.protocol.submission import WindowState
    return SimpleNamespace(window_n=42, randomness="ab" * 16, cooldown_prompts=[],
                           state=WindowState.OPEN, fill_closed=fc)


def test_fire_for_window_v6_rembourse_le_stade_precommit(monkeypatch):
    import asyncio
    monkeypatch.delenv("RELIQUARY_FIRE_CURFEW_S", raising=False)
    entries = [{"prompt_idx": 1, "env_name": "opencodeinstruct"},
               {"prompt_idx": 2, "env_name": "opencodeinstruct"},
               {"prompt_idx": 3, "env_name": "opencodeinstruct"}]
    eng = _engine_for_fire(_fire_state(_fc()), entries)
    eng._submit_entry = _fake_submit({1: ("precommit", "stale_round"),
                                      2: (None, "accepted"),
                                      3: ("body", "hash_duplicate")})
    asyncio.run(eng._fire_for_window(eng._last_state, "http://v", None, []))
    # 3 drainées − 1 rejet precommit = 2 ; le stale_round est re-queué
    assert eng._submitted_count[42] == 2
    assert [e["prompt_idx"] for e in eng._pool] == [1]
    assert getattr(eng, "_sealed_window", None) is None


def test_fire_for_window_v5_inchange(monkeypatch):
    import asyncio
    monkeypatch.delenv("RELIQUARY_FIRE_CURFEW_S", raising=False)
    entries = [{"prompt_idx": 1, "env_name": "opencodeinstruct"},
               {"prompt_idx": 2, "env_name": "opencodeinstruct"}]
    eng = _engine_for_fire(_fire_state(None), entries)
    eng._submit_entry = _fake_submit({1: ("precommit", "stale_round"),
                                      2: (None, "accepted")})
    asyncio.run(eng._fire_for_window(eng._last_state, "http://v", None, []))
    assert eng._submitted_count[42] == 2           # jamais décrémenté sous v5
    assert [e["prompt_idx"] for e in eng._pool] == [1]


# ---------------------------------------- revue item 1 : hors pool SEULEMENT phase/quota
def test_should_skip_pool():
    st = _state(_fc())
    assert engine.should_skip_pool(_state(None), CAP, "opencodeinstruct", cap=CAP) is False
    assert engine.should_skip_pool(st, CAP, "opencodeinstruct", cap=CAP) is True
    assert engine.should_skip_pool(_state(_fc(phase="draining")), 0, "opencodeinstruct", cap=CAP) is True
    assert engine.should_skip_pool(st, CAP - 1, "opencodeinstruct", cap=CAP) is False
    # JAMAIS sur remaining[env]==0 (le veto de tir retient l'entrée, le budget se remplit)
    assert engine.should_skip_pool(_state(_fc(remaining={"opencodeinstruct": 0})), 0,
                                   "opencodeinstruct", cap=CAP) is False


def test_post_grade_entry_v6_state_vieux_va_en_pool(monkeypatch):
    """Reload checkpoint 45 s / gel HF : /state vieux de 60 s, collecting,
    quota 3/32 → le groupe fini VA en pool (le veto de tir le retient)."""
    import asyncio, time
    monkeypatch.delenv("RELIQUARY_DROP_POOL_ON_CKPT", raising=False)
    eng = _engine_for_post_grade(_state(_fc(remaining={"opencodeinstruct": 0})))
    eng._last_state_ts = time.time() - 60.0
    eng._submitted_count = {42: 3}
    entry = {"prompt_idx": 7, "env_name": "opencodeinstruct"}
    entries: list = []
    asyncio.run(eng._post_grade_entry(entry, 7, entries, object()))
    assert eng._pool == [entry]


# ------------------------------------ revue item 6 : pause du bake aussi sur le cutoff
def test_pause_bake_cutoff(monkeypatch):
    st = _state(_fc())
    assert engine.should_pause_bake(st, 0.0, 0, "opencodeinstruct", now=CUTOFF - 39, cap=CAP) is True
    assert engine.should_pause_bake(st, 0.0, 0, "opencodeinstruct", now=CUTOFF - 41, cap=CAP) is False
    monkeypatch.setenv("RELIQUARY_V6_FILL_CUTOFF_MARGIN_S", "10")
    assert engine.should_pause_bake(st, 0.0, 0, "opencodeinstruct", now=CUTOFF - 39, cap=CAP) is False
    # now absent → pas de critère cutoff ; v5 → jamais
    assert engine.should_pause_bake(st, 0.0, 0, "opencodeinstruct", cap=CAP) is False
    assert engine.should_pause_bake(_state(None), 0.0, 0, "opencodeinstruct", now=CUTOFF, cap=CAP) is False


# ------------------------------------------ revue item 2 : heartbeat qui lève
def test_heartbeat_safe_avale_l_exception(caplog):
    import logging
    eng = engine.MiningEngine.__new__(engine.MiningEngine)

    def _boom(state):
        raise RuntimeError("hb")
    eng._maybe_heartbeat = _boom
    with caplog.at_level(logging.DEBUG, logger="reliquary.miner.engine"):
        eng._heartbeat_safe(None)          # ne lève pas
    assert "heartbeat" in caplog.text


def test_trigger_loop_survit_a_un_heartbeat_qui_leve_sur_503(monkeypatch):
    """Chemin 503 : un heartbeat qui lève ne doit pas sortir de _trigger_loop
    (aucun superviseur) — la boucle re-polle /state."""
    import asyncio
    from reliquary.miner import submitter
    calls = {"n": 0}

    async def _fetch(url, client=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise submitter.SubmissionError("503 no_active_window")
        raise StopAsyncIteration
    monkeypatch.setattr(submitter, "get_window_state_v2_with_resp", _fetch)
    eng = engine.MiningEngine.__new__(engine.MiningEngine)

    def _boom(state):
        raise RuntimeError("hb")
    eng._maybe_heartbeat = _boom
    with pytest.raises(StopAsyncIteration):
        asyncio.run(eng._trigger_loop("http://v", None, []))
    assert calls["n"] == 2                 # la boucle a continué après le 503


# ------------------------------------------ revue item 11 : skew d'horloge au flip
def test_clock_skew_s():
    assert engine.clock_skew_s(_state(None), 100.0) is None
    st = _state(_fc(cutoff=CUTOFF, seconds=1767.0))
    assert engine.clock_skew_s(st, CUTOFF - 1767.0 - 1.5) == pytest.approx(1.5)
    assert engine.clock_skew_s(_state(_fc(cutoff=None)), 5.0) is None
    assert engine.clock_skew_warn(1.5) is False
    assert engine.clock_skew_warn(-10.1) is True and engine.clock_skew_warn(10.1) is True
    assert engine.clock_skew_warn(None) is False
