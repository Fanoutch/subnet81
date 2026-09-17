"""Service « réplique validateur » (ops/replica_service.py) et son client.

Mesuré le 14/09 (banc, 191 rollouts vLLM finis sur EOS) : la pile HF du
validateur (flash-attn 2, torch 2.7, transformers 5.10.4) ne reproduit
l'EOS final que de 86,4 % d'entre eux, et notre garde locale HF sdpa n'est
d'accord avec elle que sur 161/191 rollouts (16 laissés passer à tort, 14
jetés à tort). La garde doit donc interroger une réplique exacte de la pile
du validateur, dans son propre venv : ce service. Ses parties pures sont
testées ici avec un forward factice ; le forward réel est upstream.
"""
from __future__ import annotations

import importlib.util
import json
import os
import socket
import threading
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
EOS = 7
VOCAB = 10


def _load_service():
    spec = importlib.util.spec_from_file_location(
        "replica_service", ROOT / "ops" / "replica_service.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeBackend:
    """Remplace modèle + fonctions upstream : logits uniformes sauf EOS."""

    def __init__(self, p_eos: float):
        self.p_eos = p_eos
        self.loads: list[str] = []
        self.forwards = 0

    def load(self, model_path):
        self.loads.append(model_path)
        return {EOS}

    def terminal_row(self, tokens):
        self.forwards += 1
        rest = (1.0 - self.p_eos) / (VOCAB - 1)
        probs = torch.full((VOCAB,), rest)
        probs[EOS] = self.p_eos
        return torch.log(probs)

    def u_at(self, randomness, prompt_idx, checkpoint_hash, rollout, j):
        return self.u

    def diagnose(self, row, token, u):
        probs = torch.softmax(row.float(), -1)
        cdf = torch.cumsum(probs, -1)
        picked = int(torch.searchsorted(cdf, torch.tensor(float(u)), right=True))
        return picked == token, 0.0, picked

    u = 0.0


def _interval(p_eos):
    rest = (1.0 - p_eos) / (VOCAB - 1)
    return rest * EOS, rest * EOS + p_eos


def _req(items, model_path="/m/rev1"):
    return {"op": "terminal", "model_path": model_path, "randomness": "ab",
            "checkpoint_hash": "rev1", "prompt_idx": 3, "items": items}


def test_verdict_ok_et_pick_eos():
    svc = _load_service()
    be = _FakeBackend(0.6)
    lo, hi = _interval(0.6)
    be.u = (lo + hi) / 2
    state = svc.ReplicaState(be)
    out = state.handle(_req([{"rollout": 0, "prompt_len": 2,
                              "tokens": [1, 2, 3, EOS]}]))
    assert out["ok"] is True
    r = out["results"][0]
    assert r["ok"] is True and r["pick"] == EOS


def test_verdict_ko_renvoie_le_token_a_tirer():
    svc = _load_service()
    be = _FakeBackend(0.6)
    lo, _ = _interval(0.6)
    be.u = lo / 2                         # tombe sur un token < EOS
    state = svc.ReplicaState(be)
    r = state.handle(_req([{"rollout": 0, "prompt_len": 2,
                            "tokens": [1, 2, 3, EOS]}]))["results"][0]
    assert r["ok"] is False
    assert r["pick"] != EOS and 0 <= r["pick"] < EOS


def test_rollout_sans_eos_final_non_juge():
    svc = _load_service()
    be = _FakeBackend(0.6)
    state = svc.ReplicaState(be)
    r = state.handle(_req([{"rollout": 0, "prompt_len": 2,
                            "tokens": [1, 2, 3, 4]}]))["results"][0]
    assert r["ok"] is None
    assert be.forwards == 0


def test_rechargement_seulement_si_le_modele_change():
    svc = _load_service()
    be = _FakeBackend(0.9)
    be.u = 0.5
    state = svc.ReplicaState(be)
    item = {"rollout": 0, "prompt_len": 2, "tokens": [1, 2, EOS]}
    state.handle(_req([item], "/m/rev1"))
    state.handle(_req([item], "/m/rev1"))
    state.handle(_req([item], "/m/rev2"))
    assert be.loads == ["/m/rev1", "/m/rev2"]


def test_requete_invalide_ne_tue_pas_le_service():
    svc = _load_service()
    state = svc.ReplicaState(_FakeBackend(0.6))
    out = state.handle({"op": "inconnue"})
    assert out["ok"] is False and "error" in out


# --------------------------------------------------------- client ↔ serveur
def test_client_serveur_unix_socket(tmp_path):
    from reliquary.miner import replica_client

    svc = _load_service()
    be = _FakeBackend(0.6)
    lo, hi = _interval(0.6)
    be.u = (lo + hi) / 2
    sock = str(tmp_path / "replica.sock")
    server = svc.make_server(sock, svc.ReplicaState(be))
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    try:
        res = replica_client.terminal_verdicts(
            sock, model_path="/m/rev1", randomness="ab", checkpoint_hash="rev1",
            prompt_idx=3,
            items=[{"rollout": 0, "prompt_len": 2, "tokens": [1, 2, EOS]},
                   {"rollout": 1, "prompt_len": 2, "tokens": [1, 2, 4]}],
            timeout=5.0)
        assert [r["ok"] for r in res] == [True, None]
        assert replica_client.ping(sock, timeout=5.0) is True
    finally:
        server.shutdown()
        server.server_close()


def test_client_renvoie_none_si_service_absent(tmp_path):
    from reliquary.miner import replica_client

    assert replica_client.terminal_verdicts(
        str(tmp_path / "absent.sock"), model_path="/m", randomness="ab",
        checkpoint_hash="r", prompt_idx=1,
        items=[{"rollout": 0, "prompt_len": 1, "tokens": [1, EOS]}],
        timeout=1.0) is None
    assert replica_client.ping(str(tmp_path / "absent.sock"), timeout=1.0) is False


# ------------------------------------------------ service multi-requêtes (B)
# Mesuré 14/09 (fen 45897-45898) : serveur mono-fil + file d'écoute de 5 →
# ``BlockingIOError(11)`` côté mineur (repli sur la garde locale) et réparations
# de 10-40 tokens payées 10-15 s d'attente (les 10 groupes d'un bake arrivent
# ensemble). Le service accepte désormais les connexions en parallèle et borne
# les forwards simultanés à ``workers`` (1 = sérialisé, comportement validé).
import time as _time


class _SlowBackend(_FakeBackend):
    def __init__(self, p_eos, delay=0.15):
        super().__init__(p_eos)
        self.delay = delay
        self.inflight = 0
        self.max_inflight = 0
        self.loading = False
        self.compute_during_load = 0
        self._lk = threading.Lock()
        self.contexts = 0

    def load(self, model_path):
        self.loading = True
        _time.sleep(self.delay)
        with self._lk:
            if self.inflight:
                self.compute_during_load += 1
        self.loading = False
        return super().load(model_path)

    def terminal_row(self, tokens):
        with self._lk:
            self.inflight += 1
            self.max_inflight = max(self.max_inflight, self.inflight)
        _time.sleep(self.delay)
        with self._lk:
            self.inflight -= 1
        return super().terminal_row(tokens)

    def worker_context(self):
        import contextlib

        self.contexts += 1
        return contextlib.nullcontext()


def _serve(tmp_path, state):
    svc = _load_service()
    sock = str(tmp_path / "r.sock")
    server = svc.make_server(sock, state)
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    return sock, server


def _parallel_calls(sock, n, model_path="/m/rev1"):
    from reliquary.miner import replica_client

    out = [None] * n

    def one(k):
        out[k] = replica_client.terminal_verdicts(
            sock, model_path=model_path, randomness="ab", checkpoint_hash="rev1",
            prompt_idx=3,
            items=[{"rollout": k, "prompt_len": 2, "tokens": [1, 2, EOS]}],
            timeout=20.0)

    ths = [threading.Thread(target=one, args=(k,)) for k in range(n)]
    t0 = _time.monotonic()
    for t in ths:
        t.start()
    for t in ths:
        t.join(30)
    return out, _time.monotonic() - t0


def test_workers_forwards_en_parallele(tmp_path):
    svc = _load_service()
    be = _SlowBackend(0.6)
    be.u = sum(_interval(0.6)) / 2
    sock, server = _serve(tmp_path, svc.ReplicaState(be, workers=4))
    try:
        out, dt = _parallel_calls(sock, 4)
    finally:
        server.shutdown()
        server.server_close()
    assert all(o and o[0]["ok"] is True for o in out)
    assert be.max_inflight >= 2
    assert be.max_inflight <= 4
    assert be.contexts >= 4, "chaque forward doit passer par worker_context"
    assert dt < 4 * be.delay


def test_un_worker_reste_serialise(tmp_path):
    svc = _load_service()
    be = _SlowBackend(0.6, delay=0.05)
    be.u = sum(_interval(0.6)) / 2
    sock, server = _serve(tmp_path, svc.ReplicaState(be))
    try:
        out, _dt = _parallel_calls(sock, 4)
    finally:
        server.shutdown()
        server.server_close()
    assert all(o and o[0]["ok"] is True for o in out)
    assert be.max_inflight == 1


def test_rafale_de_connexions_sans_refus(tmp_path):
    svc = _load_service()
    be = _SlowBackend(0.6, delay=0.01)
    be.u = sum(_interval(0.6)) / 2
    sock, server = _serve(tmp_path, svc.ReplicaState(be, workers=2))
    try:
        out, _dt = _parallel_calls(sock, 48)
    finally:
        server.shutdown()
        server.server_close()
    assert sum(1 for o in out if o is None) == 0


def test_chargement_exclusif_des_forwards(tmp_path):
    svc = _load_service()
    be = _SlowBackend(0.6, delay=0.1)
    be.u = sum(_interval(0.6)) / 2
    state = svc.ReplicaState(be, workers=4)
    sock, server = _serve(tmp_path, state)
    try:
        from reliquary.miner import replica_client

        th = threading.Thread(target=_parallel_calls, args=(sock, 4, "/m/rev1"))
        th.start()
        _time.sleep(0.03)
        assert replica_client.load(sock, "/m/rev2", timeout=10.0)
        th.join(10)
    finally:
        server.shutdown()
        server.server_close()
    assert be.compute_during_load == 0


def test_workers_depuis_lenv(monkeypatch):
    svc = _load_service()
    monkeypatch.setenv("REPLICA_WORKERS", "3")
    assert svc.ReplicaState(_FakeBackend(0.6)).workers == 3
    monkeypatch.delenv("REPLICA_WORKERS")
    assert svc.ReplicaState(_FakeBackend(0.6)).workers == 1


def test_client_reessaie_si_file_decoute_saturee(monkeypatch):
    from reliquary.miner import replica_client

    attempts = {"n": 0}
    resp = (json.dumps({"ok": True, "results": [
        {"ok": True, "pick": EOS, "cdf_miss": 0.0}]}) + "\n").encode()

    class _Sock:
        def __init__(self, *a):
            self.sent = False

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def settimeout(self, t):
            pass

        def connect(self, path):
            attempts["n"] += 1
            if attempts["n"] <= 2:
                raise BlockingIOError(11, "Resource temporarily unavailable")

        def sendall(self, b):
            self.sent = True

        def recv(self, n):
            if self.sent:
                self.sent = False
                return resp
            return b""

    monkeypatch.setattr(replica_client.socket, "socket", _Sock)
    res = replica_client.terminal_verdicts(
        "/x.sock", model_path="/m", randomness="ab", checkpoint_hash="r",
        prompt_idx=1, items=[{"rollout": 0, "prompt_len": 1, "tokens": [1, EOS]}],
        timeout=5.0)
    assert res == [{"ok": True, "pick": EOS, "cdf_miss": 0.0}]
    assert attempts["n"] == 3


# --------------------------------------------- profil du validateur (14/09)
# Lancée sans RELIQUARY_PROTOCOL_PROFILE, le code upstream retombe sur son profil
# par défaut ``qwen35-2b-auction-v2`` (T=0,6, top-k 20, domaine forced-seed-v2) :
# verdicts d'EOS et « réparations » faux toute la journée. Le service refuse
# désormais de démarrer sur un autre profil que celui attendu.
def test_garde_profil_refuse_un_autre_profil():
    svc = _load_service()
    fake = types.SimpleNamespace(PROTOCOL_PROFILE_ID="qwen35-2b-auction-v2",
                                 T_PROTO=0.6, TOP_K_PROTO=20, TOP_P_PROTO=0.95)
    err = svc.profile_error(fake, "qwen3-4b-base-dapo-reliquary-v1")
    assert err and "qwen35-2b-auction-v2" in err


def test_garde_profil_accepte_le_bon():
    svc = _load_service()
    fake = types.SimpleNamespace(PROTOCOL_PROFILE_ID="qwen3-4b-base-dapo-reliquary-v1",
                                 T_PROTO=1.0, TOP_K_PROTO=0, TOP_P_PROTO=1.0)
    assert svc.profile_error(fake, "qwen3-4b-base-dapo-reliquary-v1") is None


class _FakeBackendStats(_FakeBackend):
    def diagnose(self, row, token, u, *, stats=None):
        probs = torch.softmax(row.float(), -1)
        if stats is not None:
            hi = float(torch.cumsum(probs[: token + 1], -1)[-1])
            lo = hi - float(probs[token])
            stats["margin"] = round(min(u - lo, hi - u), 9)
            stats["p_tok"] = round(float(probs[token]), 9)
        return super().diagnose(row, token, u)


def test_marge_eos_journalisee_sans_changer_le_verdict(tmp_path, monkeypatch):
    dump = tmp_path / "marge.jsonl"
    monkeypatch.setenv("REPLICA_MARGIN_DUMP", str(dump))
    svc = _load_service()
    be = _FakeBackendStats(0.6)
    lo, hi = _interval(0.6)
    be.u = lo + 0.1
    r = svc.ReplicaState(be).handle(_req([{"rollout": 4, "prompt_len": 2,
                                           "tokens": [1, 2, 3, EOS]}]))
    assert r["results"][0]["ok"] is True and "margin" not in r["results"][0]
    row = json.loads(dump.read_text().splitlines()[0])
    assert row["rollout"] == 4 and row["prompt_idx"] == 3 and row["ok"] is True
    assert abs(row["margin"] - 0.1) < 1e-5 and abs(row["p_tok"] - 0.6) < 1e-5
    assert "t_wait" in r and "t_compute" in r


def test_pas_de_journal_sans_variable(tmp_path, monkeypatch):
    monkeypatch.delenv("REPLICA_MARGIN_DUMP", raising=False)
    svc = _load_service()
    be = _FakeBackendStats(0.6)
    lo, hi = _interval(0.6)
    be.u = (lo + hi) / 2
    r = svc.ReplicaState(be).handle(_req([{"rollout": 0, "prompt_len": 2,
                                           "tokens": [1, 2, EOS]}]))
    assert r["results"][0]["ok"] is True
    assert not list(tmp_path.iterdir())
