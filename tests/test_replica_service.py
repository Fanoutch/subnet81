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
