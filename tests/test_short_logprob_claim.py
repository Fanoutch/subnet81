"""Logprobs revendiqués des rollouts COURTS = calcul du validateur (15/09).

Mesuré (fen 45970-45994) : les 4 ``logprob_mismatch`` portent sur des groupes
contenant un rollout d'UN token (EOS tiré en position 0, p ≈ 0,001). Parmi les
groupes réellement prouvés : ≥32 tokens 146/146 passés, 2-31 tokens 5/5, un
token 2 passés / 4 échoués. Sous 32 tokens (``CHALLENGE_K``) le validateur
compare TOUTES les positions (``_verify_short_logprob_claim`` : médiane de
``expm1(|Δ|)`` ≤ 0,10) ; avec une seule position aucune médiane n'amortit
l'écart numérique entre notre forward HF sdpa et le sien (bf16/fp32 seul :
|Δ| 0,003-0,125 sur ces EOS). La réplique exécute la pile et le code du
validateur : pour ces rollouts, la valeur revendiquée devient la sienne.
"""
from __future__ import annotations

import contextlib
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from reliquary.miner import engine, replica_client

ROOT = Path(__file__).resolve().parents[1]
EOS = 7


def _load_service():
    spec = importlib.util.spec_from_file_location(
        "replica_service", ROOT / "ops" / "replica_service.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------ service réplique
class _Backend:
    def __init__(self):
        self.calls = []

    def load(self, model_path):
        return {EOS}

    def chosen_logprobs(self, tokens, prompt_len):
        self.calls.append((list(tokens), prompt_len))
        return [-0.5 - 0.1 * j for j in range(len(tokens) - prompt_len)]


def test_service_renvoie_un_logprob_par_position_de_completion():
    svc = _load_service()
    be = _Backend()
    state = svc.ReplicaState(be)
    out = state.handle({"op": "chosen_logprobs", "model_path": "/m/rev",
                        "items": [{"prompt_len": 2, "tokens": [1, 2, EOS]},
                                  {"prompt_len": 2, "tokens": [1, 2, 3, 4, EOS]}]})
    assert out["ok"] is True
    assert [[round(v, 6) for v in r] for r in out["results"]] == [[-0.5], [-0.5, -0.6, -0.7]]


def test_service_item_sans_completion_renvoie_none():
    svc = _load_service()
    be = _Backend()
    out = svc.ReplicaState(be).handle({
        "op": "chosen_logprobs", "model_path": "/m/rev",
        "items": [{"prompt_len": 3, "tokens": [1, 2, 3]}]})
    assert out["results"] == [None]
    assert be.calls == []


def test_backend_upstream_imite_le_validateur(monkeypatch):
    """log(softmax(lignes.float() / T)[token]) sur les lignes t-1 du forward,
    projetées ensemble comme ``_LazyLogitRows.index_select``."""
    svc = _load_service()
    import reliquary.shared.forward as fwd
    import reliquary.constants as consts

    monkeypatch.setattr(consts, "T_PROTO", 1.0)
    hidden = torch.randn(1, 5, 4)
    monkeypatch.setattr(fwd, "forward_single_layer",
                        lambda model, inp, mask, layer, materialize_logits=True:
                        (hidden, None))
    lm = torch.nn.Linear(4, 10, bias=False)
    be = svc.UpstreamBackend()
    be.model, be.lm_head = object(), lm
    toks = [1, 2, 3, 4, EOS]
    got = be.chosen_logprobs(toks, 3, device="cpu")
    with torch.no_grad():
        rows = lm(hidden[0].index_select(0, torch.tensor([2, 3])))
    ref = torch.log_softmax(rows.float(), -1)
    assert len(got) == 2
    assert math.isclose(got[0], ref[0, 4].item(), abs_tol=1e-5)
    assert math.isclose(got[1], ref[1, EOS].item(), abs_tol=1e-5)


# --------------------------------------------------------------------- client
def test_client_rend_none_si_nombre_incoherent(monkeypatch):
    monkeypatch.setattr(replica_client, "_call",
                        lambda s, req, t: {"ok": True, "results": [[-1.0]]})
    assert replica_client.chosen_logprobs("/s", model_path="/m", items=[{}, {}]) is None
    monkeypatch.setattr(replica_client, "_call",
                        lambda s, req, t: {"ok": True, "results": [[-1.0], None]})
    assert replica_client.chosen_logprobs("/s", model_path="/m", items=[{}, {}]) == [[-1.0], None]


# ------------------------------------------------------------ preuve du mineur
def _proof_engine(monkeypatch):
    import reliquary.shared.forward as fwd
    from reliquary.environment import code_grader

    eng = engine.MiningEngine.__new__(engine.MiningEngine)
    eng.hf_model = SimpleNamespace()              # chemin legacy (pas de lm_head)
    eng.tokenizer = SimpleNamespace(decode=lambda toks: "x")
    eng._eos_ids = [EOS]
    eng._loaded_checkpoint_path = "/snap/rev"

    def fake_forward(model, inp, mask, layer, materialize_logits=True):
        n = inp.shape[1]
        return torch.zeros(1, n, 4), torch.zeros(1, n, 10)   # log p = log 0,1

    monkeypatch.setattr(fwd, "forward_single_layer", fake_forward)
    monkeypatch.setattr(code_grader, "fork_gpu_guard", contextlib.nullcontext)
    monkeypatch.setenv("RELIQUARY_REPLICA_SOCKET", "/tmp/replica.sock")
    monkeypatch.delenv("RELIQUARY_REPLICA_SHORT_LOGPROBS", raising=False)
    return eng


LOCAL = math.log(0.1)


def test_rollout_court_revendique_la_valeur_de_la_replique(monkeypatch):
    calls = []

    def fake(sock, *, model_path, items, timeout=60.0):
        calls.append((model_path, items))
        return [[-7.07] * (len(it["tokens"]) - it["prompt_len"]) for it in items]

    monkeypatch.setattr(replica_client, "chosen_logprobs", fake)
    eng = _proof_engine(monkeypatch)
    long_toks = [1, 2] + [3] * 40 + [EOS]
    gens = [{"tokens": [1, 2, EOS], "prompt_length": 2},
            {"tokens": long_toks, "prompt_length": 2},
            {"tokens": [1, 2, 3, 3, EOS], "prompt_length": 2}]
    out = eng._proof_rollouts(gens, texts=["x"] * 3, device="cpu")
    assert out[0]["token_logprobs"] == [-7.07]
    assert out[2]["token_logprobs"] == [-7.07] * 3
    assert all(math.isclose(v, LOCAL, abs_tol=1e-6) for v in out[1]["token_logprobs"])
    # seuls les rollouts < CHALLENGE_K partent à la réplique
    assert calls[0][0] == "/snap/rev"
    assert [len(it["tokens"]) for it in calls[0][1]] == [3, 5]


def test_replique_indisponible_garde_la_valeur_locale(monkeypatch):
    monkeypatch.setattr(replica_client, "chosen_logprobs", lambda *a, **k: None)
    eng = _proof_engine(monkeypatch)
    out = eng._proof_rollouts([{"tokens": [1, 2, EOS], "prompt_length": 2}],
                              texts=["x"], device="cpu")
    assert math.isclose(out[0]["token_logprobs"][0], LOCAL, abs_tol=1e-6)


def test_reponse_invalide_pour_un_rollout_garde_la_valeur_locale(monkeypatch):
    monkeypatch.setattr(replica_client, "chosen_logprobs",
                        lambda *a, **k: [None, [-1.0, -2.0]])
    eng = _proof_engine(monkeypatch)
    gens = [{"tokens": [1, 2, EOS], "prompt_length": 2},
            {"tokens": [1, 2, EOS], "prompt_length": 2}]      # longueur incohérente
    out = eng._proof_rollouts(gens, texts=["x", "x"], device="cpu")
    assert math.isclose(out[0]["token_logprobs"][0], LOCAL, abs_tol=1e-6)
    assert math.isclose(out[1]["token_logprobs"][0], LOCAL, abs_tol=1e-6)


def test_coupure_par_variable(monkeypatch):
    calls = []
    monkeypatch.setattr(replica_client, "chosen_logprobs",
                        lambda *a, **k: calls.append(1) or [[-1.0]])
    eng = _proof_engine(monkeypatch)
    monkeypatch.setenv("RELIQUARY_REPLICA_SHORT_LOGPROBS", "0")
    out = eng._proof_rollouts([{"tokens": [1, 2, EOS], "prompt_length": 2}],
                              texts=["x"], device="cpu")
    assert calls == []
    assert math.isclose(out[0]["token_logprobs"][0], LOCAL, abs_tol=1e-6)


def test_sans_socket_aucun_appel(monkeypatch):
    calls = []
    monkeypatch.setattr(replica_client, "chosen_logprobs",
                        lambda *a, **k: calls.append(1) or [[-1.0]])
    eng = _proof_engine(monkeypatch)
    monkeypatch.delenv("RELIQUARY_REPLICA_SOCKET", raising=False)
    eng._proof_rollouts([{"tokens": [1, 2, EOS], "prompt_length": 2}],
                        texts=["x"], device="cpu")
    assert calls == []
