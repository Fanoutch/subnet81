"""Capture de ``X-Reliquary-Reject-Detail`` et ``Retry-After`` (PR #270, 16/09).

Côté validateur, ``batch_filled`` est un fourre-tout. La PR #270 (``a205b1c``,
live depuis le 16/09 ~12h) ajoute un en-tête qui sépare enfin ses causes :

- ``precommit_signature_busy`` + ``Retry-After: 1`` — le pool de vérification de
  signature (``MAX_PENDING_PROOF_QUEUE_DEPTH = 64``) est PARTAGÉ par tout le
  marché, donc saturé à l'ouverture : une saturation d'UNE seconde ;
- ``admission_queue_full`` — sur le reveal du corps, une vraie fermeture ;
- le motif de ``try_register_upload_precommit`` sinon.

Pourquoi c'est la donnée qui manque : on encaisse 520 ``batch_filled`` par
fenêtre, et un groupe refusé vers 37 s n'est réadmis qu'à 64,8 s. Sans savoir
quelle part est transitoire, on ne peut pas calibrer la cadence de réessai.

INSTRUMENTATION PURE : les valeurs sont attachées à la réponse comme ``_stage``
(attribut hors modèle, jamais sérialisé) puis écrites dans ``submits_v4.jsonl``.
Aucun format de log n'est modifié — ``slot_monitor_tick.py`` et
``harvest_window_timing.py`` parsent ces lignes.
"""
from __future__ import annotations

import asyncio

import httpx
import pytest

from reliquary.miner import submitter as sub
from reliquary.protocol.submission import RejectReason


# ------------------------------------------------------------------ extraction
class _R:
    def __init__(self, headers=None):
        self.headers = httpx.Headers(headers or {})


def test_entetes_absents():
    assert sub.reject_detail(_R()) == (None, None)


def test_entetes_presents_et_insensibles_a_la_casse():
    r = _R({"x-reliquary-reject-detail": "precommit_signature_busy",
            "retry-after": "1"})
    assert sub.reject_detail(r) == ("precommit_signature_busy", 1.0)


def test_retry_after_illisible_ou_negatif():
    assert sub.reject_detail(_R({"Retry-After": "soon"})) == (None, None)
    assert sub.reject_detail(_R({"Retry-After": "-2"})) == (None, None)


def test_objet_sans_entetes_ne_casse_rien():
    assert sub.reject_detail(object()) == (None, None)


# ------------------------------------------------------ chemin precommit complet
class _Resp:
    def __init__(self, payload, headers=None, status=200):
        self._payload = payload
        self.status_code = status
        self.headers = httpx.Headers(headers or {})

    def json(self):
        return self._payload


class _Client:
    def __init__(self, precommits, body=None):
        self._pre = list(precommits)
        self._body = body or ({"accepted": True, "reason": "accepted"}, {})

    async def post(self, url, content=None, headers=None, timeout=None):
        if url.endswith("/submit/precommit"):
            payload, hdr = self._pre.pop(0)
            return _Resp(payload, hdr)
        payload, hdr = self._body
        return _Resp(payload, hdr)

    async def aclose(self):
        return None


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


@pytest.fixture
def stub(monkeypatch):
    class _PC:
        drand_round = 12345

        def model_dump_json(self):
            return '{"stub":1}'

    monkeypatch.setattr(sub, "_build_precommit", lambda request, **kw: (b"body", _PC()))

    async def _sleep(s):
        return None

    monkeypatch.setattr(sub.asyncio, "sleep", _sleep)


class _Req:
    window_start = 1
    prompt_idx = 7
    drand_round = 12345

    def model_dump(self, mode=None):
        return {}


BUSY = {"X-Reliquary-Reject-Detail": "precommit_signature_busy", "Retry-After": "1"}


def test_rejet_precommit_porte_le_detail(stub, monkeypatch):
    monkeypatch.delenv(sub._PRECOMMIT_RETRY_DELAYS_ENV, raising=False)
    cli = _Client([({"accepted": False, "reason": "batch_filled"}, BUSY)])
    out = _run(sub._submit_with_precommit(
        "http://v", _Req(), client=cli, timeout=5, wallet=object(), randomness="r"))
    assert out.reason is RejectReason.BATCH_FILLED
    assert getattr(out, "_reject_detail") == "precommit_signature_busy"
    assert getattr(out, "_retry_after") == 1.0
    assert getattr(out, "_reject_details_seen") == {"precommit_signature_busy": 1}
    assert getattr(out, "_stage") == "precommit"          # l'existant tient


def test_rejet_sans_entete_attributs_neutres(stub, monkeypatch):
    monkeypatch.delenv(sub._PRECOMMIT_RETRY_DELAYS_ENV, raising=False)
    cli = _Client([({"accepted": False, "reason": "batch_filled"}, {})])
    out = _run(sub._submit_with_precommit(
        "http://v", _Req(), client=cli, timeout=5, wallet=object(), randomness="r"))
    assert getattr(out, "_reject_detail", None) is None
    assert getattr(out, "_retry_after", None) is None
    assert not getattr(out, "_reject_details_seen", None)


def test_reessais_internes_comptes_puis_acceptation(stub, monkeypatch):
    """Deux saturations d'une seconde, puis admis : la trace reste sur la réponse
    finale (c'est précisément le cas qu'on veut chiffrer)."""
    monkeypatch.setenv(sub._PRECOMMIT_RETRY_DELAYS_ENV, "0.3,0.3")
    monkeypatch.setattr(sub, "_drand_round_headroom_s", lambda now=None: 2.5)
    cli = _Client([
        ({"accepted": False, "reason": "batch_filled"}, BUSY),
        ({"accepted": False, "reason": "batch_filled"}, BUSY),
        ({"accepted": True, "reason": "accepted", "receipt_id": "r1"}, {}),
    ])
    out = _run(sub._submit_with_precommit(
        "http://v", _Req(), client=cli, timeout=5, wallet=object(), randomness="r"))
    assert out.accepted is True
    assert getattr(out, "_reject_details_seen") == {"precommit_signature_busy": 2}
    assert getattr(out, "_reject_detail", None) is None     # la réponse finale est OK


def test_rejet_du_corps_porte_le_detail(stub, monkeypatch):
    monkeypatch.delenv(sub._PRECOMMIT_RETRY_DELAYS_ENV, raising=False)
    cli = _Client(
        [({"accepted": True, "reason": "accepted", "receipt_id": "r1"}, {})],
        body=({"accepted": False, "reason": "batch_filled"},
              {"X-Reliquary-Reject-Detail": "admission_queue_full"}),
    )
    out = _run(sub._submit_with_precommit(
        "http://v", _Req(), client=cli, timeout=5, wallet=object(), randomness="r"))
    assert out.accepted is False
    assert getattr(out, "_reject_detail") == "admission_queue_full"
    assert getattr(out, "_reject_details_seen") == {"admission_queue_full": 1}


def test_jamais_serialise(stub, monkeypatch):
    monkeypatch.delenv(sub._PRECOMMIT_RETRY_DELAYS_ENV, raising=False)
    cli = _Client([({"accepted": False, "reason": "batch_filled"}, BUSY)])
    out = _run(sub._submit_with_precommit(
        "http://v", _Req(), client=cli, timeout=5, wallet=object(), randomness="r"))
    dumped = out.model_dump()
    assert not any("reject_detail" in k or "retry_after" in k for k in dumped)


# ------------------------------------------------------------ champs du dump
def test_champs_du_dump_uniquement_si_presents():
    from reliquary.miner import engine
    from reliquary.protocol.submission import BatchSubmissionResponse

    r = BatchSubmissionResponse(accepted=False, reason=RejectReason.BATCH_FILLED)
    assert engine.reject_detail_row_fields(r) == {}
    object.__setattr__(r, "_reject_detail", "precommit_signature_busy")
    object.__setattr__(r, "_retry_after", 1.0)
    object.__setattr__(r, "_reject_details_seen", {"precommit_signature_busy": 3})
    assert engine.reject_detail_row_fields(r) == {
        "reject_detail": "precommit_signature_busy",
        "retry_after": 1.0,
        "reject_details_seen": {"precommit_signature_busy": 3},
    }
