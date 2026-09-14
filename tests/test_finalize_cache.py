"""Finalize mis en cache au re-tir (14/09 soir).

Mesuré (fen 45916-45921) : 216-304 re-tirs par fenêtre, chacun refaisait le
finalize (engagements GRAIL sur GPU + signatures sr25519) : 113-228 s de GPU
par fenêtre, en concurrence avec génération, réparation et preuve des têtes.

Sûreté vérifiée dans le validateur 0a69244 : le dédoublonnage porte sur le
contenu en tokens (hash de rollout, groupe logique), jamais sur merkle_root ni
signatures ; l'empreinte de charge du précommit porte sur le corps complet
(nonce, round, signature d'enveloppe) qui change à chaque envoi.

Invariant : réutilisé seulement sous la MÊME randomness et le MÊME checkpoint.
"""
from __future__ import annotations

import asyncio
import types

from reliquary.miner import engine


def _eng(calls):
    eng = engine.MiningEngine.__new__(engine.MiningEngine)
    eng._local_hash = "c" * 40
    eng._submitted_env = {}
    eng.wallet = types.SimpleNamespace(hotkey=types.SimpleNamespace(ss58_address="5X"))
    eng._entry_env_name = lambda e: "opencodeinstruct"

    def fin(entry, randomness):
        calls.append(randomness)
        return [f"sub-{len(calls)}"], f"merkle-{len(calls)}"

    eng._finalize_pool_entry = fin
    seen = {}

    def build(subs, merkle, prompt_idx, state, hk, nonce):
        seen["subs"], seen["merkle"] = subs, merkle
        raise RuntimeError("stop après finalize")          # coupe l'envoi

    eng._build_signed_request_sync = build
    return eng, seen


def _st(rand):
    return types.SimpleNamespace(randomness=rand, window_n=7)


def test_reutilise_le_finalize_au_retir():
    calls = []
    eng, seen = _eng(calls)
    entry = {"prompt_idx": 3}
    asyncio.run(eng._submit_entry(entry, _st("r1"), "http://v", None, []))
    asyncio.run(eng._submit_entry(entry, _st("r1"), "http://v", None, []))
    assert calls == ["r1"]
    assert seen["merkle"] == "merkle-1"


def test_nouvelle_randomness_refait_le_finalize():
    calls = []
    eng, seen = _eng(calls)
    entry = {"prompt_idx": 3}
    asyncio.run(eng._submit_entry(entry, _st("r1"), "http://v", None, []))
    asyncio.run(eng._submit_entry(entry, _st("r2"), "http://v", None, []))
    assert calls == ["r1", "r2"]
    assert seen["merkle"] == "merkle-2"


def test_nouveau_checkpoint_refait_le_finalize():
    calls = []
    eng, seen = _eng(calls)
    entry = {"prompt_idx": 3}
    asyncio.run(eng._submit_entry(entry, _st("r1"), "http://v", None, []))
    eng._local_hash = "d" * 40
    asyncio.run(eng._submit_entry(entry, _st("r1"), "http://v", None, []))
    assert calls == ["r1", "r1"]
