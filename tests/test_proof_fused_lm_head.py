"""Fix « preuve fusée » (2026-08-19) : ne jamais matérialiser [seq, vocab].

Le forward de preuve construisait les logits pleins (~300 Mo-2,4 Go/rollout
en bf16) dont SEULES les lignes de complétion sont lues par
``_chunked_chosen_logprobs``. Le chemin fusé projette le lm_head par tranche
de lignes (technique upstream ``_LazyLogitRows``, mergée dans main) :
mémoire bornée à ``chunk × vocab``, et les lignes du prompt ne sont plus
projetées du tout.

Parité exigée : logprobs identiques au chemin legacy (mêmes maths, softmax
fp32 par tranche inchangé — seule la projection est déplacée), hidden states
strictement intouchés (les commitments en dépendent bit à bit).
"""
import math

import pytest

torch = pytest.importorskip("torch")

from reliquary.miner.engine import (  # noqa: E402
    _chunked_chosen_logprobs,
    _chunked_chosen_logprobs_fused,
)
from reliquary.shared.forward import forward_single_layer  # noqa: E402


def _mk(seq=97, hidden=32, vocab=211, seed=0):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(seq, hidden, generator=g)
    lm = torch.nn.Linear(hidden, vocab, bias=False)
    with torch.no_grad():
        lm.weight.copy_(torch.randn(vocab, hidden, generator=g))
    toks = torch.randint(0, vocab, (seq,), generator=g).tolist()
    return h, lm, toks


def test_fused_matches_legacy_exact():
    h, lm, toks = _mk()
    with torch.no_grad():
        logits = lm(h)
    for plen in (1, 40, 95):
        legacy = _chunked_chosen_logprobs(logits, toks, plen, chunk=16)
        fused = _chunked_chosen_logprobs_fused(h, lm, toks, plen, chunk=16)
        assert len(legacy) == len(fused) == len(toks) - plen
        for a, b in zip(legacy, fused):
            assert math.isfinite(b)
            assert abs(a - b) < 1e-6, (a, b)


def test_fused_temp_and_probs_paths():
    h, lm, toks = _mk(seed=3)
    with torch.no_grad():
        logits = lm(h)
    legacy = _chunked_chosen_logprobs(
        logits, toks, 10, temp=0.6, as_probs=True, chunk=8)
    fused = _chunked_chosen_logprobs_fused(
        h, lm, toks, 10, temp=0.6, as_probs=True, chunk=8)
    for a, b in zip(legacy, fused):
        assert abs(a - b) < 1e-6


def test_fused_empty_completion():
    h, lm, toks = _mk(seq=5)
    assert _chunked_chosen_logprobs_fused(h, lm, toks, 5) == []


class _Base(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.emb = torch.nn.Embedding(64, hidden)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        class _Out:
            pass
        o = _Out()
        o.last_hidden_state = self.emb(input_ids)
        return o


class _Model(torch.nn.Module):
    base_model_prefix = "m"

    def __init__(self, hidden=16, vocab=64):
        super().__init__()
        self.m = _Base(hidden)
        self.lm_head = torch.nn.Linear(hidden, vocab, bias=False)


def test_forward_skip_logits_same_hidden():
    model = _Model()
    ids = torch.randint(0, 64, (1, 33))
    h1, logits = forward_single_layer(model, ids, None, -1)
    h2, none = forward_single_layer(
        model, ids, None, -1, materialize_logits=False)
    assert none is None
    assert logits is not None
    assert torch.equal(h1, h2)
