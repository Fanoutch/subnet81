"""Batch the GRAIL proof forward across a group's rollouts.

Measured 2026-07-21: after batching phase-1 generation, the bake cost is
dominated by the proof stage — `_pre_bake_entry` runs ONE forward per rollout
(48 per 6-prompt bake, each over ~2500 tokens), ~29s per prompt. That keeps a
bake above the 100s collection window.

Rollouts in a group have DIFFERENT lengths, so batching needs right-padding plus
an attention mask. Right-padding is safe for a causal LM (a real token never
attends to a later pad), but the arithmetic is only equivalent up to float
reduction order — hence the separate GPU parity gate
(scripts/validate_proof_batch_parity.py), which is what actually authorises
this path. These tests pin the bookkeeping: shapes, mask, and per-sequence
slicing. Mixing those up yields a valid-looking proof for the WRONG rollout.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


class _FakeModel:
    """Returns hidden/logits that encode (row, position) so slicing is checkable."""

    def __init__(self, hidden=4, vocab=7):
        self.hidden, self.vocab = hidden, vocab
        self.calls = []

    def __call__(self, input_ids, attention_mask=None, **kw):
        raise AssertionError("proof forward must go through forward_single_layer")


def _fake_forward(model, input_ids, attention_mask, layer_index):
    """Stand-in for forward_single_layer: records the call, returns marked tensors."""
    model.calls.append(
        {"shape": tuple(input_ids.shape),
         "mask": None if attention_mask is None else attention_mask.tolist()}
    )
    b, s = input_ids.shape
    # hidden[r, t, :] = r*100 + t ; logits[r, t, v] = r*100 + t + v/100
    hs = torch.arange(b, dtype=torch.float32).view(b, 1, 1) * 100
    hs = hs + torch.arange(s, dtype=torch.float32).view(1, s, 1)
    hs = hs.expand(b, s, model.hidden).clone()
    lg = hs[:, :, :1] + torch.arange(model.vocab, dtype=torch.float32).view(1, 1, -1) / 100
    return hs, lg


SEQS = [
    [5, 6, 7, 8],          # len 4
    [5, 6, 9],             # len 3
    [5, 6, 7, 8, 1, 2],    # len 6  <- longest
]
PROMPT_LEN = 2


def _engine(model):
    from reliquary.miner.engine import MiningEngine

    e = MiningEngine.__new__(MiningEngine)
    e.hf_model = model
    e.proof_gpu = 0
    return e


def _run(monkeypatch, model):
    import reliquary.shared.forward as fwd

    monkeypatch.setattr(fwd, "forward_single_layer", _fake_forward)
    return _engine(model)._proof_forward_batch(SEQS, device="cpu")


def test_one_forward_for_the_whole_group(monkeypatch):
    """The win: 1 padded forward instead of len(SEQS)."""
    m = _FakeModel()
    _run(monkeypatch, m)
    assert len(m.calls) == 1
    assert m.calls[0]["shape"] == (len(SEQS), max(len(s) for s in SEQS))


def test_padding_is_masked_so_pads_cannot_be_attended(monkeypatch):
    """Without a mask the pads join the attention and corrupt every proof."""
    m = _FakeModel()
    _run(monkeypatch, m)
    mask = m.calls[0]["mask"]
    for row, seq in zip(mask, SEQS):
        assert row == [1] * len(seq) + [0] * (len(mask[0]) - len(seq))


def test_each_result_is_truncated_back_to_its_own_length(monkeypatch):
    """Carrying pad positions into the proof changes the committed sketch."""
    m = _FakeModel()
    out = _run(monkeypatch, m)
    assert [h.shape[0] for h, _ in out] == [len(s) for s in SEQS]


def test_rows_stay_in_input_order(monkeypatch):
    """Row r must be rollout r — a swap proves the wrong rollout."""
    m = _FakeModel()
    out = _run(monkeypatch, m)
    for r, (hidden, _) in enumerate(out):
        # _fake_forward marks hidden[r, t] = r*100 + t
        assert float(hidden[0, 0]) == r * 100
        assert float(hidden[1, 0]) == r * 100 + 1


def test_logits_are_returned_per_row_for_token_logprobs(monkeypatch):
    """token_logprobs are read off these logits; a shape slip shifts positions."""
    m = _FakeModel()
    out = _run(monkeypatch, m)
    for seq, (_, logits) in zip(SEQS, out):
        assert logits.shape[0] == len(seq)
        assert logits.shape[1] == m.vocab


def test_a_single_sequence_still_works(monkeypatch):
    """Degenerate case must not need a special caller path."""
    m = _FakeModel()
    import reliquary.shared.forward as fwd

    monkeypatch.setattr(fwd, "forward_single_layer", _fake_forward)
    out = _engine(m)._proof_forward_batch([[1, 2, 3]], device="cpu")
    assert len(out) == 1 and out[0][0].shape[0] == 3
