import torch

from reliquary.miner.bft import bft_assemble_rollouts

EOS = {2}
CLOSE = {7}
FORCE = [7, 8]


class _Model:
    device = "cpu"

    def generate(self, rows, attention_mask=None, max_new_tokens=0, **kw):
        # append a boxed answer then EOS to every primed row
        return torch.tensor([r.tolist() + [50, 2] for r in rows])


def _p1():
    # row0: EOS in phase1 (…,2); row1: </think>(7) no EOS; row2: neither
    return torch.tensor([[1, 1, 1, 9, 2, 0],
                         [1, 1, 1, 7, 9, 9],
                         [1, 1, 1, 9, 9, 9]])


def test_three_bft_cases():
    out = bft_assemble_rollouts(model=_Model(), phase1_tensor=_p1(),
        prompt_tokens=[1, 1, 1], think_close_ids=CLOSE, force_ids=FORCE,
        eos_ids=EOS, answer_budget=4, randomness="r", hotkey="h", prompt_idx=0,
        checkpoint_hash="c", gen_kwargs={})
    # row0: finished on EOS in phase-1, trimmed at first EOS, not forced
    assert out[0]["forced"] is False and out[0]["tokens"][-1] == 2
    assert "force_span" not in out[0]
    # row1: closed </think> but no EOS → phase-2 answer, NO force injected
    assert out[1]["forced"] is False and "force_span" not in out[1]
    # row2: neither → force injected, span length == len(FORCE)
    assert out[2]["forced"] is True
    assert out[2]["force_span"][1] - out[2]["force_span"][0] == 2
