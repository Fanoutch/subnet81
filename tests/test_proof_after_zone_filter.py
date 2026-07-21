"""Grade first, filter on sigma, ONLY then compute the GRAIL proof forward.

Measured 2026-07-21: `_pre_bake_entry` ran the proof forward (the ~3.4 s/rollout,
~27 s/prompt, 91% of cycle time) for every rollout BEFORE checking the sigma
zone — then discarded ~99.8% of groups as out-of-zone. So almost all proof
compute was thrown away.

compute_reward depends only on the generated tokens, never on the proof, so the
reward/sigma decision can be made first and the expensive forward skipped for
out-of-zone groups entirely. That is what turns ~3 submittable prompts per
window into many more.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


class _CountingEnv:
    """Grades cheaply; records how many times it was asked."""

    name = "openmathinstruct"

    def __init__(self, rewards):
        self._rewards = rewards
        self._i = 0
        self.reward_calls = 0

    def compute_reward(self, problem, completion):
        self.reward_calls += 1
        r = self._rewards[self._i]
        self._i += 1
        return r


def _engine(env, proof_counter):
    from reliquary.miner.engine import MiningEngine
    import reliquary.miner.engine as eng

    e = MiningEngine.__new__(MiningEngine)
    e.proof_gpu = 0
    e.hf_model = object()
    e.tokenizer = SimpleNamespace(decode=lambda toks: "x")
    e._local_hash = "ckpt"
    e._cached_randomness = "ab" * 32
    e._verifier = None

    # generations: 8 rollouts, tokens present, prompt_length set
    e._gens = [
        {"tokens": [1, 2, 3, 4], "prompt_length": 2, "forced": False,
         "force_span": None}
        for _ in range(8)
    ]
    e._generate_m_rollouts = lambda *a, **k: e._gens

    def _fake_forward(model, input_ids, mask, layer):
        proof_counter[0] += 1
        s = input_ids.shape[1]
        return torch.zeros(1, s, 4), torch.zeros(1, s, 7)

    import reliquary.shared.forward as fwd
    fwd.forward_single_layer = _fake_forward
    eng.forward_single_layer = _fake_forward
    return e


UNANIMOUS = [1.0] * 8      # k=8, sigma 0 -> out of zone
PAYABLE = [1.0] * 4 + [0.0] * 4   # k=4, sigma 0.5 -> in zone


def test_out_of_zone_group_skips_the_proof_forward_entirely(monkeypatch):
    """The whole point: no GRAIL compute for a group we will discard."""
    env = _CountingEnv(UNANIMOUS)
    proof = [0]
    e = _engine(env, proof)
    out = e._pre_bake_entry(0, {"prompt": "p"}, 1, env)
    assert out is None                     # dropped out-of-zone
    assert env.reward_calls == 8           # graded all 8 (cheap)
    assert proof[0] == 0, "proof forward ran for a discarded group"


def test_in_zone_group_still_computes_every_proof(monkeypatch):
    """A payable group must be fully proven — no rollout skipped."""
    env = _CountingEnv(PAYABLE)
    proof = [0]
    e = _engine(env, proof)
    out = e._pre_bake_entry(0, {"prompt": "p"}, 1, env)
    assert out is not None
    assert proof[0] == 8, "an in-zone group must prove all 8 rollouts"
    assert len(out["rollouts"]) == 8
