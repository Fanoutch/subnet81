"""The probe's math generation must follow the v7 BFT flow, not free-run.

Labels (k/m, in_zone) are only meaningful if the probe generates the way the
protocol does: v7 sampler (T/top_p/top_k), phase-1 capped at EXACTLY
BFT_THINKING_BUDGET, then — for rollouts that didn't EOS — either a natural
</think> continuation or the injected FORCE template, followed by a phase-2
answer capped at BFT_ANSWER_BUDGET. A free-run probe over-succeeds on
long-thinking prompts and mislabels them as easy.
"""
import importlib.util
import pathlib
import sys

import pytest

from reliquary.constants import (
    BFT_ANSWER_BUDGET,
    BFT_THINKING_BUDGET,
    TOP_K_PROTO,
    TOP_P_PROTO,
)

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "difficulty_probe.py"


def _load_probe():
    spec = importlib.util.spec_from_file_location("difficulty_probe", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["difficulty_probe"] = mod
    spec.loader.exec_module(mod)
    return mod


EOS = 2
CLOSE = 7          # atomic </think> id
FORCE_TAIL = [8, 9]  # encoded "\n\nFinal Answer: \boxed{"


class _Tok:
    def convert_tokens_to_ids(self, s):
        assert s == "</think>"
        return CLOSE

    def encode(self, s, add_special_tokens=False):
        assert s.startswith("\n\nFinal Answer")
        return list(FORCE_TAIL)


class _Backend:
    """Records generate_multi calls; scripted phase-1 then phase-2 output."""

    def __init__(self, phase1):
        self.calls = []
        self._phase1 = phase1

    def generate_multi(self, prompts_token_ids, n, temperature, top_p, top_k,
                       max_tokens, stop_token_ids):
        self.calls.append({
            "prompts": [list(p) for p in prompts_token_ids], "n": n,
            "temperature": temperature, "top_p": top_p, "top_k": top_k,
            "max_tokens": max_tokens, "stop_token_ids": list(stop_token_ids),
        })
        if len(self.calls) == 1:
            return self._phase1
        # phase-2: one boxed answer + EOS (+ trailing garbage that must be
        # trimmed at first EOS) per primed input.
        return [[[42, EOS, 99]] for _ in prompts_token_ids]


@pytest.fixture()
def probe():
    return _load_probe()


def test_phase1_uses_v7_sampler_and_exact_thinking_budget(probe):
    backend = _Backend(phase1=[[[5, EOS]]])
    probe.bft_generate_math(backend, _Tok(), [[1, 1]], m=1,
                            temperature=0.6, eos_ids={EOS})
    p1 = backend.calls[0]
    assert p1["top_p"] == TOP_P_PROTO
    assert p1["top_k"] == TOP_K_PROTO
    assert p1["max_tokens"] == BFT_THINKING_BUDGET  # exact, NOT min(cap, ...)
    assert p1["temperature"] == 0.6
    assert p1["n"] == 1


def test_eos_rollout_kept_as_is_no_phase2(probe):
    backend = _Backend(phase1=[[[5, EOS, 66]]])  # EOS then garbage
    out = probe.bft_generate_math(backend, _Tok(), [[1, 1]], m=1,
                                  temperature=0.6, eos_ids={EOS})
    assert out == [[[5, EOS]]]          # trimmed at first EOS, untouched
    assert len(backend.calls) == 1      # no phase-2 for finished rollouts


def test_unclosed_rollout_gets_force_template_then_answer(probe):
    backend = _Backend(phase1=[[[5, 5, 5]]])  # no EOS, no </think> → FORCE
    out = probe.bft_generate_math(backend, _Tok(), [[1, 1]], m=1,
                                  temperature=0.6, eos_ids={EOS})
    p2 = backend.calls[1]
    # phase-2 input = prompt + phase-1 + injected force ids
    assert p2["prompts"] == [[1, 1, 5, 5, 5, CLOSE] + FORCE_TAIL]
    assert p2["max_tokens"] == BFT_ANSWER_BUDGET
    assert p2["n"] == 1
    assert p2["top_p"] == TOP_P_PROTO and p2["top_k"] == TOP_K_PROTO
    # completion = phase-1 + force ids + phase-2 tail trimmed at first EOS
    assert out == [[[5, 5, 5, CLOSE] + FORCE_TAIL + [42, EOS]]]


def test_naturally_closed_rollout_continues_without_force(probe):
    backend = _Backend(phase1=[[[5, CLOSE, 5]]])  # </think> seen, no EOS
    out = probe.bft_generate_math(backend, _Tok(), [[1, 1]], m=1,
                                  temperature=0.6, eos_ids={EOS})
    p2 = backend.calls[1]
    assert p2["prompts"] == [[1, 1, 5, CLOSE, 5]]  # NO force ids injected
    assert out == [[[5, CLOSE, 5, 42, EOS]]]


def test_mixed_batch_routes_each_rollout_independently(probe):
    backend = _Backend(phase1=[
        [[5, EOS], [5, 5, 5]],   # prompt 0: finished + forced
        [[5, CLOSE, 5], [6, EOS]],  # prompt 1: natural close + finished
    ])
    out = probe.bft_generate_math(backend, _Tok(), [[1, 1], [3, 3]], m=2,
                                  temperature=0.6, eos_ids={EOS})
    assert out[0][0] == [5, EOS]
    assert out[0][1] == [5, 5, 5, CLOSE] + FORCE_TAIL + [42, EOS]
    assert out[1][0] == [5, CLOSE, 5, 42, EOS]
    assert out[1][1] == [6, EOS]
    # single phase-2 batch for the two unfinished rollouts
    assert len(backend.calls) == 2
    assert len(backend.calls[1]["prompts"]) == 2
