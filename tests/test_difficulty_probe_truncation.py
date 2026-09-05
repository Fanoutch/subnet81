"""The probe must record, per group, how many rollouts truncated (no EOS) —
those are the ones the miner drops at the 2600 cap. Diagnostic only; the label
already reflects them as failures."""
from __future__ import annotations

import importlib.util
import pathlib
import sys

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "difficulty_probe.py"


def _load_probe():
    # scripts/ is not a package (no __init__.py) → load by path, matching
    # tests/test_difficulty_probe_bft.py.
    spec = importlib.util.spec_from_file_location("difficulty_probe", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["difficulty_probe"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_count_truncated_counts_rollouts_without_eos():
    probe = _load_probe()
    eos = [99]
    rollouts = [
        [1, 2, 99],        # terminated (EOS present)
        [1, 2, 3, 4],      # truncated (no EOS)
        [99],              # terminated
        [5, 6, 7],         # truncated
    ]
    assert probe.count_truncated(rollouts, eos) == 2
    assert probe.count_truncated([], eos) == 0
    assert probe.count_truncated([[1, 2, 3]], []) == 0   # no eos ids → n/a
