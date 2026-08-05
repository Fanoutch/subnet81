"""Smoke test the training CLI end-to-end on a tiny synthetic probe file."""
from __future__ import annotations

import json
import os
import subprocess
import sys


def test_cli_trains_and_writes_model(tmp_path):
    hard = [1, 1, 0, 0, 0, 0, 0, 0]
    easy = [1, 1, 1, 1, 1, 1, 1, 1]
    rows = []
    for _ in range(8):
        rows.append({"prompt": "use recursion here", "rewards": hard,
                     "in_zone": True, "split": "train"})
        rows.append({"prompt": "use a simple loop", "rewards": easy,
                     "in_zone": False, "split": "train"})
    rows.append({"prompt": "use recursion here", "rewards": hard,
                 "in_zone": True, "split": "test"})
    rows.append({"prompt": "use a simple loop", "rewards": easy,
                 "in_zone": False, "split": "test"})
    in_path = tmp_path / "labeled.jsonl"
    in_path.write_text("\n".join(json.dumps(r) for r in rows))
    out_path = tmp_path / "model.json"

    res = subprocess.run(
        [sys.executable, "scripts/train_prompt_predictor.py",
         "--in", str(in_path), "--out", str(out_path), "--k", "1"],
        capture_output=True, text=True, cwd=".",
        env={**os.environ, "PYTHONPATH": "."},  # so `import reliquary` resolves
    )
    assert res.returncode == 0, res.stderr
    model = json.loads(out_path.read_text())
    assert "word_priors" in model and "df" in model
    # payable word learned above the unanimous one
    assert model["word_priors"]["recursion"] > model["word_priors"]["loop"]
    assert "top-" in res.stdout  # the lift line printed
