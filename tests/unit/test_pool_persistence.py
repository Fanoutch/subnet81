"""Tests for reliquary/miner/pool_persistence.py — disk-backed entry store.

The pool persistence layer is a thin wrapper over torch.save / torch.load
plus atomic-rename for crash safety. Three functions:

  * save_entry(entry, pool_dir) → Path  (atomic .tmp + rename)
  * delete_entry(path) → None           (idempotent)
  * load_pool(pool_dir, local_checkpoint_n) → list[dict]

All synchronous, called from inside the existing pool lock — no new
concurrency.
"""

import logging
import time
from pathlib import Path

import pytest
import torch

from reliquary.miner.pool_persistence import (
    delete_entry, load_pool, save_entry,
)


def _make_entry(prompt_idx: int, checkpoint_n: int = 14) -> dict:
    """Build a minimal entry shaped like the real bake output."""
    return {
        "prompt_idx": prompt_idx,
        "problem": {"prompt": "test", "answer": "1"},
        "rollouts": [
            {
                "all_tokens": [1, 2, 3, 4],
                "prompt_length": 2,
                "completion_text": "test",
                "hidden_states_cpu": torch.zeros(2, 8),
                "token_logprobs": [-0.1, -0.2],
                "reward": 1.0,
            }
        ],
        "checkpoint_n": checkpoint_n,
    }


def test_save_entry_writes_file(tmp_path: Path):
    entry = _make_entry(prompt_idx=42)
    path = save_entry(entry, tmp_path)
    assert path.parent == tmp_path
    assert path.suffix == ".pt"
    assert path.exists()
    assert "42_" in path.name  # prompt_idx prefix


def test_save_entry_atomic_no_tmp_leftover(tmp_path: Path):
    """Atomicity check: after save_entry returns, no .tmp file remains."""
    save_entry(_make_entry(prompt_idx=7), tmp_path)
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == []


def test_save_then_load_roundtrip(tmp_path: Path):
    original = _make_entry(prompt_idx=99, checkpoint_n=14)
    save_entry(original, tmp_path)
    loaded = load_pool(tmp_path, local_checkpoint_n=14)
    assert len(loaded) == 1
    assert loaded[0]["prompt_idx"] == 99
    assert loaded[0]["checkpoint_n"] == 14
    # Tensors round-trip.
    assert loaded[0]["rollouts"][0]["hidden_states_cpu"].shape == (2, 8)


def test_load_pool_returns_mtime_sorted(tmp_path: Path):
    """Older bakes drain first — load_pool sorts by file mtime ascending."""
    p1 = save_entry(_make_entry(prompt_idx=1), tmp_path)
    time.sleep(0.01)
    p2 = save_entry(_make_entry(prompt_idx=2), tmp_path)
    time.sleep(0.01)
    p3 = save_entry(_make_entry(prompt_idx=3), tmp_path)
    loaded = load_pool(tmp_path, local_checkpoint_n=14)
    assert [e["prompt_idx"] for e in loaded] == [1, 2, 3]


def test_delete_entry_removes_file(tmp_path: Path):
    path = save_entry(_make_entry(prompt_idx=5), tmp_path)
    assert path.exists()
    delete_entry(path)
    assert not path.exists()


def test_delete_entry_idempotent_on_missing(tmp_path: Path):
    """delete_entry must not raise on a non-existent file —
    races between concurrent restarts or already-fired entries
    are benign."""
    delete_entry(tmp_path / "ghost.pt")  # must not raise


def test_load_pool_skips_corrupt_files(tmp_path: Path, caplog):
    """A .pt file that fails torch.load is logged WARNING and skipped;
    healthy files in the same dir are still returned."""
    save_entry(_make_entry(prompt_idx=11), tmp_path)
    (tmp_path / "corrupt.pt").write_bytes(b"not a torch save")
    with caplog.at_level(logging.WARNING, logger="reliquary.miner.pool_persistence"):
        loaded = load_pool(tmp_path, local_checkpoint_n=14)
    assert len(loaded) == 1
    assert loaded[0]["prompt_idx"] == 11
    assert any("corrupt" in r.message for r in caplog.records)


def test_load_pool_keeps_stale_ckpt_but_warns(tmp_path: Path, caplog):
    """Entries baked under a different checkpoint are kept (optimistic)
    but tagged with a WARNING so the operator sees the count."""
    save_entry(_make_entry(prompt_idx=21, checkpoint_n=14), tmp_path)
    save_entry(_make_entry(prompt_idx=22, checkpoint_n=20), tmp_path)
    with caplog.at_level(logging.WARNING, logger="reliquary.miner.pool_persistence"):
        loaded = load_pool(tmp_path, local_checkpoint_n=20)
    assert len(loaded) == 2
    assert {e["prompt_idx"] for e in loaded} == {21, 22}
    assert any("stale" in r.message.lower() or "ckpt" in r.message.lower()
               for r in caplog.records)


def test_load_pool_handles_local_n_sentinel(tmp_path: Path):
    """At startup local_n=-1 (sentinel before first /state). All
    persisted entries differ from -1, but we keep them silently
    (the sentinel itself is the noise-source, not a real ckpt advance)."""
    save_entry(_make_entry(prompt_idx=31, checkpoint_n=14), tmp_path)
    loaded = load_pool(tmp_path, local_checkpoint_n=-1)
    assert len(loaded) == 1


def test_load_pool_empty_dir_returns_empty(tmp_path: Path):
    """Fresh launch: empty dir → []."""
    assert load_pool(tmp_path, local_checkpoint_n=-1) == []
