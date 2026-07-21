"""Disk-backed entry store for the miner's pre-baked rollout pool.

Survives miner restarts: entries baked under one launch are reloaded
on the next, so a restart costs no forfeit window beyond cold-start
vLLM warmup.

Layout: one .pt file per entry under ``pool_dir``, named
``<prompt_idx>_<timestamp_ns>.pt``. Saves are atomic via .tmp + rename.
Loads scan the directory and sort by mtime so older bakes drain first.
Corrupt files are logged and skipped (left on disk for forensic
inspection). Entries baked under a different checkpoint than the
current local_n are kept (optimistic policy — matches the live
RELIQUARY_DROP_POOL_ON_CKPT=0 default).

All functions are synchronous and called from inside the existing
``self._pool_lock`` in engine.py — no new concurrency surface.
"""

import logging
import os
import time
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def save_entry(entry: dict, pool_dir: Path) -> Path:
    """Atomically persist a baked entry. Returns the final path.

    The path is stored back on the entry as ``entry["_persist_path"]``
    so the fire path can locate the file to delete on success.
    """
    pool_dir = Path(pool_dir)
    pool_dir.mkdir(parents=True, exist_ok=True)
    prompt_idx = int(entry["prompt_idx"])
    ts_ns = time.time_ns()
    final = pool_dir / f"{prompt_idx}_{ts_ns}.pt"
    tmp = pool_dir / f"{prompt_idx}_{ts_ns}.pt.tmp"
    torch.save(entry, tmp)
    os.rename(tmp, final)  # atomic on POSIX
    entry["_persist_path"] = final
    return final


def delete_entry(path: Path) -> None:
    """Remove a persisted entry. Idempotent: missing file is silently
    skipped (races between a concurrent restart's reload and an
    already-fired entry are benign)."""
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass


def load_pool(pool_dir: Path, local_checkpoint_n: int) -> list[dict]:
    """Scan ``pool_dir`` and torch.load every ``*.pt`` file.

    Returns entries sorted by file mtime ascending so older bakes are
    drained first by the fire path. Corrupt files are logged at WARNING
    and skipped. Entries whose ``checkpoint_n`` differs from
    ``local_checkpoint_n`` are KEPT (optimistic) but tagged with a
    single summary WARNING so operators see the count. When
    ``local_checkpoint_n == -1`` (sentinel before first /state) the
    staleness warning is suppressed — the sentinel itself is the
    mismatch source, not a real ckpt advance.

    Each loaded entry has ``entry["_persist_path"]`` set to the file
    path so the fire path can locate and delete it on success.
    """
    pool_dir = Path(pool_dir)
    if not pool_dir.exists():
        return []

    files = sorted(pool_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    loaded: list[dict] = []
    stale_count = 0
    for f in files:
        try:
            entry = torch.load(f, map_location="cpu", weights_only=False)
        except Exception as e:
            logger.warning(
                "pool_persistence: skipping corrupt file %s (%s)", f, e,
            )
            continue
        entry["_persist_path"] = f
        if local_checkpoint_n != -1 and entry.get("checkpoint_n") != local_checkpoint_n:
            stale_count += 1
        loaded.append(entry)

    if stale_count:
        logger.warning(
            "pool_persistence: %d/%d reloaded entries have stale ckpt "
            "(local=%d); keeping optimistically",
            stale_count, len(loaded), local_checkpoint_n,
        )
    return loaded
