"""Bucket index over the raw qwedsacf/competition_math dataset.

Loads (type, level) per prompt_idx once at startup; used by the selector
as a hyperprior for cold prompts.
"""
from __future__ import annotations


def _load_raw_dataset():
    """Load the raw HF dataset rows. Wrapped for mock-ability in tests."""
    import datasets as hf_datasets
    return hf_datasets.load_dataset(
        "qwedsacf/competition_math", split="train"
    )


class BucketIndex:
    """Maps prompt_idx -> (type, level) tuple, the bucket key."""

    def __init__(self) -> None:
        self._rows = list(_load_raw_dataset())
        self._n = len(self._rows)

    def __len__(self) -> int:
        return self._n

    def bucket_of(self, prompt_idx: int) -> tuple[str, str]:
        """Return (type, level) for *prompt_idx*. Wraps modulo dataset length."""
        idx = prompt_idx % self._n
        row = self._rows[idx]
        type_ = row.get("type") or "unknown"
        level = row.get("level") or "unknown"
        return (type_, level)
