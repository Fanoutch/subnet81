"""One-time build of the id->cases artifact for local opencode grading.

Joins the public miner mirror (R0mAI/opencodeinstruct-prompts) to the public
source nvidia/OpenCodeInstruct by ``id``, extracting the unit_tests (assertion
strings = args+expected). Writes a compact ``{id: [assertion, ...]}`` JSON.
The full NVIDIA dataset is streamed (never fully downloaded).

Verified 2026-06-11: id-join works (mirror ids present in NVIDIA with matching
input + unit_tests). See docs/superpowers/specs/2026-06-11-miner-multi-env-design.md.
"""
from __future__ import annotations

import json
import os
import sys

MIRROR_REPO = "R0mAI/opencodeinstruct-prompts"
SOURCE_REPO = "nvidia/OpenCodeInstruct"
OUT_PATH = os.environ.get("RELIQUARY_OCI_CASES_PATH", "data/opencode_cases.json")


def cases_from_unit_tests(raw) -> list[str]:
    """Parse the string-encoded unit_tests list into clean assertion strings.

    Returns [] on any malformed input (never raises).
    """
    if not isinstance(raw, str):
        return []
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [s.strip() for s in parsed if isinstance(s, str) and s.strip()]


def build(out_path: str = OUT_PATH, scan_cap: int = 5_000_000) -> dict[str, list[str]]:
    """Stream NVIDIA, recover cases for every id in the mirror, write artifact."""
    from datasets import load_dataset

    mirror = load_dataset(MIRROR_REPO, split="train")
    want = {r["id"] for r in mirror}
    print(f"mirror: {len(want)} ids", file=sys.stderr)

    src = load_dataset(SOURCE_REPO, split="train", streaming=True)
    out: dict[str, list[str]] = {}
    scanned = 0
    for row in src:
        scanned += 1
        rid = row.get("id")
        if rid in want and rid not in out:
            cases = cases_from_unit_tests(row.get("unit_tests"))
            if cases:
                out[rid] = cases
        if len(out) >= len(want) or scanned >= scan_cap:
            break
    print(f"scanned {scanned}, recovered {len(out)}/{len(want)}", file=sys.stderr)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f)
    return out


if __name__ == "__main__":
    build()
