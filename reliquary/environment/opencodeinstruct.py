"""OpenCodeInstruct code-execution environment (miner side).

Aligned with the validator's curated env (origin/main): loads the reproducible
curated subset R0mAI/opencodeinstruct-curated via VirtualParquetDataset, appends
the grader's function-call contract to the prompt (so prompt tokens — GRAIL-bound
— match the validator), and grades completions LOCALLY against the embedded
structured_cases with the validator's exact semantics (code_grader). The local
reward is used to compute sigma and pre-select in-zone groups; the validator
re-grades authoritatively (validator_authoritative_reward=True).

_extract_python / _load_dataset / _contract_instruction are copied VERBATIM from
the validator's reliquary/environment/opencodeinstruct.py so prompt + extraction
are byte-identical.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import ClassVar

from reliquary.constants import GRADER_EVAL_TIMEOUT_SECONDS
from reliquary.environment.code_grader import grade_structured_cases


# ---------------------------------------------------------------------------
# Code extraction + dataset + contract (VERBATIM from validator)
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(
    r"(```|~~~)(?:python3?|py)?\s*\n(.*?)\n\1",
    re.DOTALL,
)


def _extract_python(completion: str) -> str:
    """Extract Python code from a model completion (last fenced block wins)."""
    if not completion:
        return ""
    matches = _FENCE_RE.findall(completion)
    if matches:
        return matches[-1][1]
    return completion


def _load_dataset(repo: str, revision: str):
    """Lazy virtual-parquet view of the curated dataset.

    A ``save_to_disk`` directory path is loaded eagerly (offline / fixtures);
    a ``owner/name`` repo id is wrapped in a ``VirtualParquetDataset`` so only
    the row-groups a window touches are fetched — no multi-GB bulk download.
    """
    path = Path(repo).expanduser()
    if path.exists() and (path / "dataset_info.json").exists():
        import datasets as hf
        return hf.load_from_disk(str(path))
    from reliquary.environment.virtual_parquet import VirtualParquetDataset
    return VirtualParquetDataset(repo, revision, columns=["input", "structured_cases"])


def _contract_instruction(cases: list[dict]) -> str:
    """The grader calls a named function and checks its RETURN value, but the raw
    prompts are stdin/stdout-framed and rarely name the function. Append the exact
    contract (name + "return, don't print") derived from the cases so the model
    writes a callable returning function instead of guessing. Empty for non-
    function entries (nothing to pin)."""
    for case in cases:
        entry = case.get("entry") or {}
        name = entry.get("name")
        if entry.get("kind") == "function" and name:
            nargs = len(case.get("args") or [])
            args = "argument" if nargs == 1 else "arguments"
            return (
                f"\n\nWrite your solution as a Python function named `{name}` that "
                f"takes {nargs} {args} and returns the result; do not read from "
                f"stdin or print."
            )
    return ""


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------


class OpenCodeInstructEnvironment:
    """nvidia/OpenCodeInstruct curated subset — Python codegen, continuous reward.

    Reward is passed/total over the embedded structured_cases (continuous in
    [0,1]); the σ-zone selection uses the continuous branch (see engine
    _try_select). The validator re-grades authoritatively.
    """

    name: str = "opencodeinstruct"
    validator_authoritative_reward: ClassVar[bool] = True
    continuous_reward: ClassVar[bool] = True  # dispatch: σ-continuous selection

    _dataset_cache: ClassVar = {}
    _CURATED_REPO: ClassVar[str] = "R0mAI/opencodeinstruct-curated"
    _CURATED_REVISION: ClassVar[str] = "d3caaefc3b46f8642b251f9efaeccf0d1e95b0a7"

    def __init__(self) -> None:
        repo = os.environ.get("RELIQUARY_OCI_REPO", self._CURATED_REPO)
        revision = os.environ.get("RELIQUARY_OCI_REVISION", self._CURATED_REVISION)
        cache = OpenCodeInstructEnvironment._dataset_cache
        if isinstance(cache, dict):
            key = (repo, revision)
            if key not in cache:
                cache[key] = _load_dataset(repo, revision)
            self._dataset = cache[key]
        else:
            # Tests may monkeypatch _dataset_cache directly with a fake dataset.
            self._dataset = cache
        self._cases_by_id: dict[str, list[dict]] = {}

    def __len__(self) -> int:
        return len(self._dataset)

    def get_problem(self, index: int) -> dict:
        idx = index % len(self._dataset)
        row = self._dataset[idx]
        prompt: str = row["input"]
        cases = self._row_cases(row)
        # Pin the grader's function-call contract onto the prompt. Changes prompt
        # tokens (GRAIL-bound), so this must match the validator byte-for-byte.
        prompt = prompt + _contract_instruction(cases)
        problem_id = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        case_id = hashlib.sha256(
            (problem_id + json.dumps(cases, sort_keys=True, separators=(",", ":"))).encode()
        ).hexdigest()[:16]
        self._cases_by_id[case_id] = cases
        return {"prompt": prompt, "ground_truth": case_id, "id": problem_id}

    def compute_reward(self, problem: dict, completion: str) -> float:
        case_id = problem.get("ground_truth", "")
        if not isinstance(case_id, str):
            return 0.0
        cases = self._cases_by_id.get(case_id)
        if not cases:
            return 0.0
        code = _extract_python(completion or "")
        return grade_structured_cases(
            code, cases, timeout_s=float(GRADER_EVAL_TIMEOUT_SECONDS),
        )

    @staticmethod
    def _row_cases(row) -> list[dict]:
        raw = row.get("structured_cases", [])
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                return []
        if not isinstance(raw, list):
            return []
        return [dict(c) for c in raw if isinstance(c, dict)]
