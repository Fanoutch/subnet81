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
are byte-identical. PORT 25/08 des PR upstream #202/#203 (image 7b4a407) :
depuis le protocole 5, le span noté est le dernier bloc fencé qui DÉFINIT la
fonction d'entrée du contrat, et l'absence de bloc fencé ne vaut plus repli sur
la complétion brute. v2-v4 restent byte-exacts (notre repli).
"""

from __future__ import annotations

import ast
import hashlib

from reliquary.protocol.profiles import render_active_prompt
import json
import os
import re
from pathlib import Path
from typing import ClassVar

from reliquary.constants import GRADER_EVAL_TIMEOUT_SECONDS
from reliquary.environment.code_grader import (grade_structured_cases,
                                               grade_structured_cases_ex)


# ---------------------------------------------------------------------------
# Code extraction + dataset + contract (VERBATIM from validator)
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(
    r"(```|~~~)(?:python3?|py)?\s*\n(.*?)\n\1",
    re.DOTALL,
)


def _entry_function_name(cases: list[dict]) -> str | None:
    """The contract's graded entry function, or None when it isn't a function.

    Same source as ``_contract_instruction``: the cases carry the exact name the
    grader will call, so the extractor can pin the graded block to a definition
    rather than to a position. Method entries define no top-level ``def``, so
    they pin nothing.
    """
    for case in cases or ():
        entry = case.get("entry") or {}
        name = entry.get("name")
        if entry.get("kind") == "function" and name:
            return str(name)
    return None


def _defines_top_level_entry(source: str, entry_name: str) -> bool:
    """Whether *source* defines the exact callable the grader will resolve."""

    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, TypeError):
        return False
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == entry_name
        for node in tree.body
    )


def _select_python_span(
    completion: str,
    entry_name: str | None = None,
    *,
    protocol_version: int | None = None,
) -> tuple[str, int, int] | None:
    """Return the exact fenced code span selected for execution.

    Offsets are completion-relative so semantic checks inspect the same bytes.
    v2-v4 retain their legacy last-fence/raw-completion behavior; v5 keeps its
    prompt and generation unchanged while selecting the intended fenced block.
    """

    if not completion:
        return None
    if protocol_version is None:
        from reliquary.constants import PROTOCOL_VERSION

        protocol_version = PROTOCOL_VERSION

    entry_rule = int(protocol_version) >= 5
    matches = list(_FENCE_RE.finditer(completion))
    if not matches:
        return None if entry_rule else (completion, 0, len(completion))
    if entry_name and entry_rule:
        for match in reversed(matches):
            body = match.group(2)
            if _defines_top_level_entry(body, entry_name):
                return body, match.start(2), match.end(2)
    match = matches[-1]
    return match.group(2), match.start(2), match.end(2)


def _extract_python(completion: str, entry_name: str | None = None) -> str:
    """Extract Python code from a model completion.

    PORT upstream PR #202/#203 (live sur le validateur depuis le 25/08 ~21:30
    UTC, image 7b4a407). Stratégie : depuis le protocole 5, on note le DERNIER
    bloc fencé qui *définit* ``entry_name`` ; sinon le dernier bloc.

    Sans aucun bloc fencé, v2-v4 renvoient la complétion brute (comportement
    historique conservé BYTE-EXACT — c'est notre repli) ; à partir de v5 le
    bloc fencé est le seul canal de réponse, donc rien n'est noté.

    Pourquoi la définition bat la position : « le dernier bloc gagne »
    supposait que le modèle termine par son implémentation. Sous le prompt de
    raisonnement v5, il termine régulièrement par une démonstration d'usage ou
    un listing de sortie attendue — noter ce bloc met à 0 une réponse correcte.

    ⚠️ Ce changement ne touche QUE la notation locale. Le prompt
    (``get_problem``) est inchangé : les tokens, donc le forced-seed, sont
    intacts. L'environnement code est ``validator_authoritative_reward=True``
    — l'écart ne provoquait aucun rejet, il faussait notre SÉLECTION
    (sigma, k, in_zone, valeur d'enchère).
    """
    selected = _select_python_span(completion, entry_name=entry_name)
    return selected[0] if selected is not None else ""


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
    # MIROIR LOCAL (24/08) : la tranche est tirée au hasard dans 2,48 M
    # d'indices à chaque fenêtre, donc les row-groups sont TOUJOURS froids —
    # 0,95 s de réseau HF par fenêtre, et 5,4 % de fenêtres où un
    # « handshake timed out » fait exploser le budget de classement (+9,5 s
    # sur le premier groupe). Le fichier fait 1,39 Go, le disque en a 85 de
    # libres. Variable absente => chemin distant historique, inchangé.
    local_root = os.environ.get("RELIQUARY_PARQUET_LOCAL_ROOT") or None
    ds = VirtualParquetDataset(
        repo, revision, columns=["input", "structured_cases"],
        local_root=local_root,
    )
    # ⚠️ GARDE : `len()` est le CONSENSUS PROMPT-RANGE — le validateur en
    # dérive la tranche de chaque fenêtre. Un miroir incomplet donnerait une
    # tranche décalée et 100 % de `prompt_out_of_range`, un échec total et
    # SILENCIEUX. On préfère lever au démarrage.
    expected = os.environ.get("RELIQUARY_PARQUET_EXPECTED_LEN")
    if expected:
        ds = _LenGuardedDataset(ds, int(expected))
    return ds


class _LenGuardedDataset:
    """Enveloppe qui refuse un `len()` différent de celui attendu.

    Transparente pour tout le reste (`__getitem__`, attributs), pour que le
    chemin de production reste identique au chemin historique.
    """

    def __init__(self, inner, expected_len: int) -> None:
        self._inner = inner
        self._expected_len = expected_len

    def __len__(self) -> int:
        n = len(self._inner)
        if n != self._expected_len:
            raise RuntimeError(
                f"miroir parquet incohérent : len={n}, attendu "
                f"{self._expected_len}. len() est le consensus prompt-range — "
                f"continuer donnerait une tranche décalée et 100 % de "
                f"prompt_out_of_range."
            )
        return n

    def __getitem__(self, idx):
        return self._inner[idx]

    def __getattr__(self, name):
        return getattr(self._inner, name)


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
        contract = _contract_instruction(cases)
        # PORT v5 (upstream PR #190, live 23/08) : depuis le protocole 5 le
        # validateur n'attend plus la concaténation `énoncé + contrat` mais un
        # TEMPLATE versionné qu'il publie dans son generation_contract. Le
        # prompt fixant les tokens, et les tokens le forced-seed, un écart d'un
        # caractère fait rejeter 100 % des rollouts.
        # `None` = protocole < 5 ou environnement non profilé → chemin legacy
        # strictement inchangé, donc le repli v4 reste byte-exact.
        rendered_prompt = render_active_prompt(
            self.name, problem=prompt, contract=contract,
        )
        prompt = prompt + contract if rendered_prompt is None else rendered_prompt
        problem_id = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        case_id = hashlib.sha256(
            (problem_id + json.dumps(cases, sort_keys=True, separators=(",", ":"))).encode()
        ).hexdigest()[:16]
        self._cases_by_id[case_id] = cases
        return {"prompt": prompt, "ground_truth": case_id, "id": problem_id}

    def compute_reward(self, problem: dict, completion: str) -> float:
        return self.compute_reward_ex(problem, completion)[0]

    def compute_reward_ex(
        self, problem: dict, completion: str,
    ) -> tuple[float, bool]:
        """(reward, timed_out) — le drapeau permet a la decision de zone
        d'imputer les timeouts au lieu de les compter comme des echecs
        (fausse dispersion -> 16,3 %% d'out_of_zone au verdict, 01/09)."""
        case_id = problem.get("ground_truth", "")
        if not isinstance(case_id, str):
            return 0.0, False
        cases = self._cases_by_id.get(case_id)
        if not cases:
            return 0.0, False
        code = _extract_python(
            completion or "", entry_name=_entry_function_name(cases),
        )
        return grade_structured_cases_ex(
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
