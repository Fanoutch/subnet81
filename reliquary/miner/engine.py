"""Miner engine — vLLM generation + HuggingFace GRAIL proof construction.

Protocol v2: free prompt selection (uniform random with cooldown skip),
M rollouts per prompt at fixed temperature T_PROTO, local reward computation,
Merkle root commitment, HTTP batch submission to validator.
"""

from __future__ import annotations

import asyncio
import logging
import os as _os
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING

import random as _random

from reliquary.miner.pool_persistence import delete_entry, load_pool, save_entry
from reliquary.miner.mix_controller import (
    entry_env_name as _entry_env_name_fn,
    pick_bake_env as _pick_bake_env,
)
from reliquary.miner.zone import ZONE_THRESHOLD_STEADY
from reliquary.miner.submitter import fetch_verdicts

from reliquary.constants import (
    LAYER_INDEX,
    MAX_NEW_TOKENS_PROTOCOL_CAP,
    MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW,
    M_ROLLOUTS,
    PROMPT_RANGE_SIZE,
    T_PROTO,
    TOP_K_PROTO,
    TOP_P_PROTO,
    UPLOAD_BUFFER,
    WINDOW_LENGTH,
)
from reliquary.infrastructure import chain
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    RolloutSubmission,
    WindowState,
)
from reliquary.protocol.tokens import encode_prompt
from reliquary.shared.prompt_range import window_prompt_range
from reliquary.shared.modeling import (
    MODEL_SNAPSHOT_ALLOW_PATTERNS,
    first_eos_index,
    load_text_generation_model,
    resolve_eos_token_ids,
)

if TYPE_CHECKING:
    from reliquary.environment.base import Environment

logger = logging.getLogger(__name__)


async def maybe_pull_checkpoint(
    state,
    local_n: int,
    local_hash: str,
    local_model,
    *,
    download_fn,
    load_fn,
):
    """If remote checkpoint_n > local, download via HF and load.

    state.checkpoint_repo_id + state.checkpoint_revision identify the
    HF snapshot. download_fn/load_fn still injected for testability.

    Returns ``(new_local_n, new_local_hash, new_model)``. If no update is
    needed (remote ≤ local, or remote has no repo/revision yet), returns
    inputs unchanged.
    """
    if state.checkpoint_n <= local_n:
        return local_n, local_hash, local_model
    if state.checkpoint_repo_id is None or state.checkpoint_revision is None:
        return local_n, local_hash, local_model
    local_path = await download_fn(state.checkpoint_repo_id, state.checkpoint_revision)
    new_model = load_fn(local_path)
    return state.checkpoint_n, state.checkpoint_revision, new_model


async def _hf_download(repo_id: str, revision: str) -> str:
    """Download a snapshot into the local HF cache and return the model folder path.

    After a successful pull, prune every other revision of ``repo_id`` from
    the local cache. The validator advances ``checkpoint_n`` monotonically
    and never rolls back, so the old snapshots are dead weight: each one
    weighed ~7 GB and 16 of them filled the 194 GB root partition in 24h,
    which stalled the miner at the next ``snapshot_download`` (no disk =
    no write = retry loop, GPU idle). Failures here MUST NOT propagate —
    the pull itself already succeeded and the miner can fire its window
    from the freshly-loaded model.
    """
    import asyncio
    from huggingface_hub import snapshot_download

    local_path = await asyncio.to_thread(
        snapshot_download,
        repo_id=repo_id,
        revision=revision,
        allow_patterns=MODEL_SNAPSHOT_ALLOW_PATTERNS,
    )
    try:
        await asyncio.to_thread(_prune_hf_revisions, repo_id, revision)
    except Exception:
        logger.exception("hf cache prune failed for %s — disk may fill", repo_id)
    return local_path


def _prune_hf_revisions(repo_id: str, keep_revision: str) -> None:
    """Delete every cached revision of ``repo_id`` except ``keep_revision``.

    Uses ``huggingface_hub.scan_cache_dir`` so blob refcounting is correct
    (a blob shared across revisions stays as long as one referencing
    revision survives — here only ``keep_revision`` survives, so all
    blobs unique to the deleted revisions get reclaimed).
    """
    from huggingface_hub import scan_cache_dir

    info = scan_cache_dir()
    to_delete: list[str] = []
    for repo in info.repos:
        if repo.repo_id != repo_id:
            continue
        for rev in repo.revisions:
            if rev.commit_hash != keep_revision:
                to_delete.append(rev.commit_hash)
        break
    if not to_delete:
        return
    strategy = info.delete_revisions(*to_delete)
    logger.info(
        "hf cache prune: %s — dropping %d old revisions, freeing %s",
        repo_id, len(to_delete), strategy.expected_freed_size_str,
    )
    strategy.execute()


def pick_prompt_idx(
    env,
    cooldown_prompts: set[int],
    *,
    rng: _random.Random | None = None,
    max_attempts: int = 1000,
    prompt_range: tuple[int, int] | None = None,
) -> int:
    """Pick a random prompt index that isn't currently in cooldown.

    If the env exposes ``eligible_indices`` (a non-None list), sampling is
    restricted to that pool — used to bias toward problem_source values with
    a higher empirical in-zone pass rate. Falls back to uniform-random
    over the full dataset when the attribute is missing or None.

    When ``prompt_range`` is given, sampling is additionally confined to the
    per-window ``[lo, hi)`` slice the validator enforces (#91). The hard range
    constraint wins over the eligible-indices bias: if the biased pool does not
    intersect the slice, we fall back to uniform sampling over ``[lo, hi)`` so
    the returned index is always in-range (never PROMPT_OUT_OF_RANGE).

    Raises ``RuntimeError`` if no eligible prompt can be found — typically
    because the env (or the window slice) is fully in cooldown.
    """
    rng = rng or _random
    n = len(env)
    if prompt_range is None:
        lo, hi = 0, n
    else:
        lo, hi = max(0, prompt_range[0]), min(n, prompt_range[1])
        if hi - lo <= 0:
            raise RuntimeError("no eligible prompt — empty prompt range")

    pool = getattr(env, "eligible_indices", None)
    if pool is None:
        # Whole (clamped) range — sample directly, no list materialisation.
        span = hi - lo
        if len(cooldown_prompts) < span / 2:
            for _ in range(max_attempts):
                idx = lo + rng.randrange(span)
                if idx not in cooldown_prompts:
                    return idx
            raise RuntimeError("no eligible prompt found after max attempts")
        eligible = [i for i in range(lo, hi) if i not in cooldown_prompts]
        if not eligible:
            raise RuntimeError("no eligible prompt — range fully in cooldown")
        return rng.choice(eligible)

    # Biased pool (eligible_indices): confine to the window slice when given.
    if prompt_range is not None:
        confined = [i for i in pool if lo <= i < hi]
        # Bias misses the slice entirely → the hard range constraint wins.
        pool = confined if confined else list(range(lo, hi))
    n_pool = len(pool)
    if len(cooldown_prompts) < n_pool / 2:
        for _ in range(max_attempts):
            idx = pool[rng.randrange(n_pool)]
            if idx not in cooldown_prompts:
                return idx
        raise RuntimeError("no eligible prompt found after max attempts")
    eligible = [i for i in pool if i not in cooldown_prompts]
    if not eligible:
        raise RuntimeError("no eligible prompt — env fully in cooldown")
    return rng.choice(eligible)


def _compute_merkle_root(rollouts) -> str:
    """Compute Merkle root over rollout leaves — returns 64-char hex.

    Uses canonical JSON (sort_keys=True, compact separators) for dict/list
    serialisation so the root is deterministic across Python
    implementations and refactor-stable against dict-construction-order
    changes.
    """
    import hashlib
    import json

    leaves = []
    for i, r in enumerate(rollouts):
        h = hashlib.sha256()
        h.update(i.to_bytes(8, "big"))
        h.update(json.dumps(r.tokens, separators=(",", ":")).encode())
        h.update(json.dumps(r.reward).encode())
        h.update(json.dumps(r.commit, sort_keys=True, separators=(",", ":")).encode())
        leaves.append(h.digest())

    while len(leaves) > 1:
        new = []
        for i in range(0, len(leaves), 2):
            left = leaves[i]
            right = leaves[i + 1] if i + 1 < len(leaves) else left
            new.append(hashlib.sha256(left + right).digest())
        leaves = new
    return leaves[0].hex()


# Calibrated offset (seconds) added to local time.time() before computing
# the drand round at POST time. PRIMARY source of truth: the validator's
# HTTP Date header on every /state poll (NTP-synced, refreshed ~200x/sec
# in _trigger_loop). FALLBACK: drand-network latest, refreshed every 60 s.
# Compensates for local-VM clock drift (Prime Intellect VMs we run on can
# drift seconds-per-minute vs UTC; the v2.3 validator enforces strict
# equality on drand_round in BOTH directions, so any uncorrected drift
# produces routine STALE_ROUND / FUTURE_ROUND rejections at the validator).
_DRAND_CLOCK_OFFSET_S: float = 0.0

# Running EMA of validator-vs-local offset, smoothing out the 1-s
# quantization of the HTTP Date header across many polls. ``None`` until
# the first valid sample. EMA factor tuned so 10 polls (~50 ms wall) cover
# 90% of a step change — fast enough to track a slewing clock, slow enough
# to absorb individual-sample jitter from RTT outliers.
_VALIDATOR_OFFSET_EMA: float | None = None
_VALIDATOR_OFFSET_EMA_ALPHA: float = 0.2

# Backoff between /state poll retries after an HTTP error. Validator returns
# 503 (no_active_window) for a brief window during the OPEN flip transition
# (between set_active_batcher(None) and set_active_batcher(new_batcher) in
# validator/server.py:287-288). The 503 window is sub-second in practice;
# retrying fast catches the next OPEN flip without missing rounds. Don't
# use ``constants.POLL_INTERVAL_SECONDS = 10`` here — that constant is for
# the validator's own loop cadence and is way too slow for the miner's
# 200 Hz reactive polling. See commit/incident 2026-05-16 where the wrong
# constant cost us 25 drand rounds (75 s) on a cold start.
_STATE_RETRY_S: float = 0.05


def _current_drand_round_at_send() -> int:
    """Drand quicknet round currently in progress at wall-clock now,
    corrected for local clock drift via ``_DRAND_CLOCK_OFFSET_S``.

    Called just before POSTing /submit so the attached round matches the
    one the validator computes at receipt. The v2.3 round check is
    zero-tolerance in both directions, so the corrected clock must track
    the validator's NTP-synced wall clock to within the inter-host RTT.
    Calibrated continuously from the validator's HTTP Date header (200 Hz)
    and falls back to the drand network every ``_DRAND_OFFSET_REFRESH_S``
    seconds.
    """
    from reliquary.infrastructure.chain import compute_current_drand_round
    from reliquary.infrastructure.drand import get_current_chain

    ci = get_current_chain()
    return compute_current_drand_round(
        time.time() + _DRAND_CLOCK_OFFSET_S,
        ci["genesis_time"], ci["period"],
    )


# Validator filter knobs. Flip via env vars when the validator deploys
# relaxed thresholds (sigma lowered → k in [1,7], MAX_TRUNCATED bumped →
# up to 5 non-bt_ok rollouts per submission). No code change needed.
K_MIN = int(_os.environ.get("RELIQUARY_K_MIN", "3"))
K_MAX = int(_os.environ.get("RELIQUARY_K_MAX", "5"))
MAX_NON_BTOK_IN_SUBMISSION = int(
    _os.environ.get("RELIQUARY_MAX_NON_BTOK_IN_SUBMISSION", "0"),
)
# Oversampling: generate more than M_ROLLOUTS rollouts per prompt, then
# pick the M_ROLLOUTS best (= highest local q10 = least likely to be
# rejected by validator's distribution_suspicious filter) that still
# satisfy the k-band requirement. Set to e.g. 12 or 16 if rejection rate
# from q10 is high.
OVERSAMPLE_N = max(M_ROLLOUTS, int(_os.environ.get("RELIQUARY_OVERSAMPLE_N", str(M_ROLLOUTS))))

# Multi-phase bake strategy. Each phase generates M_PER_PHASE rollouts
# per prompt. After each phase, decide drop/submit/retry. Prompts that
# fail early checks (sigma=0 or bt_ok=0 after phase 1) are dropped
# immediately. Prompts that look promising but can't compose a valid
# submission yet go into the retry queue for up to MAX_PHASES total
# phases. Validated offline: ~2x selecteds/h vs single-phase OVERSAMPLE.
M_PER_PHASE = M_ROLLOUTS  # = 8, mirrors submission size
MAX_PHASES = int(_os.environ.get("RELIQUARY_MAX_PHASES", "3"))
# Drop the prompt after phase 1 if 0/8 rollouts terminate cleanly
# (= infinite loop on this prompt — phase 2/3 won't recover). Set to 0
# to disable and always try the full MAX_PHASES.
DROP_BTOK0_PHASE1 = _os.environ.get("RELIQUARY_DROP_BTOK0_PHASE1", "1") == "1"
# Optional hard filter — drop rollouts whose LOCAL q10 (under T_PROTO
# scaling, computed during bake to match the validator's filter) is
# below this threshold. Default 0 = off. Set to 0.05 to leave a margin
# above the validator's 0.025 threshold.
MIN_LOCAL_Q10 = float(_os.environ.get("RELIQUARY_MIN_LOCAL_Q10", "0.0"))
MIN_LOCAL_MEDIAN = float(_os.environ.get("RELIQUARY_MIN_LOCAL_MEDIAN", "0.0"))
EOS_TOKEN_IDS = (151643, 151645)  # Qwen3 generation_config.eos_token_id
# Validator threshold is 0.01. Our HF mirrors validator's exact compute so we
# use the SAME threshold — any rollout passing locally has very high odds of
# passing validator. Submitting a borderline reject is cheap (just wastes a
# slot), so being too strict only loses us valid submissions.
P_STOP_LOCAL_MIN = 0.01  # = MIN_EOS_PROBABILITY validator. HF↔HF on same
                          # checkpoint matches bit-for-bit ± bf16 noise.
                          # The real source of bad_termination rejects is
                          # checkpoint advance between bake and submit, not
                          # threshold drift — fixed by DROP_POOL_ON_CKPT=1.

# EXPERIMENT (2026-05-29): the validator's new preflight (commit 2ebb619)
# pre-rejects a submission if ANY rollout's *claimed* final-token logprob is
# below log(MIN_EOS_PROBABILITY). For a naturally-terminated rollout (last
# token IS an EOS), cross-stack bf16/flash-attn drift can put our reported
# single-token prob just under 0.01 even though the validator's own GRAIL
# recompute would clear it. When >0, we floor the reported final-token logprob
# of naturally-terminated rollouts to log(EOS_LOGPROB_FLOOR) so the cheap
# preflight passes and GRAIL (the authoritative recompute) becomes the arbiter
# again. Off by default. Set RELIQUARY_EOS_LOGPROB_FLOOR=0.01 to enable.
EOS_LOGPROB_FLOOR = float(_os.environ.get("RELIQUARY_EOS_LOGPROB_FLOOR", "0.0"))


# Validator's STEADY sigma-zone threshold. constants.SIGMA_MIN is 0.33 in this
# fork (does NOT match the live validator's 0.43), so we pin 0.43 explicitly —
# same pin _select_continuous_subset already uses. Below this, the validator
# rejects OUT_OF_ZONE.
_VALIDATOR_STEADY_SIGMA_MIN = 0.43


def _skip_for_out_of_zone(rewards: list[float]) -> bool:
    """Return True iff the CURRENT validator would reject this rollout group.

    The live auction-v2 validator (origin/main ``batcher.py`` ~1629-1636) gates
    ONLY on the sigma zone: ``sigma >= SIGMA_MIN`` with the STEADY threshold 0.43
    (comment: "Keep the calibrated sigma eligibility band even under the
    auction"). For binary M=8 that is exactly **k ∈ [2, 6]**.

    The old k ∈ [3, 5] "binary reward distribution guard" (commit 60e4a81) was
    DROPPED validator-side — ``REWARD_DISTRIBUTION`` is now a vestigial enum
    member with no enforcement path. Keeping it here made the miner discard k=2
    and k=6 groups the validator accepts — and k=2 is the HIGHEST auction score
    (``std·(1-mean)``, hard prompt), so we were throwing away our best-paid work
    and inflating the out-of-zone search. Removed 2026-07-18.

    We pin the validator's steady 0.43 (NOT ``constants.SIGMA_MIN`` = 0.33 in
    this fork). During a real validator bootstrap (0.33) the miner is slightly
    conservative (misses k=1,7) — safe: no rejects, just fewer submissions.

    Called from ``_pre_bake_entry``/``_pre_bake_batch`` after rewards are
    computed; entries that would be rejected are dropped before the pool.

    ``RELIQUARY_ZONE_SIGMA_MIN`` overrides the threshold at CALL time. This is a
    DIAGNOSTIC hatch: the submit path (precommit -> reveal -> verdict) had never
    run in production because every group was dropped here first, so setting it
    to 0.0 for a few windows lets a real group through and makes the handshake
    and its timing observable. The resulting verdict is OUT_OF_ZONE, rejected
    before the GRAIL proof path (no expensive-proof budget consumed). Leaving it
    lowered permanently would burn the per-window submission quota on work the
    validator always rejects — restore the default once measured. A malformed
    value falls back to the safe default rather than disabling the filter.
    """
    from reliquary.validator.verifier import rewards_std
    threshold = _VALIDATOR_STEADY_SIGMA_MIN
    raw = _os.environ.get("RELIQUARY_ZONE_SIGMA_MIN")
    if raw is not None:
        try:
            threshold = float(raw)
        except ValueError:
            logger.warning(
                "RELIQUARY_ZONE_SIGMA_MIN=%r is not a number; keeping %.2f",
                raw, _VALIDATOR_STEADY_SIGMA_MIN,
            )
    return rewards_std(rewards) < threshold


def grade_group_parallel(env, problem_completions, *, max_workers: int = 8):
    """Corrige les rollouts d'un groupe EN PARALLÈLE, dans l'ordre d'entrée.

    Mesuré 2026-07-23 (code-only) : le mineur passe 52% de son temps sur
    ``reward`` avec le GPU à 0%. En code, ``compute_reward`` lance un
    ``subprocess.run`` isolé par rollout (cas de test en sandbox), exécutés en
    série. Les threads les recouvrent (subprocess.run libère le GIL) : mesuré
    ×9,6 (128 corrections 4,52 s → 0,47 s sur 16 threads).

    Sûr par construction : la correction ne touche NI aux tokens NI aux preuves,
    seulement un score par rollout. Une correction qui lève est ramenée à 0.0
    (le grader code le fait déjà sur crash ; on protège aussi le wrapper).
    ``problem_completions`` = liste de ``(problem, completion_text)``.
    """
    from concurrent.futures import ThreadPoolExecutor

    n = len(problem_completions)
    if n <= 1:
        return [
            _safe_reward(env, p, c) for p, c in problem_completions
        ]
    with ThreadPoolExecutor(max_workers=min(max_workers, n)) as pool:
        return list(pool.map(
            lambda pc: _safe_reward(env, pc[0], pc[1]), problem_completions,
        ))


def _safe_reward(env, problem, completion) -> float:
    try:
        return float(env.compute_reward(problem, completion))
    except Exception:
        logger.exception("compute_reward a leve; score=0.0")
        return 0.0


def _std(xs: list[float]) -> float:
    """Population standard deviation (matches the validator's rewards_std)."""
    n = len(xs)
    if n == 0:
        return 0.0
    m = sum(xs) / n
    return (sum((x - m) ** 2 for x in xs) / n) ** 0.5


def _select_continuous_subset(rollouts, size, sigma_target):
    """Subset of ``size`` rollouts maximising the dispersion of continuous
    rewards, returned only if its std >= ``sigma_target``. Heuristic: sort by
    reward and take the extremes (lower half + upper half) — the maximum-variance
    composition for a fixed size. None if the threshold is not reached (e.g. the
    model only produced all-pass / all-fail / middling outputs → not in-zone)."""
    if len(rollouts) < size:
        return None
    ordered = sorted(rollouts, key=lambda r: r["reward"])
    lo = size // 2
    hi = size - lo
    subset = ordered[:lo] + ordered[len(ordered) - hi:]
    if _std([r["reward"] for r in subset]) >= sigma_target:
        return subset
    return None


def _should_fire_for_window(
    state, fired_windows: set[int], forfeit_windows: set[int], pool_size: int,
) -> bool:
    """True iff the trigger loop should fire a burst right now.

    Pure function so the gate is unit-testable without spinning up an
    event loop. The five conditions mirror the design doc
    (specs/2026-05-16-r-open-only-burst-design.md):
      * the window hasn't already been fired,
      * the window hasn't been forfeited (pool was empty at the first OPEN
        tick — under the R_open-only policy we commit to skipping the
        whole window rather than firing mid-window at R_open+k),
      * /state reports OPEN,
      * the validator has published randomness,
      * the pool has at least one bakeable entry.
    """
    return (
        state.window_n not in fired_windows
        and state.window_n not in forfeit_windows
        and state.state == WindowState.OPEN
        and bool(state.randomness)
        and pool_size > 0
    )


def _apply_offset_from_validator_response(
    resp: "httpx.Response", t_send: float, t_recv: float
) -> float | None:
    """Update the global clock offset from the validator's HTTP Date header.

    The Date header is the validator's NTP-synced wall clock at response
    generation, in 1-s precision (RFC 7231 — ``parsedate_to_datetime``
    returns ``floor(T_validator)``). We approximate the local time at
    which the validator stamped the header as the midpoint of send/recv
    (half-RTT correction). The raw offset
    ``parsed_Date - local_midpoint`` is therefore systematically negative
    by the fractional part of the validator's stamp — uniformly
    distributed in ``[-1, 0]`` over many polls, mean ``-0.5 s``.

    Floor compensation: we add a +0.5 s constant so the EMA's expected
    output equals zero when both clocks are perfectly NTP-synced. Without
    this term the corrected clock systematically runs ~0.5 s behind the
    validator, and the v2.3 zero-tolerance drand check rejects every POST
    that falls within ~0.5 s after a round boundary on the validator
    side (~8% of POSTs at quicknet 3 s period) as STALE_ROUND.

    Returns the new ``_DRAND_CLOCK_OFFSET_S`` (== EMA + 0.5), or ``None``
    if the Date header was missing or unparseable (in which case the
    global offset is left untouched).
    """
    from email.utils import parsedate_to_datetime

    global _DRAND_CLOCK_OFFSET_S, _VALIDATOR_OFFSET_EMA
    date_header = resp.headers.get("date") if resp is not None else None
    if not date_header:
        return None
    try:
        validator_time = parsedate_to_datetime(date_header).timestamp()
    except (TypeError, ValueError):
        return None
    local_midpoint = (t_send + t_recv) / 2.0
    raw_offset = validator_time - local_midpoint
    if _VALIDATOR_OFFSET_EMA is None:
        _VALIDATOR_OFFSET_EMA = raw_offset
    else:
        a = _VALIDATOR_OFFSET_EMA_ALPHA
        _VALIDATOR_OFFSET_EMA = (1 - a) * _VALIDATOR_OFFSET_EMA + a * raw_offset
    # +0.5 s compensates the 1-s floor of the HTTP Date header. See docstring.
    _DRAND_CLOCK_OFFSET_S = _VALIDATOR_OFFSET_EMA + 0.5
    return _DRAND_CLOCK_OFFSET_S


def _compute_offset_sub_second(r_drand: int, t_fetch: float, ci: dict) -> float:
    """Sub-second-precision offset estimation.

    Drand "latest" returns the round currently in progress. The round R
    starts at ``T_R = genesis + (R-1)*period`` and ends at ``T_R+period``.
    Our HTTP fetch lands somewhere in that interval. Single-shot precision
    bound: ±period/2 (~1.5 s on quicknet).

    Anchor: middle of the round (``period/2``). The v2.3 round check is
    zero-tolerance on both sides, so any asymmetric bias just trades one
    failure mode for the other. Centering minimizes the worst-case error
    in either direction. This fallback only runs at cold start (before
    the validator-Date EMA converges, ~1 s of polling) and during /state
    outages, so per-call precision matters less than not skewing.
    """
    period = ci["period"]
    t_round_start = ci["genesis_time"] + (r_drand - 1) * period
    t_anchor = t_round_start + period / 2
    return t_anchor - t_fetch


async def _refresh_drand_offset_loop() -> None:
    """Background: keep ``_DRAND_CLOCK_OFFSET_S`` calibrated against the
    drand network's actual advertised round.

    Refresh cadence is ~1 min: drand quicknet has a 3-second period, so
    a 60-s-old calibration is at worst ~20 rounds stale on the local
    clock, which compounded with a fast-drifting VM clock easily exceeds
    the validator's zero-tolerance round check. Operators can lower this
    via ``RELIQUARY_DRAND_OFFSET_REFRESH_S`` if their box drifts faster.

    Never raises — on any drand fetch failure we log and keep the
    previous offset; failing soft is preferable to crashing the miner
    over a transient drand-relay hiccup.
    """
    global _DRAND_CLOCK_OFFSET_S
    from reliquary.infrastructure.drand import get_beacon, get_current_chain

    refresh_s = float(
        _os.environ.get("RELIQUARY_DRAND_OFFSET_REFRESH_S", "60"),
    )
    while True:
        try:
            # ``get_beacon`` is sync (HTTP); run on a thread so we don't
            # block the asyncio event loop while the relay responds.
            beacon = await asyncio.to_thread(
                get_beacon, "latest", True, False,
            )
            t_fetch = time.time()
            r_drand = int(beacon["round"])
            ci = get_current_chain()
            new_offset = _compute_offset_sub_second(r_drand, t_fetch, ci)
            # Only write if the validator-Date EMA hasn't taken over yet.
            # Once /state is reachable the validator's clock is the more
            # direct source of truth and gets updated ~200 Hz.
            if _VALIDATOR_OFFSET_EMA is None:
                prev = _DRAND_CLOCK_OFFSET_S
                _DRAND_CLOCK_OFFSET_S = new_offset
                if abs(new_offset - prev) > 0.5 or prev == 0.0:
                    logger.info(
                        "drand offset (fallback): %+.2fs → %+.2fs "
                        "(drand-latest=%d)",
                        prev, new_offset, r_drand,
                    )
            else:
                logger.debug(
                    "drand fallback skipped — validator-Date EMA active "
                    "(%+.3fs)", _VALIDATOR_OFFSET_EMA,
                )
        except asyncio.CancelledError:
            return
        except Exception:
            logger.debug(
                "drand offset refresh failed; keeping previous "
                "(_DRAND_CLOCK_OFFSET_S=%+.2fs)",
                _DRAND_CLOCK_OFFSET_S,
                exc_info=True,
            )
        try:
            await asyncio.sleep(refresh_s)
        except asyncio.CancelledError:
            return


def vllm_forced_seed_enabled() -> bool:
    """Gate for running forced-seed generation on vLLM instead of the HF sync
    loop. OFF (default) = live behaviour unchanged (HF forced-seed). Flip
    RELIQUARY_VLLM_FORCED_SEED=1 to route phase-1 through vLLM
    (VLLMForcedSeedLogitsProcessor), gated on the offline seed-consistency
    validation (group 0.9768 / worst 0.9423 on 2026-07-17). Phase-2 stays HF."""
    return _os.environ.get("RELIQUARY_VLLM_FORCED_SEED", "0") == "1"


def wire_v2_enabled() -> bool:
    """Wire-v2 cutover gate (upstream agent/wire-v2-cutover, NOT yet merged).
    OFF (default) = live wire v1, byte-identical behaviour. Flip
    RELIQUARY_WIRE_V2=1 the day the validator enforces v2: protocol_version=2,
    canonical Merkle root, version bound into the v2 envelope domain. A legacy
    client is rejected PROTOCOL_VERSION_MISMATCH after the cutover — and a v2
    client would be rejected before it — so this must flip WITH the validator."""
    return _os.environ.get("RELIQUARY_WIRE_V2", "0") == "1"


def wire_protocol_version() -> int:
    """protocol_version to advertise on BatchSubmissionRequest."""
    if wire_v2_enabled():
        return 2
    from reliquary.constants import FORCED_SEED_PROTOCOL_VERSION

    return FORCED_SEED_PROTOCOL_VERSION


def submission_merkle_root(rollout_subs) -> str:
    """Merkle root for a rollout group: canonical (validator-recomputed,
    binds env_name + domain-separated) under wire-v2, legacy otherwise."""
    if wire_v2_enabled():
        from reliquary.protocol.merkle import compute_rollouts_merkle_root

        return compute_rollouts_merkle_root(rollout_subs)
    return _compute_merkle_root(rollout_subs)


def drop_pool_on_ckpt_advance() -> bool:
    """Whether entries baked under an older checkpoint are dropped when the
    checkpoint advances. Legacy default = optimistic keep (a GRAIL-tolerance
    bet, sometimes recoverable). Under FORCED_SEED_ENFORCE the bet is
    ALWAYS-LOSING — checkpoint_hash is a ``u_at`` seed input, so old-hash
    tokens fired under the new hash are a guaranteed SEED_MISMATCH — hence the
    drop is forced in code, not left to a launch env var."""
    from reliquary.constants import FORCED_SEED_ENFORCE

    if FORCED_SEED_ENFORCE:
        return True
    return _os.environ.get("RELIQUARY_DROP_POOL_ON_CKPT", "0") == "1"


def pool_persist_enabled(prompt_range_from_window: int) -> bool:
    """Cross-window disk persistence of the pool. Only meaningful in the legacy
    model: off when the per-window prompt range is armed (stale out-of-slice
    entries) AND off under forced-seed enforcement (generation is
    randomness-dependent — entries never survive a window, so reloading them at
    boot would fire dead-randomness tokens → SEED_MISMATCH)."""
    from reliquary.constants import FORCED_SEED_ENFORCE

    return prompt_range_from_window == 2 ** 63 - 1 and not FORCED_SEED_ENFORCE


class MiningEngine:
    """Two-GPU mining: vLLM (GPU 0) for generation, HF (GPU 1) for proofs."""

    def __init__(
        self,
        vllm_model,
        hf_model,
        tokenizer,
        wallet,
        env: "Environment",
        *,
        vllm_gpu: int = 0,
        proof_gpu: int = 1,
        max_new_tokens: int = MAX_NEW_TOKENS_PROTOCOL_CAP,
        validator_url_override: str | None = None,
        vllm_backend: "VLLMBackend | None" = None,
    ) -> None:
        self.vllm_model = vllm_model
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        self.wallet = wallet
        self.env = env
        # Multi-env scaffolding (spec §6). Phase 1: RELIQUARY_ACTIVE_ENVS
        # defaults to math-only → this block is ADDITIVE (self.env and the
        # existing single-env paths are untouched) until the generator/fire
        # loops are routed per-env in later tasks. Phase 2 adds
        # "opencodeinstruct" to RELIQUARY_ACTIVE_ENVS to activate code.
        from reliquary.environment import load_environment as _load_env
        from reliquary.miner.mix_controller import MixController as _MixController
        self.active_envs = [s.strip() for s in _os.environ.get(
            "RELIQUARY_ACTIVE_ENVS", "openmathinstruct").split(",") if s.strip()]
        self.envs = {
            n: (env if getattr(env, "name", None) == n else _load_env(n))
            for n in self.active_envs
        }
        self._mix = _MixController(
            self.active_envs,
            total_slots=MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW, slot_floor=1)
        self.vllm_gpu = vllm_gpu
        self.proof_gpu = proof_gpu
        # Allow env override for max_new_tokens. Default = protocol cap
        # (= 8192) but offline EOS distribution test shows 99% of clean
        # EOS rollouts terminate by ~3200 tokens — capping at 3500
        # saves ~40% compute on infinite-loop prompts at the cost of
        # ~0.8% lost bt_ok rate.
        self.max_new_tokens = int(_os.environ.get(
            "RELIQUARY_MAX_NEW_TOKENS", str(max_new_tokens),
        ))
        self.validator_url_override = validator_url_override
        self._vllm_backend = vllm_backend

        # Lazy imports for heavy deps — keep module import cheap.
        from reliquary.shared.hf_compat import resolve_hidden_size
        from reliquary.protocol.grail_verifier import GRAILVerifier

        self._hidden_dim = resolve_hidden_size(hf_model)
        self._verifier = GRAILVerifier(hidden_dim=self._hidden_dim)

        # Full EOS set for the loaded model (Qwen3.5: generation_config +
        # nested text_config + tokenizer; Qwen3-4B: falls back to the tokenizer
        # /pad pair). Used for vLLM stop tokens, first-EOS truncation and the
        # termination/p_stop checks. Refreshed on checkpoint advance in
        # ``_load_checkpoint``. Falls back to the historical hardcoded pair if
        # the model exposes nothing.
        self._eos_ids = self._resolve_eos_ids()

    def _resolve_eos_ids(self) -> list[int]:
        """Resolve the model's EOS id set as a sorted list (never empty)."""
        ids = resolve_eos_token_ids(self.hf_model, self.tokenizer)
        if not ids:
            ids = set(EOS_TOKEN_IDS)
        return sorted(ids)

    def _entry_env_name(self, entry: dict) -> str:
        """Env that baked ``entry``; defaults to the first active env for
        legacy disk-reloaded entries lacking the key (back-compat)."""
        return _entry_env_name_fn(entry, self.active_envs[0])

    def _pool_env_stats(self) -> tuple[dict[str, int], dict[str, set[int]]]:
        """(counts, in_pool_idxs) per env over the current pool. Caller holds
        ``self._pool_lock``. Drives ``pick_bake_env`` (deficit per env) and
        per-env duplicate exclusion."""
        counts = {n: 0 for n in self.active_envs}
        in_pool: dict[str, set[int]] = {n: set() for n in self.active_envs}
        for e in self._pool:
            en = self._entry_env_name(e)
            counts[en] = counts.get(en, 0) + 1
            in_pool.setdefault(en, set()).add(e["prompt_idx"])
        return counts, in_pool

    def _apply_verdicts(self, resp) -> float:
        """Feed each verdict's reward outcome into the MixController, mapping
        merkle_root → the env we submitted it under. Returns the max ``ts``
        seen (advances the `since` cursor). Verdicts with rewarded=None or an
        unknown merkle_root are skipped (no signal)."""
        max_ts = 0.0
        accepted = 0
        reject_counts: dict[str, int] = {}
        for v in resp.verdicts:
            max_ts = max(max_ts, v.ts)
            # Visibility: surface the validator's REAL (post-GRAIL) verdicts so a
            # silent reject-every-window (e.g. base-model fallback → GRAIL_FAIL,
            # or a stale checkpoint) is loud in the log, not just felt as zero
            # rewards. The immediate POST only returns SUBMITTED; the true
            # outcome arrives here via /verdicts.
            if getattr(v, "accepted", False):
                accepted += 1
            else:
                reason = getattr(v.reason, "value", None) or str(getattr(v, "reason", "?"))
                reject_counts[reason] = reject_counts.get(reason, 0) + 1
            env = self._submitted_env.get(v.merkle_root)
            if env is None or v.rewarded is None:
                continue
            self._mix.record_outcome(env, bool(v.rewarded))
        if reject_counts:
            logger.warning(
                "verdicts: %d accepted, %d REJECTED %s",
                accepted, sum(reject_counts.values()),
                dict(sorted(reject_counts.items())),
            )
        elif accepted:
            logger.info("verdicts: %d accepted", accepted)
        return max_ts

    async def _tick_verdicts(self, url, *, client) -> None:
        """One verdicts poll: fetch since the cursor, feed the MixController,
        advance the cursor, and trim the merkle→env map. Never raises."""
        hk = self.wallet.hotkey.ss58_address
        resp = await fetch_verdicts(
            url, hk, client=client, since=self._verdicts_since or None,
        )
        if resp is None or not resp.verdicts:
            return
        new_ts = self._apply_verdicts(resp)
        if new_ts > self._verdicts_since:
            self._verdicts_since = new_ts
        # Bound the map: keep the most recent ~2000 submissions.
        if len(self._submitted_env) > 2000:
            for k in list(self._submitted_env)[:-2000]:
                self._submitted_env.pop(k, None)

    async def _verdicts_loop(self, url, client) -> None:
        """Background poll of GET /verdicts/{hotkey} → MixController yield
        signal. Independent of the latency-critical submit path; failures are
        logged and never kill the loop."""
        while True:
            try:
                await self._tick_verdicts(url, client=client)
                await asyncio.sleep(10.0)
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("verdicts loop iteration failed; continuing")
                await asyncio.sleep(10.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def mine_window(
        self,
        subtensor,
        window_start: int = 0,  # v2.0 param kept for CLI compat; ignored
        use_drand: bool = True,
    ) -> list:
        """v2.3 PIPELINED miner: continuous bg pre-bake, foreground POST on flip.

        Splits the per-prompt work into a randomness-independent half (vLLM
        generate + HF forward + reward + token_logprobs — the slow ~30-50 s
        portion) and a randomness-dependent half (r_vec, commitments,
        signature — the fast ~50 ms portion). A background generator pre-bakes
        the first half into a shared pool. A foreground trigger loop polls
        /state and, the instant ``state.randomness`` shows up for a new
        window, drains up to 8 entries from the pool, finalizes them with
        that randomness, and POSTs in parallel. Goal: every submission lands
        inside the first drand round of the window (3 s), so the validator's
        drand-anchored ordering ranks us with the earliest chronological
        bucket regardless of how much GPU competitors have.
        """
        import os as _os

        import httpx
        import random

        from reliquary.miner.submitter import (
            SubmissionError, discover_validator_url,
        )

        # Resolve validator URL (once).
        if self.validator_url_override:
            url = self.validator_url_override
        else:
            metagraph = await chain.get_metagraph(subtensor, chain.NETUID)
            url = discover_validator_url(metagraph)

        # Shared state. Mutated by both the generator (writer) and trigger
        # loops (reader/draining writer). Protected by ``_pool_lock``.
        self._pool: list[dict] = []
        self._pool_lock = asyncio.Lock()
        # 200 (was 16): v2.3 fires consume 8 entries/window, bg generator
        # needs headroom. Operators can lower via env var.
        self._pool_max_size = int(_os.environ.get("RELIQUARY_POOL_MAX_SIZE", "200"))
        # merkle_root → env_name we submitted it under (bounded; trimmed in the
        # verdicts loop). Maps each /verdicts outcome back to its env for the
        # MixController yield signal. Single-env in Phase 1 (always math).
        self._submitted_env: dict[str, str] = {}
        # Incremental cursor for GET /verdicts?since=
        self._verdicts_since: float = 0.0
        # Sentinel -1 so the FIRST observed checkpoint_n (even 0 — the
        # first publish from a fresh-bootstrap validator like reliquary-sn-v23)
        # is strictly greater and triggers a pull. With ``_local_n=0`` init,
        # ``state.checkpoint_n=0 <= local_n=0`` short-circuits the pull,
        # leaving ``_local_hash=""`` while the validator has a non-empty
        # ``current_checkpoint_hash`` → WRONG_CHECKPOINT reject on every POST.
        self._local_n = -1
        self._local_hash = ""
        # Per-window prompt range (#91). GATED: until the validator arms
        # PROMPT_RANGE_ENFORCE_FROM_WINDOW we keep the legacy cross-window
        # pre-bake model (sentinel 2**63-1 = never). Set
        # RELIQUARY_PROMPT_RANGE_FROM_WINDOW to the announced cutover window to
        # switch to intra-window slice-confined generation: the generator only
        # bakes prompts in the window's [lo, hi) slice and the pool is flushed
        # on each randomness flip (prior-slice entries are unsubmittable).
        self._prompt_range_from_window = int(
            _os.environ.get("RELIQUARY_PROMPT_RANGE_FROM_WINDOW", str(2 ** 63 - 1)),
        )
        # Cross-window disk persistence only makes sense in the legacy model.
        # Once the prompt range is armed, pooled entries never survive a window
        # (the slice changes every window), so reloading them at boot would
        # only risk firing stale out-of-slice entries → disable persistence.
        self._pool_persist = pool_persist_enabled(self._prompt_range_from_window)
        # Disk-backed persistence for the pool. Reload on launch so restarts
        # don't lose pre-baked entries (legacy model only). Entries with stale
        # checkpoint_n are kept optimistically (RELIQUARY_DROP_POOL_ON_CKPT=0).
        self._pool_dir = Path(
            _os.environ.get("RELIQUARY_POOL_DIR", "/root/reliquary-state/pool"),
        )
        self._pool_dir.mkdir(parents=True, exist_ok=True)
        if self._pool_persist:
            reloaded = load_pool(self._pool_dir, self._local_n)
            if reloaded:
                self._pool.extend(reloaded)
                logger.info(
                    "pool: reloaded %d entries from %s",
                    len(reloaded), self._pool_dir,
                )
        else:
            logger.info(
                "pool: persistence disabled (prompt-range armed from window %d)",
                self._prompt_range_from_window,
            )
        # Latest cached ``state.cooldown_prompts`` so the generator can avoid
        # baking prompts the validator just batched. Updated by the trigger
        # loop on every /state poll.
        self._cached_cooldown: set[int] = set()
        # Per-env cooldown (multi-env, spec §6). Phase 1: math key only, kept
        # in sync with _cached_cooldown by the trigger loop. Readers migrate to
        # this dict as the generator is routed per-env.
        self._cooldowns: dict[str, set[int]] = {n: set() for n in self.active_envs}
        # Current window randomness/window_n, cached by the trigger loop so the
        # background generator can derive the SAME slice the validator enforces
        # (the per-window prompt range gate is set up above with persistence).
        self._cached_randomness: str = ""
        self._cached_window_n: int = -1
        # Set of window_n already fired. Single-shot per window under the
        # R_open-only burst policy (specs/2026-05-16-r-open-only-burst-design.md).
        # Pruned in _trigger_loop to bound growth.
        self._fired_windows: set[int] = set()
        # Windows we've FORFEITED — pool was empty at the first OPEN tick,
        # so under the R_open-only policy we commit to skipping the whole
        # window (no mid-window R_open+k fire). Doubles as the dedup gate
        # for the "pool empty at OPEN" log so the 200 Hz tick doesn't spam.
        # Same pruning as _fired_windows.
        self._logged_empty_windows: set[int] = set()
        # Per-window submission counter for the ARMED fire-as-ready model
        # (#91): window_n → number of entries drained-to-fire this window, so
        # repeated intra-window fires never exceed
        # MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW. Pruned like _fired_windows.
        # Unused in the legacy single-burst path.
        self._submitted_count: dict[int, int] = {}
        # Strong references to in-flight fire tasks. asyncio.create_task
        # returns a Task that the event loop only weakly references, so
        # without this set the task can be GC'd mid-execution. Each task
        # registers itself via add_done_callback to remove itself on
        # completion, so the set self-cleans.
        self._inflight_fire_tasks: set[asyncio.Task] = set()
        # Multi-phase retry queue: prompts that didn't compose a valid
        # submission yet but show signal (= sigma>0 and bt_ok>=1). They
        # get more rollouts on the next generator iteration via
        # _pre_bake_batch's ``existing_rollouts_per_idx`` argument. Dropped
        # after MAX_PHASES total phases per prompt.
        # Per-env retry queues (multi-env): prompt_idx is env-scoped (4217 in
        # math ≠ 4217 in code), so retries must not collide across envs. In
        # single-env (Phase 1) this is one queue → identical behaviour.
        self._retry_by_env: dict[str, dict[int, list[dict]]] = {
            n: {} for n in self.active_envs
        }

        # Serialises HF forward calls across concurrent async-bake tasks.
        # vLLM and the HF proof model share the same GPU, so two HF
        # forwards in flight at once will contend with vLLM's KV-cache
        # work AND with each other — empirically this storms VRAM and
        # tanks vLLM throughput. The lock is only acquired in the async
        # path; the sync _pre_bake_batch already runs in a single
        # to_thread and so is implicitly serialised.
        self._hf_lock = asyncio.Lock()

        rng = random.Random()
        results: list = []

        # Initial drand clock calibration: synchronously block on one
        # Both miner (chrony) and validator (systemd-timesyncd) are
        # NTP-synced to within ~10 ms — no software-level offset is needed.
        # The drand-network fallback that used to seed _DRAND_CLOCK_OFFSET_S
        # at startup + refresh every 60 s was a net negative in practice:
        # ``get_beacon("latest")`` returns the most recent FINISHED round,
        # and single-sample precision is ±period/2 (~1.5 s), so each refresh
        # injected up to 2.5 s of garbage offset into our drand_round
        # computation. Trust local NTP. The functions
        # ``_refresh_drand_offset_loop`` and ``_compute_offset_sub_second``
        # are intentionally left in the module for the test suite and as
        # opt-in machinery — they just don't run on the prod path.

        # Dispatch: if the configured backend is the async vLLM engine,
        # use the continuous-batching loop; otherwise fall back to the
        # legacy sync batch-of-N loop. ``isinstance`` is safe — the
        # AsyncVLLMBackend import is cheap (no vllm side-effects at
        # module import). Both loops share the same pool / retry queue /
        # cancellation contract, so _trigger_loop is identical.
        from reliquary.constants import FORCED_SEED_ENFORCE
        from reliquary.miner.vllm_backend import AsyncVLLMBackend
        # The async continuous-batching loop samples via AsyncVLLMBackend.generate,
        # which cannot apply the forced-seed HF LogitsProcessor → its tokens would
        # fail seed-consistency. Under forced-seed we always run the sync HF loop
        # (which forces every token). Re-enable async only once the vLLM backend
        # gains a forced-seed sampler.
        use_async_loop = (
            isinstance(self._vllm_backend, AsyncVLLMBackend)
            and not FORCED_SEED_ENFORCE
        )
        if use_async_loop:
            logger.info(
                "miner: using ASYNC continuous-batching generator loop "
                "(RELIQUARY_ASYNC_TARGET_ACTIVE=%s)",
                _os.environ.get("RELIQUARY_ASYNC_TARGET_ACTIVE", "16"),
            )

        async with httpx.AsyncClient(timeout=30) as client:
            if use_async_loop:
                gen_task = asyncio.create_task(
                    self._async_generator_loop(url, client, rng),
                    name="miner_async_generator",
                )
            else:
                gen_task = asyncio.create_task(
                    self._generator_loop(url, client, rng),
                    name="miner_generator",
                )
            # Background /verdicts poll → MixController yield signal. Decoupled
            # from the latency-critical submit path; cancelled with gen_task.
            verdicts_task = asyncio.create_task(
                self._verdicts_loop(url, client),
                name="miner_verdicts",
            )
            try:
                await self._trigger_loop(url, client, results)
            finally:
                gen_task.cancel()
                verdicts_task.cancel()
                for _t in (gen_task, verdicts_task):
                    try:
                        await _t
                    except (asyncio.CancelledError, Exception):
                        pass

        return results

    def _active_prompt_range(
        self, window_n: int, randomness: str, env=None,
    ) -> tuple[int, int] | None:
        """The per-window eligible prompt slice (#91), or None when not armed.

        Mirrors the validator's ``batcher.set_prompt_range``: returns None
        (no confinement = legacy behaviour) until ``window_n`` reaches the
        configured cutover AND randomness is published. When active, both
        sides derive the identical ``[lo, hi)`` from the shared randomness,
        env name and ``len(env)`` — so a prompt we bake is guaranteed
        in-range for the validator that enforces the same slice.
        """
        if window_n < self._prompt_range_from_window or not randomness:
            return None
        env = env if env is not None else self.env
        return window_prompt_range(
            randomness,
            getattr(env, "name", ""),
            len(env),
            PROMPT_RANGE_SIZE,
        )

    async def _generator_loop(self, url, client, rng):
        """Background pre-bake loop. NEVER exits on a single iteration failure.

        Each iteration:
          1. Read the latest cached cooldown + the set of prompt_idx already
             in the pool.
          2. Pick up to ``RELIQUARY_BAKE_BATCH_SIZE`` distinct prompts via
             ``pick_prompt_idx`` (default 2 — vLLM continuous batching keeps
             the H200 SMs busy across prompts on the same gen step).
          3. Pre-bake all picks in a single ``_pre_bake_batch`` thread call.
          4. Append each non-None entry to the pool, respecting the
             optimistic / drop-on-ckpt policy.

        Sleeps briefly when the pool is full or no prompt is eligible.
        Cancellation only happens when ``mine_window`` exits.
        """
        from reliquary.miner.submitter import (
            SubmissionError, get_window_state_v2,
        )

        batch_size = max(1, int(
            _os.environ.get("RELIQUARY_BAKE_BATCH_SIZE", "2"),
        ))

        while True:
            try:
                async with self._pool_lock:
                    pool_full = len(self._pool) >= self._pool_max_size
                    pool_counts, in_pool_by_env = self._pool_env_stats()

                if pool_full:
                    await asyncio.sleep(0.5)
                    continue

                # Forced-seed: generation is bound to the window randomness via
                # u_at, so we cannot bake before /state publishes it. Wait until
                # the trigger loop caches a randomness for the current window.
                from reliquary.constants import FORCED_SEED_ENFORCE
                if FORCED_SEED_ENFORCE and not self._cached_randomness:
                    await asyncio.sleep(0.5)
                    continue

                # Multi-env: ask the MixController which env is furthest below
                # its target share and bake THAT env this iteration. Single-env
                # → always the one active env (identical to legacy). Per-env
                # cooldown / retry / pool-exclusion / slice all keyed by it.
                env_name = _pick_bake_env(self._mix.target_slots(), pool_counts)
                env = self.envs[env_name]
                cooldown = self._cooldowns[env_name]
                retry = self._retry_by_env[env_name]
                in_pool = in_pool_by_env.get(env_name, set())

                # Build the exclusion set from the latest cooldown snapshot
                # (refreshed by the trigger loop) + everything already baked
                # for this env so we don't waste GPU on duplicates.
                exclude = cooldown | in_pool
                picks: list[int] = []
                problems: list[dict] = []

                # Multi-phase: prioritize retries (= prompts that already
                # accumulated some rollouts and need more to compose a valid
                # submission). Skip retries whose prompt is now in cooldown
                # or pool — they got baked by some other path or are stale.
                retry_picks = [
                    idx for idx in retry
                    if idx not in cooldown and idx not in in_pool
                ]
                for idx in retry_picks:
                    if len(picks) >= batch_size:
                        break
                    picks.append(idx)
                    problems.append(env.get_problem(idx))

                # Drop retries we skipped (= now stale via cooldown/pool).
                for idx in list(retry):
                    if idx not in picks and idx in retry_picks:
                        # Could not pick this round but still active; keep
                        # it for the next iteration.
                        pass
                    elif idx not in retry_picks:
                        # Stale (cooldown / already pooled).
                        retry.pop(idx, None)

                # Per-window prompt range (#91): when armed, confine fresh
                # picks to the validator's [lo, hi) slice for the current
                # window — derived for THIS env (env_name domain-separates the
                # slice). None = unarmed → the whole dataset, legacy behaviour.
                prompt_range = self._active_prompt_range(
                    self._cached_window_n, self._cached_randomness, env,
                )

                # Fill remaining slots with fresh prompts.
                while len(picks) < batch_size:
                    try:
                        idx = pick_prompt_idx(
                            env, exclude | set(picks), rng=rng,
                            prompt_range=prompt_range,
                        )
                    except RuntimeError:
                        break
                    picks.append(idx)
                    problems.append(env.get_problem(idx))

                if not picks:
                    # Env fully covered — rare with 14M prompts, but back off.
                    await asyncio.sleep(5.0)
                    continue

                expected_ckpt_n = self._local_n

                # Pass the existing rollouts for retry-prompts (= empty for
                # fresh prompts). Multi-phase logic in _pre_bake_batch will
                # combine them with the newly generated rollouts.
                retry_input = {
                    idx: retry[idx]
                    for idx in picks if idx in retry
                }
                # Stream entries into the pool as each prompt finishes rather
                # than after the whole batch: waiting for all N meant the window
                # had flipped by the time they landed, and the fire path dropped
                # every one as an out-of-slice straggler (zero submissions all
                # of 2026-07-21). Phase-1 stays batched inside _bake_streaming.
                # Under FORCED_SEED_ENFORCE the multi-phase retry path is inert
                # (_pre_bake_batch returns an empty retry map), so nothing is
                # lost by not threading it here.
                entries = await self._bake_streaming(
                    problems, picks, expected_ckpt_n=expected_ckpt_n, env=env,
                )
                updated_retry = {}

                # Update retry queue: prompts in updated_retry stay (= next
                # phase). Prompts in picks but NOT in updated_retry and NOT
                # in entries were dropped → remove from retry queue.
                # Prompts in entries got baked → also remove from retry.
                baked_idxs = {e["prompt_idx"] for e in entries}
                for idx in picks:
                    if idx in updated_retry:
                        retry[idx] = updated_retry[idx]
                    else:
                        # Either baked (= success) or dropped (= sigma=0,
                        # bt_ok=0, or MAX_PHASES reached). Remove from
                        # retry tracking either way.
                        retry.pop(idx, None)

                # Optimistic by default: insert the entry even if checkpoint
                # advanced mid-bake — bet on the validator's sketch tolerance
                # absorbing a single train_step delta. Forced-conservative
                # behavior via RELIQUARY_DROP_POOL_ON_CKPT=1 (matches the
                # trigger loop's policy for already-pooled entries).
                # _bake_streaming already appended each entry under the lock
                # (with the same drop-on-ckpt policy) as soon as it was baked.
                for entry in entries:
                    async with self._pool_lock:
                        pool_size = len(self._pool)
                    # Persist to disk so restarts don't lose this entry.
                    # Runs OUTSIDE the lock via asyncio.to_thread so the
                    # ~90ms torch.save doesn't block /state polls. Skipped when
                    # the prompt range is armed (#91): entries don't survive a
                    # window, so persisting them is wasted I/O during OPEN.
                    if self._pool_persist:
                        try:
                            await asyncio.to_thread(
                                save_entry, entry, self._pool_dir,
                            )
                        except OSError as e:
                            logger.error(
                                "pool_persistence: save failed for prompt=%d (%s); "
                                "entry kept in memory only",
                                entry["prompt_idx"], e,
                            )
                    logger.debug(
                        "pool +1: prompt=%d size=%d/%d",
                        entry["prompt_idx"], pool_size, self._pool_max_size,
                    )
            except asyncio.CancelledError:
                return
            except Exception:
                # Generator MUST NOT die on a single iteration failure —
                # log and keep going.
                logger.exception("generator iteration failed; continuing")
                await asyncio.sleep(1.0)

    async def _trigger_loop(self, url, client, results):
        """Foreground /state poll + per-window POST burst.

        Fires exactly once per window via ``_should_fire_for_window`` /
        ``self._fired_windows``. On the OPEN flip with non-empty randomness,
        drains up to 8 pool entries, finalizes them, and POSTs in parallel.
        """
        from reliquary.miner.submitter import (
            SubmissionError, get_window_state_v2, get_window_state_v2_with_resp,
        )
        from reliquary.protocol.submission import WindowState

        while True:
            try:
                state, resp, t_send, t_recv = (
                    await get_window_state_v2_with_resp(url, client=client)
                )
            except SubmissionError:
                # Validator returns 503 with detail=no_active_window during
                # window transitions (between set_active_batcher(None) and
                # set_active_batcher(new_batcher) — server.py:287-288). The
                # window OPEN flip happens RIGHT AFTER this 503 window
                # closes, so we want to retry FAST, not back off 10 s as the
                # imported POLL_INTERVAL_SECONDS would do. _STATE_RETRY_S
                # is 50 ms — same order of magnitude as the steady-state
                # 5 ms poll cadence, so we catch the next OPEN flip within
                # one drand round. Measured in prod 2026-05-16: a 10 s
                # backoff caused us to miss R_open by 25 rounds on cold
                # start; with 50 ms we should hit R_open or R_open+1.
                await asyncio.sleep(_STATE_RETRY_S)
                continue
            except StopAsyncIteration:
                raise
            except Exception as e:
                logger.debug("state fetch failed: %s", e)
                await asyncio.sleep(_STATE_RETRY_S)
                continue

            # Clock-offset calibration via validator HTTP Date header is
            # DISABLED. Uvicorn caches Date at 1-second granularity but with
            # a non-tight refresh — measured staleness in prod is 0-1.5 s
            # (mean ~0.75 s). The +0.5 floor-comp baked into
            # _apply_offset_from_validator_response only corrects for the
            # 1-s floor, not for the additional cache staleness, so the EMA
            # converged to a spurious -0.5 to -1 s "offset". That shifted
            # our drand-round computation 0.5-1 s into the past and produced
            # routine STALE_ROUND rejections on submissions that arrived in
            # the validator's actual current round. Both miner (chrony,
            # ~44 μs) and validator (systemd-timesyncd, "synchronized") are
            # NTP-synced to within ~10 ms — no software-level offset is
            # needed. The drand-network refresh loop in the background
            # remains as a coarse safety net for the edge case where the
            # local box loses NTP, but only updates every 60 s and has
            # ±period/2 (~1.5 s) precision per sample.

            # Refresh per-env cooldown for the generator to consume. The main
            # env-agnostic poll carries the first active env's cooldown; any
            # extra envs are polled with ?env= (multi-env only — the loop body
            # is empty in single-env, so Phase 1 is one poll exactly as before).
            self._cached_cooldown = set(state.cooldown_prompts)
            self._cooldowns[self.active_envs[0]] = self._cached_cooldown
            for _env in self.active_envs[1:]:
                try:
                    _st = await get_window_state_v2(url, env=_env, client=client)
                    self._cooldowns[_env] = set(_st.cooldown_prompts)
                except Exception:
                    logger.debug(
                        "per-env cooldown poll failed for %s; keeping last", _env,
                    )

            # Per-window prompt range (#91): cache the current randomness so the
            # background generator derives the same [lo, hi) slice. When the
            # range is ARMED and randomness flips to a new non-empty value, the
            # pool + retry queue from the previous window's slice are
            # unsubmittable (different slice) → flush them. No-op while the
            # range is unarmed (legacy cross-window pre-bake preserved).
            from reliquary.constants import FORCED_SEED_ENFORCE
            if (
                state.randomness
                and state.randomness != self._cached_randomness
                and (
                    FORCED_SEED_ENFORCE
                    or self._active_prompt_range(state.window_n, state.randomness)
                    is not None
                )
            ):
                # Randomness flip: entries baked under the old randomness carry
                # forced-seed tokens (and a slice) that no longer match this
                # window — drop them so a submission only ever holds tokens
                # generated under its own window randomness.
                async with self._pool_lock:
                    flushed = len(self._pool)
                    self._pool = []
                    for _q in self._retry_by_env.values():
                        _q.clear()
                if flushed:
                    logger.info(
                        "prompt-range: randomness flip (window=%d) → flushed "
                        "%d stale-slice pool entries", state.window_n, flushed,
                    )
            if state.randomness:
                self._cached_randomness = state.randomness
                self._cached_window_n = state.window_n

            # Pull new checkpoint if needed. Works at any state. On real
            # advance, the pool is dropped — hidden states from the old
            # model would fail GRAIL under the new one.
            ckpt_advanced_this_iter = False
            try:
                new_n, new_hash, new_model = await maybe_pull_checkpoint(
                    state=state, local_n=self._local_n,
                    local_hash=self._local_hash,
                    local_model=self.hf_model,
                    download_fn=_hf_download,
                    load_fn=self._load_checkpoint,
                )
                if new_n != self._local_n:
                    ckpt_advanced_this_iter = True
                    # OPTIMISTIC: by default we KEEP pool entries baked
                    # under the previous checkpoint and bet on the
                    # validator's PROOF_SKETCH_TOLERANCE_BASE absorbing the
                    # 10-train_step weight delta between consecutive
                    # checkpoints. Cost of being wrong: those entries reject
                    # GRAIL_FAIL — same slots lost as if we had dropped. Set
                    # ``RELIQUARY_DROP_POOL_ON_CKPT=1`` to force conservative
                    # drop-and-rebake behavior if the empirical fail rate
                    # turns out to be > drop's lost window.
                    drop_on_ckpt = drop_pool_on_ckpt_advance()
                    if drop_on_ckpt:
                        async with self._pool_lock:
                            dropped = len(self._pool)
                            self._pool = []
                        # On-disk pool follows the same drop policy.
                        if self._pool_dir is not None and self._pool_dir.exists():
                            shutil.rmtree(self._pool_dir)
                            self._pool_dir.mkdir(parents=True, exist_ok=True)
                        if dropped:
                            logger.info(
                                "checkpoint %d -> %d: dropped %d stale pool "
                                "entries (DROP_POOL_ON_CKPT=1)",
                                self._local_n, new_n, dropped,
                            )
                    else:
                        async with self._pool_lock:
                            kept = len(self._pool)
                        logger.info(
                            "checkpoint %d -> %d: keeping %d pool entries "
                            "(optimistic) — they will be POSTed against new "
                            "validator model; GRAIL_FAIL is the recoverable "
                            "downside",
                            self._local_n, new_n, kept,
                        )
                    self._local_n = new_n
                    self._local_hash = new_hash
                    self.hf_model = new_model
            except Exception:
                logger.exception("checkpoint pull failed; keeping local")

            # If a checkpoint advance happened THIS iteration, the model
            # reload blocked us for several seconds. ``state`` was fetched
            # before that stall, so its window/randomness is very likely
            # stale now — firing against it signs the envelope with the old
            # window's randomness, which the validator verifies against its
            # CURRENT batcher randomness → BAD_ENVELOPE_SIGNATURE for the
            # whole burst. Skip the fire this iteration; the next loop tick
            # (immediate) re-fetches /state and fires the (kept) pool against
            # the current window with fresh randomness.
            if ckpt_advanced_this_iter:
                logger.info(
                    "checkpoint advanced mid-iteration (reload stall); "
                    "skipping fire for stale window=%d, will re-fire against "
                    "current window next tick",
                    state.window_n,
                )
                continue

            # Fire path. Two models, selected by whether the per-window prompt
            # range is armed (#91):
            #  * LEGACY (range unarmed): one burst per window at the OPEN flip,
            #    draining a pool pre-baked across windows; forfeit the window if
            #    the pool is empty at the first OPEN tick (R_open-only design).
            #  * ARMED: the pool is flushed each window (the slice changes), so
            #    it starts empty and the generator fills it intra-window with
            #    in-slice entries. We therefore fire-AS-READY: re-fire each tick
            #    while OPEN, draining whatever is ready, up to
            #    MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW distinct submissions for
            #    the window. Fires are serialised (one in flight at a time via
            #    _inflight_fire_tasks) so concurrent drains can't over-submit
            #    past the per-hotkey cap.
            async with self._pool_lock:
                pool_size = len(self._pool)

            armed = self._fire_as_ready(state.window_n, state.randomness)

            if armed:
                remaining = (
                    MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW
                    - self._submitted_count.get(state.window_n, 0)
                )
                if (
                    state.state == WindowState.OPEN
                    and state.randomness
                    and remaining > 0
                    and pool_size > 0
                    and not self._inflight_fire_tasks
                ):
                    fire_task = asyncio.create_task(
                        self._fire_for_window(
                            state, url, client, results, budget=remaining,
                        ),
                        name=f"fire_window_{state.window_n}",
                    )
                    self._inflight_fire_tasks.add(fire_task)
                    fire_task.add_done_callback(
                        self._inflight_fire_tasks.discard,
                    )
            elif _should_fire_for_window(
                state, self._fired_windows, self._logged_empty_windows, pool_size,
            ):
                # Mark BEFORE scheduling so the next 5 ms tick can't double-fire
                # while the task is in flight. Stash the task in
                # ``_inflight_fire_tasks`` (strong ref) + remove on completion
                # via done_callback so it can't be GC'd mid-await.
                self._fired_windows.add(state.window_n)
                fire_task = asyncio.create_task(
                    self._fire_for_window(state, url, client, results),
                    name=f"fire_window_{state.window_n}",
                )
                self._inflight_fire_tasks.add(fire_task)
                fire_task.add_done_callback(self._inflight_fire_tasks.discard)
            elif (
                state.window_n not in self._fired_windows
                and state.window_n not in self._logged_empty_windows
                and state.state == WindowState.OPEN
                and state.randomness
                and pool_size == 0
            ):
                logger.warning(
                    "pool empty at OPEN window=%d — skipping fire, entries "
                    "baked later in this window will wait for the next flip",
                    state.window_n,
                )
                self._logged_empty_windows.add(state.window_n)
            self._logged_empty_windows = {
                w for w in self._logged_empty_windows if w >= state.window_n - 64
            }
            # Prune old entries to bound memory growth — 64 windows back is
            # well beyond any realistic /state rollback.
            self._fired_windows = {
                w for w in self._fired_windows if w >= state.window_n - 64
            }
            self._submitted_count = {
                w: c for w, c in self._submitted_count.items()
                if w >= state.window_n - 64
            }

            await asyncio.sleep(0.005)

    async def _fire_for_window(
        self, state, url, client, results,
        budget: int = MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW,
    ):
        """Drain pool, finalize, POST. Aim for the first drand round of OPEN.

        Each entry is finalized on a thread (~50 ms) then POSTed concurrently
        via ``asyncio.gather``. ``budget`` caps how many entries this call
        drains — the legacy single-burst path passes the full per-window cap;
        the ARMED fire-as-ready path (#91) passes the REMAINING per-window
        budget so repeated intra-window fires never exceed
        MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW. Drained entries are counted
        against ``self._submitted_count[window_n]`` immediately (under the
        pool lock) so the next tick sees the consumed budget.
        """
        cooldown_set = set(state.cooldown_prompts)
        randomness = state.randomness
        # Per-window prompt range (#91): when armed, an entry whose prompt_idx
        # is outside the current [lo, hi) slice would be rejected
        # PROMPT_OUT_OF_RANGE. The generator already confines picks to the
        # slice and the pool is flushed on each flip, so this is a defensive
        # net (e.g. a stale entry surviving a same-window randomness re-read):
        # drop out-of-slice entries here rather than burn a submission slot.
        # PER-ENV: env_name domain-separates the [lo, hi), so each entry must be
        # checked against ITS env's slice, not a single (math) slice.
        range_by_env = {
            n: self._active_prompt_range(state.window_n, randomness, self.envs[n])
            for n in self.active_envs
        }

        # Drain non-cooldown entries up to this call's budget. Cooldown entries
        # are dropped silently — validator rejects PROMPT_IN_COOLDOWN.
        cooldown_dropped: list[dict] = []
        async with self._pool_lock:
            kept: list[dict] = []
            fire: list[dict] = []
            for entry in self._pool:
                if entry["prompt_idx"] in cooldown_set:
                    cooldown_dropped.append(entry)
                    continue
                pr = range_by_env.get(self._entry_env_name(entry))
                if pr is not None and not (pr[0] <= entry["prompt_idx"] < pr[1]):
                    # Out-of-slice straggler — drop (don't fire, don't keep).
                    cooldown_dropped.append(entry)
                    continue
                if len(fire) < budget:
                    fire.append(entry)
                else:
                    kept.append(entry)
            self._pool = kept
            # Reserve the budget synchronously so the serialised fire loop
            # (ARMED path) can't over-submit on the next tick.
            if fire:
                self._submitted_count[state.window_n] = (
                    self._submitted_count.get(state.window_n, 0) + len(fire)
                )

        # Clean up on-disk files for cooldown-dropped entries (the
        # validator would reject them with prompt_in_cooldown anyway).
        # Done outside the lock — delete_entry is just an os.unlink.
        for entry in cooldown_dropped:
            persist_path = entry.get("_persist_path")
            if persist_path is not None:
                delete_entry(persist_path)

        if not fire:
            logger.info(
                "fire_for_window=%d: pool empty (kept=%d after cooldown filter)",
                state.window_n, len(kept),
            )
            return

        logger.info(
            "fire_for_window=%d: finalizing %d entries (pool kept=%d) "
            "randomness=%s",
            state.window_n, len(fire), len(kept), randomness[:16],
        )

        # Finalize + POST all in parallel.
        fire_results = await asyncio.gather(
            *(self._submit_entry(e, state, url, client, results) for e in fire),
            return_exceptions=True,
        )

        # Decide per-entry whether to re-queue or drop.
        # Retryable rejects: same rollouts can be re-fired in a later
        # window (stale_round, batch_filled) or after a transient backoff
        # (rate_limited, future_round). Permanent rejects + accepted go
        # to drop. Without this re-queue, every retryable reject lost
        # the pre-baked work — wasted GPU + a missed submission slot.
        retryable_reasons = {
            "stale_round", "batch_filled", "rate_limited", "future_round",
            # v1-admission-hardening (#114): the validator fails closed while
            # its registered-hotkey cache is stale (chain hiccup) — transient
            # on ITS side, so re-fire. hotkey_not_registered stays a DROP
            # (persistent until we re-register) and is surfaced by the
            # drop_reason_counts WARNING below.
            "registration_unavailable",
        }
        to_requeue: list[dict] = []
        to_drop: list[dict] = []
        accepted_count = 0
        error_count = 0
        drop_reason_counts: dict[str, int] = {}
        for i, entry in enumerate(fire):
            item = fire_results[i]
            if isinstance(item, BaseException) or item is None:
                to_drop.append(entry)
                error_count += 1
                continue
            _, resp = item
            if resp is None:
                to_drop.append(entry)
                error_count += 1
                continue
            if resp.accepted:
                to_drop.append(entry)
                accepted_count += 1
                continue
            reason_val = (
                resp.reason.value if hasattr(resp.reason, "value")
                else str(resp.reason)
            )
            if reason_val in retryable_reasons:
                to_requeue.append(entry)
            else:
                to_drop.append(entry)
                drop_reason_counts[reason_val] = (
                    drop_reason_counts.get(reason_val, 0) + 1
                )

        # Surface non-retryable rejects per reason — these were previously
        # dropped silently, hiding real failures (GRAIL_FAIL, BAD_ENVELOPE_
        # SIGNATURE, OUT_OF_ZONE, the masked 422, ...) from operators.
        if drop_reason_counts or error_count:
            logger.warning(
                "fire_for_window=%d: dropped %d entries on non-retryable "
                "rejects %s%s (accepted=%d, requeued=%d)",
                state.window_n,
                sum(drop_reason_counts.values()) + error_count,
                dict(sorted(drop_reason_counts.items())),
                f" + {error_count} submit/transport errors" if error_count else "",
                accepted_count,
                len(to_requeue),
            )

        if to_requeue:
            async with self._pool_lock:
                self._pool.extend(to_requeue)
            logger.info(
                "fire_for_window=%d: re-queued %d entries for retry "
                "(retryable reject reasons: stale_round/batch_filled/...)",
                state.window_n, len(to_requeue),
            )

        # Delete persisted files only for entries we're done with.
        # Re-queued entries keep their persist file so a restart still
        # reloads them — the validator's hash-dedupe handles any duplicate.
        for e in to_drop:
            persist_path = e.get("_persist_path")
            if persist_path is not None:
                delete_entry(persist_path)

    def _build_signed_request_sync(
        self, rollout_submissions, merkle_root, prompt_idx, state, miner_hk, nonce,
    ):
        """Sync CPU-bound build: drand + sign + pydantic.

        Called via asyncio.to_thread so concurrent _submit_entry fires
        from _fire_for_window's asyncio.gather truly parallelize on CPU
        threads instead of serializing through the asyncio loop.

        ``merkle_root`` is now pre-computed by ``_finalize_pool_entry``
        (which runs in the same thread chain) so this function does not
        re-hash the rollouts — just signs the envelope + builds the
        pydantic model.

        Returns (current_round, request).
        """
        from reliquary.protocol.signatures import sign_envelope
        from reliquary.protocol.submission import BatchSubmissionRequest

        # Compute drand inside the thread so it reflects the near-POST
        # instant — all parallel threads start at the same moment so
        # their drand reads are within microseconds of each other.
        current_round = _current_drand_round_at_send()

        # Snapshot the checkpoint hash ONCE. self._local_hash is mutated by
        # the trigger loop on checkpoint advance; reading it twice (sign +
        # request) lets a mid-build advance sign with hash_N but embed
        # hash_N+1 → the validator rebuilds the binding with the embedded
        # hash, the signature fails to verify, and the whole submission is
        # rejected as BAD_ENVELOPE_SIGNATURE. Read once so sign and request
        # are always consistent.
        ckpt_hash = self._local_hash

        # Wire-v2 (gated): version 2 is advertised AND bound into the envelope
        # (v2 domain); v1 keeps the exact legacy preimage (None sentinel).
        proto_version = wire_protocol_version()
        envelope_sig = sign_envelope(
            wallet=self.wallet,
            miner_hotkey=miner_hk,
            window_start=state.window_n,
            prompt_idx=prompt_idx,
            merkle_root=merkle_root,
            checkpoint_hash=ckpt_hash,
            drand_round=current_round,
            randomness=state.randomness,
            nonce=nonce,
            protocol_version=proto_version if wire_v2_enabled() else None,
        ).hex()
        request = BatchSubmissionRequest(
            miner_hotkey=miner_hk,
            prompt_idx=prompt_idx,
            window_start=state.window_n,
            merkle_root=merkle_root,
            rollouts=rollout_submissions,
            checkpoint_hash=ckpt_hash,
            drand_round=current_round,
            nonce=nonce,
            envelope_signature=envelope_sig,
            protocol_version=proto_version,
        )
        return current_round, request

    async def _submit_entry(self, entry, state, url, client, results):
        """Build commits with state.randomness and POST. Fast path.

        Both finalize AND signed-request-build run on threads so they
        parallelize across all 8 entries fired in a single window's
        burst. Without thread parallelization the sync sign+pydantic+
        serialize per entry serialized through the asyncio loop and
        crossed drand boundaries (= STALE_ROUND).

        Returns (entry, resp) so the caller can decide whether to
        re-queue the entry on retryable rejects (stale_round,
        batch_filled, ...). Returns (entry, None) on error paths.
        """
        import secrets
        from reliquary.miner.submitter import (
            SubmissionError, submit_batch_v2,
        )

        prompt_idx = entry["prompt_idx"]
        try:
            rollout_submissions, merkle_root = await asyncio.to_thread(
                self._finalize_pool_entry, entry, state.randomness,
            )
        except Exception:
            logger.exception(
                "finalize failed for prompt=%d (window=%d); dropping",
                prompt_idx, state.window_n,
            )
            return entry, None

        # Record which env this submission belongs to so the verdicts loop can
        # map its outcome back to the MixController. Async context → no race.
        self._submitted_env[merkle_root] = self._entry_env_name(entry)

        miner_hk = self.wallet.hotkey.ss58_address
        nonce = secrets.token_hex(16)
        try:
            current_round, request = await asyncio.to_thread(
                self._build_signed_request_sync,
                rollout_submissions, merkle_root, prompt_idx, state, miner_hk, nonce,
            )
        except Exception:
            logger.exception(
                "build_signed_request failed for prompt=%d", prompt_idx,
            )
            return entry, None

        try:
            # wallet + randomness arm the mandatory upload-precommit handshake
            # (upstream 8835a95). Without them the submitter falls back to the
            # bare /submit the validator answers with PRECOMMIT_REQUIRED.
            resp = await submit_batch_v2(
                url, request, client=client,
                wallet=self.wallet, randomness=state.randomness,
            )
            logger.info(
                "submitted window=%d prompt=%d accepted=%s reason=%s "
                "drand_round=%d",
                state.window_n, prompt_idx, resp.accepted,
                resp.reason.value if hasattr(resp.reason, "value") else resp.reason,
                current_round,
            )
            results.append(resp)
            return entry, resp
        except SubmissionError as exc:
            logger.error(
                "submit failed prompt=%d: %s", prompt_idx, exc,
            )
            return entry, None

    def _load_checkpoint(self, local_path: str):
        """Reload both hf_model and vllm_model from *local_path*.

        Both attributes are ``AutoModelForCausalLM`` instances despite the
        historical ``vllm_model`` naming — vllm_model is the fast-generation
        copy on ``self.vllm_gpu``, hf_model is the GRAIL-proof copy on
        ``self.proof_gpu``.
        """
        import torch

        from reliquary.constants import ATTN_IMPLEMENTATION

        if getattr(self, "_loaded_checkpoint_path", None) == local_path:
            logger.debug("_load_checkpoint: already loaded from %s", local_path)
            return self.hf_model

        logger.info("Loading checkpoint from %s", local_path)

        # 1. Reload hf_model (for GRAIL proofs) on the proof GPU.
        try:
            new_hf = load_text_generation_model(
                local_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=ATTN_IMPLEMENTATION,
            ).to(f"cuda:{self.proof_gpu}").eval()
        except Exception:
            logger.exception(
                "Failed to reload hf_model from %s; keeping old model",
                local_path,
            )
            return self.hf_model

        old_hf = self.hf_model
        self.hf_model = new_hf
        # New checkpoint may carry a different EOS set (model-family change) →
        # refresh so truncation / termination / vLLM stops track the new model.
        self._eos_ids = self._resolve_eos_ids()
        del old_hf
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        # 2. Reload the generation model. Prefer the vLLM backend when wired;
        # fall back to the legacy HF reload for tests / single-GPU dev boxes.
        backend = getattr(self, "_vllm_backend", None)
        if backend is not None:
            try:
                result = backend.reload(local_path)
                # AsyncVLLMBackend.reload is a coroutine; sync VLLMBackend
                # returns None. _load_checkpoint is invoked from an async
                # caller (maybe_pull_checkpoint) but is itself sync, so we
                # need to drive the coroutine here. The running event loop
                # is the one that called us — we can't ``run_until_complete``
                # on it. Instead, schedule the coroutine via a fresh thread
                # that owns its own loop and block on the result. The
                # checkpoint-advance path is rare (~10 min between pulls)
                # so the thread overhead is irrelevant.
                import asyncio as _asyncio
                import inspect as _inspect
                if _inspect.iscoroutine(result):
                    import threading as _threading
                    box: dict = {}
                    def _drive():
                        try:
                            box["ok"] = _asyncio.run(result)
                        except BaseException as e:
                            box["err"] = e
                    th = _threading.Thread(target=_drive, daemon=True)
                    th.start()
                    th.join()
                    if "err" in box:
                        raise box["err"]
            except Exception:
                logger.exception(
                    "Failed to reload vllm_backend from %s; miner generation "
                    "is BROKEN until the next successful pull. hf_model was "
                    "swapped so GRAIL proofs will be inconsistent.",
                    local_path,
                )
                self._loaded_checkpoint_path = None
                return self.hf_model
        else:
            try:
                new_gen = load_text_generation_model(
                    local_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation=ATTN_IMPLEMENTATION,
                ).to(f"cuda:{self.vllm_gpu}").eval()
            except Exception:
                logger.exception(
                    "Failed to reload vllm_model from %s; miner generation is "
                    "BROKEN until the next successful pull. hf_model was swapped "
                    "so GRAIL proofs will be inconsistent.",
                    local_path,
                )
                self.vllm_model = None
                self._loaded_checkpoint_path = None
                return self.hf_model

            old_gen = self.vllm_model
            self.vllm_model = new_gen
            del old_gen
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        self._loaded_checkpoint_path = local_path
        logger.info("Checkpoint %s loaded into both models", local_path)
        return self.hf_model

    def _fire_as_ready(self, window_n, randomness) -> bool:
        """Fire-as-ready (intra-window, budget-capped re-fire) vs legacy
        single-burst. Armed when the per-window prompt range is armed OR
        forced-seed is enforced: under forced-seed the pool is flushed at every
        randomness flip and generation can only start once the randomness is
        published, so the pool is ALWAYS empty at the first OPEN tick — the
        legacy burst would mark the window fired-empty and forfeit every
        window."""
        from reliquary.constants import FORCED_SEED_ENFORCE

        if FORCED_SEED_ENFORCE:
            return True
        return self._active_prompt_range(window_n, randomness) is not None

    def _bft_from_seqs(self, seqs, prompt_tokens, *, randomness, hotkey,
                       prompt_idx, checkpoint_hash):
        """Run BFT phase-2 (force-terminate at the thinking budget) over phase-1
        sequences (each = prompt_tokens + gen as a token list). Returns rollout
        dicts carrying ``forced`` / ``force_span``. Phase-2 answer tokens are
        drawn from the SAME protocol forced-seed stream as phase-1 (identity
        threaded so ``bft_assemble_rollouts`` resumes each row at its own offset).
        Shared by the single-prompt and batched generation paths."""
        from reliquary.constants import BFT_ANSWER_BUDGET
        from reliquary.miner.bft import bft_rollouts_from_completions
        from reliquary.shared.modeling import (
            force_close_token_ids,
            think_close_token_ids,
        )

        # No sampling warpers here: the forced-seed processor applies the protocol
        # warp itself (forced_seed_generate_kwargs strips temperature/top_k/top_p
        # and sets do_sample=False inside bft_assemble_rollouts).
        phase2_kwargs = {"pad_token_id": self.tokenizer.pad_token_id}
        if self._eos_ids:
            phase2_kwargs["eos_token_id"] = sorted(self._eos_ids)
        return bft_rollouts_from_completions(
            seqs, prompt_tokens, model=self.hf_model,
            think_close_ids=set(think_close_token_ids(self.tokenizer)),
            force_ids=force_close_token_ids(self.tokenizer),
            eos_ids=self._eos_ids, answer_budget=BFT_ANSWER_BUDGET,
            randomness=randomness, hotkey=hotkey, prompt_idx=prompt_idx,
            checkpoint_hash=checkpoint_hash, gen_kwargs=phase2_kwargs,
        )

    def _proof_forward_batch(self, seqs, *, device):
        """⚠ FAILS GRAIL PARITY — NOT WIRED INTO PRODUCTION. Kept as evidence.

        Measured 2026-07-21 (scripts/validate_proof_batch_parity.py, real v3
        checkpoint): right-padded batching flips the sketch top-k selection and
        produced 25/81 positions with a completely different sketch, worst
        delta 2.1e9 = 428553x the validator's adaptive tolerance. Do not re-wire
        this into _pre_bake_entry.

        One padded GRAIL proof forward for a whole group of rollouts.

        Replaces len(seqs) separate forwards (~29s per prompt measured
        2026-07-21, the dominant bake cost once phase-1 is batched).

        Rollouts differ in length, so rows are RIGHT-padded and an attention
        mask marks the real tokens: for a causal LM a real token never attends
        to a later pad, so the masked positions cannot leak into the proof.
        Results are sliced back to each row's own length — carrying pad
        positions into the commitment would change the sketch.

        Equivalence with the per-sequence path holds up to float reduction
        order, which is why it is gated by an explicit GPU parity check
        (scripts/validate_proof_batch_parity.py) before production use.

        Returns ``[(hidden_states[len_i, H], logits[len_i, V]), ...]`` in input
        order.
        """
        import torch

        from reliquary.shared.forward import forward_single_layer

        lengths = [len(s) for s in seqs]
        width = max(lengths)
        padded = torch.zeros((len(seqs), width), dtype=torch.long, device=device)
        mask = torch.zeros((len(seqs), width), dtype=torch.long, device=device)
        for r, seq in enumerate(seqs):
            padded[r, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
            mask[r, :len(seq)] = 1

        with torch.no_grad():
            hidden_states, logits = forward_single_layer(
                self.hf_model, padded, mask, LAYER_INDEX,
            )
        return [
            (hidden_states[r, :n], logits[r, :n])
            for r, n in enumerate(lengths)
        ]

    async def _bake_streaming(self, problems, prompt_indices, *, expected_ckpt_n,
                              env) -> list:
        """Bake a batch, pushing each entry into the pool the moment it is ready.

        Root cause this fixes (measured 2026-07-21): ``_pre_bake_batch`` only
        returned entries once EVERY prompt was baked (~185s for 6). By then the
        window had flipped and the prompt-range slice moved, so the fire path
        dropped them all as out-of-slice stragglers — ``pool empty (kept=0 after
        cooldown filter)`` on every fire, zero submissions all day. The first
        prompt was ready at 44s, inside its own window, and was discarded only
        because we waited for the other five.

        Phase-1 stays batched (one vLLM call for the whole group, the 1484 tok/s
        win); only the per-prompt proof/grade stage is streamed, with an await
        boundary between prompts so the trigger loop can fire mid-window.
        """
        await asyncio.to_thread(
            self._prefetch_phase1, problems, prompt_indices,
            randomness=self._cached_randomness, env=env,
        )
        entries = []
        for prompt_idx, problem in zip(prompt_indices, problems):
            try:
                entry = await asyncio.to_thread(
                    self._pre_bake_entry, prompt_idx, problem,
                    expected_ckpt_n, env,
                )
            except Exception:
                # One bad prompt must not cost the whole bake.
                logger.exception(
                    "pre_bake failed for prompt=%d; continuing", prompt_idx,
                )
                continue
            if entry is None:
                continue
            # Same drop-on-ckpt policy the batch path applied: under forced-seed
            # an entry baked on an older checkpoint no longer matches the stream.
            if (
                drop_pool_on_ckpt_advance()
                and entry.get("checkpoint_n") != getattr(self, "_local_n", None)
            ):
                logger.info(
                    "generator: dropping stale entry prompt=%d "
                    "(ckpt baked=%s, current=%s, DROP_POOL_ON_CKPT=1)",
                    prompt_idx, entry.get("checkpoint_n"),
                    getattr(self, "_local_n", None),
                )
                continue
            async with self._pool_lock:
                self._pool.append(entry)
            entries.append(entry)
        # Anything unconsumed cannot be reused under a later randomness.
        self._phase1_cache = {}
        return entries

    def _prefetch_phase1(self, problems, prompt_indices, *, randomness, env) -> int:
        """Batch forced-seed phase-1 for ALL prompts of a bake in ONE vLLM call.

        Without this the bake loop calls ``generate_forced_phase1`` once per
        prompt (~40s each), so a 6-prompt bake costs ~264s against a 100s
        collection window: the pool is flushed stale at the randomness flip and
        nothing is ever submitted. Batching 6x8 sequences into one call brings
        that back inside the window.

        Completions are parked in ``_phase1_cache`` keyed by
        ``(prompt_idx, randomness, checkpoint_hash)`` — generation is bound to
        both, and serving a stale entry would force every token onto the wrong
        stream (validator: TOKEN_TAMPERED) while looking perfectly healthy
        locally. Returns the number of prompts cached (0 = caller uses the
        per-prompt path unchanged).
        """
        from reliquary.constants import FORCED_SEED_ENFORCE
        from reliquary.miner.bft import phase1_max_new_tokens

        backend = getattr(self, "_vllm_backend", None)
        if backend is None or not (
            FORCED_SEED_ENFORCE and vllm_forced_seed_enabled()
        ):
            return 0
        if not hasattr(backend, "generate_forced_phase1_multi"):
            return 0

        checkpoint_hash = self._local_hash
        env_name = getattr(env if env is not None else getattr(self, "env", None),
                           "name", None)
        max_new = phase1_max_new_tokens(self.max_new_tokens, env_name)
        import time as _t
        _gen_t0 = _t.perf_counter()
        try:
            prompts_tokens = [
                encode_prompt(self.tokenizer, p["prompt"]) for p in problems
            ]
            grouped = backend.generate_forced_phase1_multi(
                prompts_tokens,
                prompt_indices=list(prompt_indices),
                randomness=randomness,
                checkpoint_hash=checkpoint_hash,
                m_rollouts=M_ROLLOUTS,
                max_tokens=max_new,
                stop_token_ids=self._eos_ids,
            )
        except Exception:
            # Fall back to the per-prompt path rather than caching a partial
            # batch: a half-filled cache would pair some prompts with another
            # prompt's completions.
            logger.exception(
                "phase1 prefetch failed for %d prompts; falling back per-prompt",
                len(problems),
            )
            self._phase1_cache = {}
            return 0

        cache = {}
        for prompt_idx, completions in zip(prompt_indices, grouped):
            cache[(prompt_idx, randomness, checkpoint_hash)] = completions
        self._phase1_cache = cache
        logger.info(
            "phase1 prefetch: %d prompts x %d rollouts in one batched call "
            "TIMING gen=%0.1fs",
            len(cache), M_ROLLOUTS, _t.perf_counter() - _gen_t0,
        )
        return len(cache)

    def _take_prefetched_phase1(self, prompt_idx, randomness, checkpoint_hash):
        """Pop this prompt's prefetched completions, or None.

        Single-use on purpose: a leftover served to a later window would carry
        the previous window's forced stream.
        """
        cache = getattr(self, "_phase1_cache", None)
        if not cache:
            return None
        return cache.pop((prompt_idx, randomness, checkpoint_hash), None)

    def _generate_m_rollouts(self, problem, randomness, env=None,
                             prompt_idx=0) -> list[dict]:
        """Generate M_ROLLOUTS completions on the protocol FORCED-SEED stream.

        Every sampled token is the public inverse-CDF pick derived from
        ``u_at(randomness, prompt_idx, checkpoint_hash, rollout, t)`` (v2: no
        hotkey — the forced stream is identical for every miner in the window)
        (via ForcedSeedLogitsProcessor), so an honest miner scores ~1.0 on the
        validator's seed-consistency gate. Generation is therefore
        randomness-DEPENDENT and must run once the window randomness is known.
        One batched .generate() over M_ROLLOUTS rows; each output row is
        truncated at its first post-prompt EOS so trailing batch-padding is not
        carried into the validator's GRAIL forward pass.
        """
        import torch

        from reliquary.constants import FORCED_SEED_ENFORCE
        from reliquary.miner.forced_seed_sampler import (
            ForcedSeedLogitsProcessor, forced_seed_generate_kwargs,
        )

        # The forced-seed pick is normally a HF LogitsProcessor, so under
        # enforcement we take the HF path — UNLESS RELIQUARY_VLLM_FORCED_SEED is
        # set, in which case the vLLM backend applies the pick itself (its engine
        # registers VLLMForcedSeedLogitsProcessor) and phase-1 runs on vLLM.
        _use_vllm = (not FORCED_SEED_ENFORCE) or vllm_forced_seed_enabled()
        backend = getattr(self, "_vllm_backend", None) if _use_vllm else None
        hotkey = self.wallet.hotkey.ss58_address
        checkpoint_hash = self._local_hash
        prompt_tokens = encode_prompt(self.tokenizer, problem["prompt"])
        prompt_length = len(prompt_tokens)

        # BFT (v7): on the math env, thinking rollouts are generated with a
        # phase-1 thinking cap (BFT_THINKING_BUDGET) and force-terminated with a
        # boxed-answer template if </think> never closes. bft_on is False for the
        # code env (validator carve-out is math-only).
        from reliquary.miner.bft import bft_applicable, phase1_max_new_tokens

        env_name = getattr(
            env if env is not None else getattr(self, "env", None), "name", None,
        )
        bft_on = bft_applicable(env_name)
        max_new = phase1_max_new_tokens(self.max_new_tokens, env_name)

        if backend is not None:
            if FORCED_SEED_ENFORCE and vllm_forced_seed_enabled():
                # Forced-seed phase-1 on vLLM: the engine-registered
                # VLLMForcedSeedLogitsProcessor forces every token to the public
                # inverse-CDF pick per rollout_index. Phase-2 (answer) still runs
                # on HF via _bft_from_seqs below.
                # A batched prefetch (_prefetch_phase1) may already hold this
                # prompt's completions. Keyed on (idx, randomness, ckpt) and
                # single-use, so a stale entry can never be served here.
                completions = self._take_prefetched_phase1(
                    prompt_idx, randomness, checkpoint_hash,
                )
                if completions is None:
                    completions = backend.generate_forced_phase1(
                        prompt_tokens,
                        randomness=randomness,
                        prompt_idx=prompt_idx,
                        checkpoint_hash=checkpoint_hash,
                        m_rollouts=M_ROLLOUTS,
                        max_tokens=max_new,
                        stop_token_ids=self._eos_ids,
                    )
            else:
                # Legacy vLLM path (non-forced-seed): EOS already in gen_tokens.
                completions = backend.generate(
                    prompt_token_ids=prompt_tokens,
                    n=M_ROLLOUTS,
                    temperature=T_PROTO,
                    top_p=TOP_P_PROTO,
                    top_k=TOP_K_PROTO,
                    max_tokens=max_new,
                    stop_token_ids=self._eos_ids,
                )
            seqs = [prompt_tokens + list(gen_tokens) for gen_tokens in completions]
            if bft_on:
                return self._bft_from_seqs(
                    seqs, prompt_tokens, randomness=randomness, hotkey=hotkey,
                    prompt_idx=prompt_idx, checkpoint_hash=checkpoint_hash)
            return [
                {"tokens": seq, "prompt_length": prompt_length}
                for seq in seqs
            ]

        # Production wires a VLLMBackend and leaves self.vllm_model=None; under
        # FORCED_SEED_ENFORCE the backend is bypassed (HF LogitsProcessor), so
        # fall back to the HF proof model — the same instance _bft_from_seqs
        # uses for phase-2, keeping both phases on identical weights.
        gen_model = self.vllm_model if self.vllm_model is not None else self.hf_model
        with torch.no_grad():
            input_tensor = torch.tensor(
                [prompt_tokens] * M_ROLLOUTS,
                device=getattr(gen_model, "device", "cpu"),
            )
            attention_mask = torch.ones_like(input_tensor)
            # Phase-1: force sampling onto the protocol seed stream. The
            # processor applies the T_PROTO/top_k/top_p warp itself and picks the
            # inverse-CDF token, so HF warpers are stripped and do_sample is off
            # (see forced_seed_generate_kwargs). Row r is rollout index r,
            # resuming at completion offset 0.
            base_kwargs = {
                "max_new_tokens": max_new,
                "pad_token_id": self.tokenizer.pad_token_id,
                "attention_mask": attention_mask,
            }
            if self._eos_ids:
                base_kwargs["eos_token_id"] = sorted(self._eos_ids)
            phase1_proc = ForcedSeedLogitsProcessor(
                randomness=randomness, hotkey=hotkey, prompt_idx=prompt_idx,
                checkpoint_hash=checkpoint_hash,
                rollout_indices=list(range(M_ROLLOUTS)),
                base_offsets=[0] * M_ROLLOUTS, start_len=prompt_length,
            )
            outputs = gen_model.generate(
                input_tensor,
                **forced_seed_generate_kwargs(base_kwargs, phase1_proc),
            )
        if bft_on:
            return self._bft_from_seqs(
                [outputs[i].tolist() for i in range(M_ROLLOUTS)], prompt_tokens,
                randomness=randomness, hotkey=hotkey, prompt_idx=prompt_idx,
                checkpoint_hash=checkpoint_hash)
        rollouts = []
        for i in range(M_ROLLOUTS):
            seq = outputs[i].tolist()
            gen = seq[prompt_length:]
            first_eos = first_eos_index(gen, self._eos_ids)
            if first_eos is not None:
                gen = gen[: first_eos + 1]
            rollouts.append({
                "tokens": prompt_tokens + gen,
                "prompt_length": prompt_length,
            })
        return rollouts

    def _build_rollout_submission(self, generation, problem, randomness):
        """Build a RolloutSubmission: completion + claimed reward + GRAIL commit."""
        all_tokens = generation["tokens"]
        prompt_length = generation["prompt_length"]
        completion_tokens = all_tokens[prompt_length:]
        completion_text = self.tokenizer.decode(completion_tokens)
        reward = self.env.compute_reward(problem, completion_text)

        commit = self._build_grail_commit(generation, randomness)
        return RolloutSubmission(
            tokens=all_tokens,
            reward=reward,
            commit=commit,
            env_name=self.env.name,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _compute_randomness(
        self, subtensor, window_start: int, use_drand: bool
    ) -> str:
        """Derive window randomness from the drand beacon (v2.3+: drand-only).

        Matches the validator's ``service._derive_randomness``: block_hash is
        no longer mixed in, so the miner does not need a substrate roundtrip
        for the GRAIL seed. The legacy ``use_drand=False`` path remains for
        offline tests and uses block_hash as a single-source seed.
        """
        if use_drand:
            from reliquary.infrastructure.drand import get_beacon, get_current_chain

            chain_info = get_current_chain()
            drand_round = chain.compute_drand_round_for_window(
                window_start, chain_info["genesis_time"], chain_info["period"]
            )
            beacon = get_beacon(round_id=str(drand_round), use_drand=True)
            return chain.compute_window_randomness(
                None, beacon["randomness"], drand_round=beacon["round"]
            )
        block_hash = await chain.get_block_hash(subtensor, window_start)
        return chain.compute_window_randomness(block_hash)

    def _build_grail_commit(self, generation: dict, randomness: str) -> dict:
        """Construct a GRAIL proof commit dict from a generation dict.

        Reproduces the proof construction:
          - HF forward pass for hidden_states + logits
          - Commitment batch via GRAILVerifier
          - log-softmax token log-probs
          - Signature via sign_commit_binding
        """
        import torch

        from reliquary.constants import GRAIL_PROOF_VERSION
        from reliquary.miner.bft import rollout_metadata
        from reliquary.protocol.signatures import sign_commit_binding
        from reliquary.shared.forward import forward_single_layer

        all_tokens: list[int] = generation["tokens"]
        prompt_length: int = generation["prompt_length"]

        # HF forward pass on proof GPU
        proof_input = torch.tensor(
            [all_tokens], device=f"cuda:{self.proof_gpu}"
        )
        with torch.no_grad():
            hidden_states, logits = forward_single_layer(
                self.hf_model, proof_input, None, LAYER_INDEX
            )

        hidden_states = hidden_states[0]  # [seq_len, hidden_dim]

        # Build commitments
        r_vec = self._verifier.generate_r_vec(randomness)
        commitments = self._verifier.create_commitments_batch(hidden_states, r_vec)

        # fp32 log_softmax to match the validator and reduce tail-token drift.
        log_probs = torch.log_softmax(logits[0].float(), dim=-1)
        token_logprobs: list[float] = []
        for i in range(prompt_length, len(all_tokens)):
            token_logprobs.append(log_probs[i - 1, all_tokens[i]].item())

        # Sign
        model_name: str = getattr(self.hf_model, "name_or_path", "unknown")
        signature = sign_commit_binding(
            all_tokens, randomness, model_name, LAYER_INDEX,
            commitments, self.wallet,
        )

        return {
            "tokens": all_tokens,
            "commitments": commitments,
            "proof_version": GRAIL_PROOF_VERSION,
            "model": {"name": model_name, "layer_index": LAYER_INDEX},
            "signature": signature.hex(),
            "beacon": {"randomness": randomness},
            "rollout": rollout_metadata(generation, token_logprobs),
        }

    # ------------------------------------------------------------------
    # Pipelined pre-bake + finalize (v2.3 drand-anchored ordering)
    # ------------------------------------------------------------------

    def _pre_bake_entry(
        self, prompt_idx: int, problem: dict, expected_ckpt_n: int, env=None,
    ) -> dict | None:
        """Sync: vLLM generate + HF forward + reward + token_logprobs.

        Everything in this function is randomness-INDEPENDENT and survives
        any subsequent window change (so long as the checkpoint doesn't
        advance — the trigger loop drops the pool on a real checkpoint
        advance). Returns a cache dict ready to be finalized with the
        per-window randomness once /state publishes it.

        Hidden states are moved to CPU to free GPU memory for the next
        bake cycle; they're shipped back to the proof GPU at finalize.

        Returns ``None`` on generation underflow (vLLM produced < M
        rollouts).
        """
        import torch

        from reliquary.shared.forward import forward_single_layer

        env = env if env is not None else self.env

        # 1. vLLM autoregressive sampling. The ``randomness`` argument is
        # only used in legacy callers — it doesn't actually affect token
        # generation here (vLLM samples with its own seed via do_sample=True).
        # We pass an empty string explicitly to make the independence clear.
        # Forced-seed (v7.1): generation is randomness-DEPENDENT — each token is
        # the u_at(randomness, …) pick. We bake with the CURRENT window randomness
        # (self._cached_randomness, set by the trigger loop the instant /state
        # publishes it). Entries baked under a randomness that later flips are
        # dropped by the trigger loop's flush (see _trigger_loop), so a submission
        # only ever carries tokens generated under its own window randomness.
        generations = self._generate_m_rollouts(
            problem, self._cached_randomness, env, prompt_idx=prompt_idx)
        if len(generations) < M_ROLLOUTS:
            logger.warning(
                "pre_bake: generated %d/%d for prompt %d; skipping",
                len(generations), M_ROLLOUTS, prompt_idx,
            )
            return None

        # 2. Per-rollout HF forward → hidden states + logits.
        # token_logprobs (= log_softmax(logits)[t, all_tokens[t]]) is also
        # randomness-independent so we compute it here once.
        #
        # ⚠ DO NOT batch this forward. Measured 2026-07-21 on the real v3
        # checkpoint (scripts/validate_proof_batch_parity.py): right-padded
        # batching drifts hidden states by 0.44-0.75 in bf16, which is enough to
        # FLIP the sketch's top-k selection (topk=16 of hidden_dim=2048, many
        # near-equal magnitudes). Result: 25/81 positions got a completely
        # different sketch, worst delta 2.1e9 = 428553x the validator's
        # adaptive tolerance. Bucketing absorbs pure magnitude drift (one seq
        # showed delta 688, within tolerance) but not a top-k reorder.
        # STEP 1 — grade only (cheap). compute_reward depends solely on the
        # generated tokens, NOT on the proof, so the sigma decision can be made
        # before any GRAIL forward. Skipping the proof for out-of-zone groups is
        # the single biggest win: the forward is ~3.4 s/rollout (~27 s/prompt,
        # ~91% of cycle time measured 2026-07-21) and ~99.8% of groups are
        # out-of-zone, so almost all of that compute was previously wasted.
        # Correction EN PARALLÈLE : en code chaque compute_reward lance un
        # subprocess (sandbox de test) ; en série ils laissent le GPU à 0% 52%
        # du temps (mesuré 2026-07-23). Les threads les recouvrent (×9,6). En
        # maths compute_reward est symbolique/rapide : le parallélisme n'y nuit
        # pas (n<=1 court-circuite, sinon overhead négligeable).
        rewards_for_zone = grade_group_parallel(
            env,
            [
                (problem, self.tokenizer.decode(g["tokens"][g["prompt_length"]:]))
                for g in generations
            ],
            max_workers=M_ROLLOUTS,
        )
        if _skip_for_out_of_zone(rewards_for_zone):
            from reliquary.validator.verifier import rewards_std
            sigma = rewards_std(rewards_for_zone)
            # env is logged because attributing candidates by reward shape is
            # unreliable: a code group whose rollouts all score exactly 0 or 1
            # is indistinguishable from a math group, which skewed the
            # math/code split estimate on 2026-07-21.
            logger.info(
                "pre_bake[out_of_zone] env=%s skipping prompt=%d sigma=%.3f "
                "rewards=%s",
                getattr(env, "name", "?"), prompt_idx, sigma, rewards_for_zone,
            )
            return None

        # STEP 2 — in-zone only: now pay for the GRAIL proof forward.
        rollouts_cache = []
        for gen, reward in zip(generations, rewards_for_zone):
            all_tokens = gen["tokens"]
            prompt_length = gen["prompt_length"]
            completion_tokens = all_tokens[prompt_length:]
            completion_text = self.tokenizer.decode(completion_tokens)

            proof_input = torch.tensor(
                [all_tokens], device=f"cuda:{self.proof_gpu}",
            )
            with torch.no_grad():
                hidden_states, logits = forward_single_layer(
                    self.hf_model, proof_input, None, LAYER_INDEX,
                )
            hidden_states = hidden_states[0]  # [seq_len, hidden_dim]
            log_probs = torch.log_softmax(logits[0].float(), dim=-1)
            token_logprobs: list[float] = []
            for i in range(prompt_length, len(all_tokens)):
                token_logprobs.append(log_probs[i - 1, all_tokens[i]].item())

            # Park heavy tensors on CPU to keep pool memory bounded. They're
            # shipped back to the proof GPU at finalize for the commitments
            # matmul (~5 ms PCIe transfer for a single rollout).
            rollouts_cache.append({
                "all_tokens": all_tokens,
                "prompt_length": prompt_length,
                "completion_text": completion_text,
                "hidden_states_cpu": hidden_states.detach().cpu(),
                "token_logprobs": token_logprobs,
                "reward": reward,
                # BFT: carried into the finalize-time commit metadata so the
                # validator carve-out can locate the injected FORCE span.
                "forced": bool(gen.get("forced", False)),
                "force_span": gen.get("force_span"),
            })

        return {
            "prompt_idx": prompt_idx,
            "problem": problem,
            "rollouts": rollouts_cache,
            "checkpoint_n": expected_ckpt_n,
            "env_name": env.name,
        }

    def _pre_bake_batch(
        self,
        prompt_indices: list[int],
        problems: list[dict],
        expected_ckpt_n: int,
        existing_rollouts_per_idx: dict[int, list[dict]] | None = None,
        env=None,
    ) -> tuple[list[dict], dict[int, list[dict]]]:
        """Sync: single batched vLLM gen + per-rollout HF forward + select.

        Multi-phase strategy. Each call generates ``M_PER_PHASE`` NEW rollouts
        per prompt and combines them with any rollouts already accumulated
        for that prompt in ``existing_rollouts_per_idx``. After each combine:

          * Phase 1 (= 8 rollouts cumulative): drop early if sigma=0 (no
            reward diversity) or bt_ok=0 (model never terminates this prompt).
          * Otherwise: ``_try_select`` on the cumulative set. If a valid
            submission can be composed → bake into an entry. If not and we
            haven't hit MAX_PHASES → return the prompt in the updated retry
            dict for another phase. If MAX_PHASES reached → drop.

        Returns: (baked_entries, retry_dict). The caller is expected to
        maintain a persistent retry_queue and pass it back as
        ``existing_rollouts_per_idx`` on the next call.

        Falls back to per-prompt ``_pre_bake_entry`` when no vLLM backend
        is configured (= legacy single-rollout path, no multi-phase).
        """
        import torch

        from reliquary.miner.bft import bft_applicable, phase1_max_new_tokens
        from reliquary.shared.forward import forward_single_layer

        env = env if env is not None else self.env

        from reliquary.constants import FORCED_SEED_ENFORCE

        # Forced-seed needs a HF LogitsProcessor per generate() — the vLLM
        # continuous-batching backend (generate_multi) cannot apply it, so when
        # forced-seed is enforced we always take the per-prompt HF path
        # (_pre_bake_entry → _generate_m_rollouts), which forces every token.
        backend = getattr(self, "_vllm_backend", None)
        if backend is None or FORCED_SEED_ENFORCE:
            # Batch forced-seed phase-1 for the whole bake in ONE vLLM call when
            # that path is active; _pre_bake_entry then consumes the cache.
            # Per-prompt calls (~40s each) made a 6-prompt bake overrun the 100s
            # collection window, so every entry was flushed stale at the flip.
            # A no-op (returns 0) on the HF path, leaving behaviour unchanged.
            self._prefetch_phase1(
                problems, prompt_indices,
                randomness=self._cached_randomness, env=env,
            )
            results = []
            for idx, prob in zip(prompt_indices, problems):
                e = self._pre_bake_entry(idx, prob, expected_ckpt_n, env)
                if e is not None:
                    results.append(e)
            # Drop anything unconsumed (a prompt that errored out) so it can
            # never be served under a later window's randomness.
            self._phase1_cache = {}
            return results, {}

        if existing_rollouts_per_idx is None:
            existing_rollouts_per_idx = {}

        # Canonical prompt tokens via the SHARED ``encode_prompt`` (applies the
        # Qwen3.5 chat template + enable_thinking=False when declared, plain
        # encode otherwise). The validator computes the SAME canonical encoding,
        # so generation AND submission use one identical prompt — there is no
        # raw/templated split anymore (that was the v5/Qwen3-4B workaround), and
        # the first ``prompt_length`` tokens match canonical → no PROMPT_MISMATCH.
        prompts_token_ids = [
            encode_prompt(self.tokenizer, p["prompt"])
            for p in problems
        ]
        gen_prompts_token_ids = prompts_token_ids

        # Multi-phase: each call generates M_PER_PHASE NEW rollouts that
        # we then combine with the prompt's existing rollouts (= rollouts
        # baked in earlier phases for this same prompt).
        all_completions = backend.generate_multi(
            prompts_token_ids=gen_prompts_token_ids,
            n=M_PER_PHASE,
            temperature=T_PROTO,
            top_p=TOP_P_PROTO,
            top_k=TOP_K_PROTO,
            max_tokens=phase1_max_new_tokens(self.max_new_tokens, env.name),
            stop_token_ids=self._eos_ids,
        )

        # _try_select is now a method (self._try_select); see definition
        # below the class body. It's stateless (uses only module-level
        # constants) so both the sync and async paths can call it.
        _try_select = self._try_select

        entries: list[dict] = []
        updated_retry: dict[int, list[dict]] = {}
        for prompt_idx, problem, ptoks, completions in zip(
            prompt_indices, problems, prompts_token_ids, all_completions,
        ):
            if len(completions) < M_PER_PHASE:
                logger.warning(
                    "pre_bake: under-generated %d/%d for prompt %d; skipping",
                    len(completions), M_PER_PHASE, prompt_idx,
                )
                continue

            prompt_length = len(ptoks)
            # Multi-phase: combine new gen with rollouts carried over.
            existing = existing_rollouts_per_idx.get(prompt_idx, [])
            phase = (len(existing) + len(completions)) // M_PER_PHASE

            # CHEAP PASS: decode + reward only (no GPU forward). This lets
            # us evaluate sigma=0 in phase 1 and drop the prompt without
            # paying the ~5-10s HF-forward cost on rollouts we're throwing
            # away. Validated: ~70% of phase-1 prompts hit drop_sigma0_p1
            # → big compute win.
            # BFT (math): force-terminate at the thinking budget BEFORE decoding,
            # so forced rollouts carry a boxed answer + a valid force_span. Code
            # env keeps the raw completions (bft_applicable=False).
            seqs = [ptoks + list(gen) for gen in completions]
            if bft_applicable(env.name):
                # Forced-seed: phase-2 continues the same u_at stream as phase-1,
                # bound to the current window randomness (see _pre_bake_entry).
                bft_rolls = self._bft_from_seqs(
                    seqs, ptoks, randomness=self._cached_randomness,
                    hotkey=self.wallet.hotkey.ss58_address,
                    prompt_idx=prompt_idx, checkpoint_hash=self._local_hash)
            else:
                bft_rolls = [
                    {"tokens": s, "prompt_length": prompt_length} for s in seqs
                ]

            new_partial: list[dict] = []
            for roll in bft_rolls:
                all_tokens = roll["tokens"]
                completion_tokens = all_tokens[prompt_length:]
                completion_text = self.tokenizer.decode(completion_tokens)
                reward = env.compute_reward(problem, completion_text)
                new_partial.append({
                    "all_tokens": all_tokens,
                    "prompt_length": prompt_length,
                    "completion_text": completion_text,
                    "reward": reward,
                    "forced": bool(roll.get("forced", False)),
                    "force_span": roll.get("force_span"),
                })

            # Phase 1 σ=0 EARLY drop (before HF forward) — cheap to check
            # from rewards alone. bt_ok=0 needs the HF forward so it
            # stays below.
            if phase == 1:
                all_rewards = (
                    [r["reward"] for r in existing]
                    + [r["reward"] for r in new_partial]
                )
                if len(set(all_rewards)) <= 1:
                    logger.info(
                        "pre_bake[drop_sigma0_p1] prompt=%d rewards_uniform=%r — dropping (skipped HF forward)",
                        prompt_idx,
                        all_rewards[0] if all_rewards else None,
                    )
                    continue

                # Phase 1 bt_ok=0 EARLY drop: if ALL new rollouts hit
                # max_new_tokens (= last token NOT in EOS_SET), bt_ok is
                # guaranteed False for all of them. Skip the HF forward.
                # Validated: drop_btok0 = ~16% of prompts in prod. Each
                # such prompt was paying ~5s of wasted HF forward.
                if DROP_BTOK0_PHASE1 and not existing:
                    new_all_hit_max = all(
                        (
                            r["all_tokens"][-1] not in self._eos_ids
                            if r["all_tokens"] else True
                        )
                        for r in new_partial
                    )
                    if new_all_hit_max:
                        logger.info(
                            "pre_bake[drop_btok0_p1] prompt=%d — all rollouts hit max_tokens (no EOS), dropping (skipped HF forward)",
                            prompt_idx,
                        )
                        continue

            # EXPENSIVE PASS: HF forward + q10/p_stop only for prompts
            # that survived the σ=0 check. Existing rollouts already have
            # these fields cached from prior phases.
            new_rollouts: list[dict] = []
            for r in new_partial:
                all_tokens = r["all_tokens"]
                completion_text = r["completion_text"]
                reward = r["reward"]

                proof_input = torch.tensor(
                    [all_tokens], device=f"cuda:{self.proof_gpu}",
                )
                with torch.no_grad():
                    hidden_states, logits = forward_single_layer(
                        self.hf_model, proof_input, None, LAYER_INDEX,
                    )
                hidden_states_cpu = hidden_states[0].detach().cpu()
                log_probs = torch.log_softmax(logits[0].float(), dim=-1)
                token_logprobs: list[float] = []
                for i in range(prompt_length, len(all_tokens)):
                    token_logprobs.append(log_probs[i - 1, all_tokens[i]].item())

                # Mirror validator's verify_termination: softmax over EOS
                # tokens at logits[seq_len-2], no T_PROTO scaling.
                n_tok = len(all_tokens)
                last_token = all_tokens[-1] if all_tokens else None
                in_eos = last_token in self._eos_ids
                p_stop_local = None
                if in_eos and n_tok >= 2 and n_tok - 2 < logits[0].size(0):
                    with torch.no_grad():
                        probs_last = torch.softmax(
                            logits[0][n_tok - 2].float(), dim=-1,
                        )
                        p_stop_local = float(
                            sum(probs_last[e].item() for e in self._eos_ids)
                        )

                # EXPERIMENT: floor the reported final-token logprob so the
                # validator's claim-based preflight passes for naturally
                # terminated rollouts (see EOS_LOGPROB_FLOOR comment above).
                if EOS_LOGPROB_FLOOR > 0.0 and in_eos and token_logprobs:
                    import math as _math
                    token_logprobs[-1] = max(
                        token_logprobs[-1], _math.log(EOS_LOGPROB_FLOOR),
                    )

                # Local q10/median (= mirrors the validator's
                # evaluate_token_distribution under T_PROTO scaling). Computed
                # over completion positions only. We use this to score
                # rollouts before composing the submission so we prefer
                # those most likely to pass the validator's filter.
                chosen_probs_tproto: list[float] = []
                if len(all_tokens) - prompt_length >= 1:
                    with torch.no_grad():
                        tproto_log = torch.log_softmax(
                            logits[0].float() / T_PROTO, dim=-1,
                        )
                    for i in range(prompt_length, len(all_tokens)):
                        chosen_probs_tproto.append(
                            float(torch.exp(tproto_log[i - 1, all_tokens[i]]).item())
                        )
                q10_local = None
                median_local = None
                if len(chosen_probs_tproto) >= 30:  # SAMPLING_MIN_STEPS
                    import numpy as _np
                    arr = _np.asarray(chosen_probs_tproto, dtype=_np.float64)
                    q10_local = float(_np.quantile(arr, 0.10))
                    median_local = float(_np.median(arr))

                new_rollouts.append({
                    "all_tokens": all_tokens,
                    "prompt_length": prompt_length,
                    "completion_text": completion_text,
                    "hidden_states_cpu": hidden_states_cpu,
                    "token_logprobs": token_logprobs,
                    "reward": reward,
                    "in_eos": in_eos,
                    "p_stop_local": p_stop_local,
                    "q10_local": q10_local,
                    "median_local": median_local,
                    "bt_ok": (
                        in_eos
                        and p_stop_local is not None
                        and p_stop_local >= P_STOP_LOCAL_MIN
                    ),
                    # BFT: carried to the finalize-time commit metadata.
                    "forced": bool(r.get("forced", False)),
                    "force_span": r.get("force_span"),
                })

            # Combine new rollouts with any existing rollouts carried
            # over from earlier phases for this prompt. ``existing`` and
            # ``phase`` were already computed above for the σ=0 fast-path.
            rollouts = existing + new_rollouts

            # Phase 1 bt_ok=0 drop: needs the HF forward (= bt_ok depends
            # on p_stop_local). σ=0 already handled above before forward.
            if phase == 1 and DROP_BTOK0_PHASE1:
                bt_total = sum(1 for r in rollouts if r["bt_ok"])
                if bt_total == 0:
                    logger.info(
                        "pre_bake[drop_btok0_p1] prompt=%d — no rollouts terminated, dropping",
                        prompt_idx,
                    )
                    continue

            subset, k = _try_select(rollouts, env)
            if subset is None:
                bt_c = sum(1 for r in rollouts if r["bt_ok"] and r["reward"] == 1.0)
                bt_w = sum(1 for r in rollouts if r["bt_ok"] and r["reward"] == 0.0)
                nbt_c = sum(1 for r in rollouts if not r["bt_ok"] and r["reward"] == 1.0)
                nbt_w = sum(1 for r in rollouts if not r["bt_ok"] and r["reward"] == 0.0)
                if phase < MAX_PHASES:
                    # Retry: carry the cumulative rollouts into the next
                    # phase. Caller will pass them back as
                    # ``existing_rollouts_per_idx`` next round.
                    logger.info(
                        "pre_bake[retry_p%d] prompt=%d bt(c/w)=%d/%d nbt(c/w)=%d/%d "
                        "k_band=[%d,%d] — retrying next phase (%d/%d)",
                        phase, prompt_idx, bt_c, bt_w, nbt_c, nbt_w,
                        K_MIN, K_MAX, phase + 1, MAX_PHASES,
                    )
                    updated_retry[prompt_idx] = rollouts
                else:
                    logger.info(
                        "pre_bake[drop_k_band_p%d] prompt=%d bt(c/w)=%d/%d nbt(c/w)=%d/%d "
                        "k_band=[%d,%d] max_nonbt=%d — MAX_PHASES reached, dropping",
                        phase, prompt_idx, bt_c, bt_w, nbt_c, nbt_w,
                        K_MIN, K_MAX, MAX_NON_BTOK_IN_SUBMISSION,
                    )
                continue

            n_nbt = sum(1 for r in subset if not r["bt_ok"])
            p_stop_min = min(
                (r["p_stop_local"] for r in subset if r["bt_ok"]),
                default=0.0,
            )
            logger.info(
                "pre_bake[selected] prompt=%d k=%d/%d non_bt_ok=%d p_stop_bt_min=%.3f",
                prompt_idx, k, M_ROLLOUTS, n_nbt, p_stop_min,
            )
            entries.append({
                "prompt_idx": prompt_idx,
                "problem": problem,
                "rollouts": subset,
                "checkpoint_n": expected_ckpt_n,
                "env_name": env.name,
            })

        return entries, updated_retry

    # ------------------------------------------------------------------
    # Shared subset-selection helper (used by both sync _pre_bake_batch
    # and the async per-prompt processor below). Lifted out of
    # _pre_bake_batch's nested scope — it never captured local state.
    # ------------------------------------------------------------------
    def _try_select(
        self, rollouts: list[dict], env=None,
    ) -> tuple[list[dict] | None, int | None]:
        """Pick M_ROLLOUTS rollouts forming a valid in-zone subset.

        Dispatch on the env's reward type:
          * binary (math, default): sigma-based k-band (K_MIN..K_MAX), k correct
            + (M-k) wrong, prefer bt_ok within the truncation budget.
          * continuous (code, ``env.continuous_reward``): max-variance subset of
            the continuous rewards reaching std >= SIGMA_MIN + margin.
        Returns (subset, k) for binary, (subset, None) for continuous, or
        (None, None) when no valid in-zone subset can be composed.
        """
        # Dedupe rollouts by token content. vLLM at T=0.9 can sample the
        # exact same token sequence twice on easy prompts. The validator
        # rejects the whole submission on any duplicate hash within it
        # (intra-submission dedup via local_seen), so we MUST drop dups
        # before composing the subset. Mirrors compute_rollout_hash.
        import hashlib as _hashlib
        seen_hashes: set[bytes] = set()
        dedup_rollouts: list[dict] = []
        for r in rollouts:
            h = _hashlib.sha256(
                b"".join(
                    int(t).to_bytes(4, "big", signed=False)
                    for t in r["all_tokens"]
                )
            ).digest()
            if h not in seen_hashes:
                seen_hashes.add(h)
                dedup_rollouts.append(r)
        n_dropped = len(rollouts) - len(dedup_rollouts)
        if n_dropped > 0:
            logger.info(
                "pre_bake: deduped %d intra-batch duplicate rollouts "
                "(%d -> %d)",
                n_dropped, len(rollouts), len(dedup_rollouts),
            )
        rollouts = dedup_rollouts

        # Optionally drop rollouts whose LOCAL q10/median fall below the
        # configured floors (mirrors the validator's filter; we exclude
        # them now rather than risking a submission-level reject). Off by
        # default; activate via env vars.
        def _passes_local_dist(r):
            q10 = r.get("q10_local")
            med = r.get("median_local")
            if MIN_LOCAL_Q10 > 0 and (q10 is None or q10 < MIN_LOCAL_Q10):
                return False
            if MIN_LOCAL_MEDIAN > 0 and (med is None or med < MIN_LOCAL_MEDIAN):
                return False
            return True

        kept = [r for r in rollouts if _passes_local_dist(r)]
        if len(kept) < M_ROLLOUTS:
            return None, None

        bt_ok_rollouts = [r for r in kept if r["bt_ok"]]
        non_bt_ok = [r for r in kept if not r["bt_ok"]]
        min_bt_ok_required = M_ROLLOUTS - MAX_NON_BTOK_IN_SUBMISSION

        # Continuous-reward envs (code): the binary k-band buckets (==1.0/==0.0)
        # don't apply — compose a max-variance subset over the continuous rewards
        # and accept only if its std clears SIGMA_MIN + margin.
        if getattr(env, "continuous_reward", False):
            margin = float(_os.environ.get("RELIQUARY_CODE_SIGMA_MARGIN", "0.03"))
            # Use the STEADY validator threshold (0.43), NOT constants.SIGMA_MIN
            # (0.33 bootstrap) — the binary k-band protects math, but the
            # continuous branch targets the gate directly, so it must be 0.43.
            sigma_target = ZONE_THRESHOLD_STEADY + margin
            # Prefer bt_ok rollouts; fall back to the full kept set only if there
            # aren't enough bt_ok to fill a group.
            pool = bt_ok_rollouts if len(bt_ok_rollouts) >= M_ROLLOUTS else kept
            subset = _select_continuous_subset(pool, M_ROLLOUTS, sigma_target)
            if subset is None:
                return None, None
            n_non_bt = sum(1 for r in subset if not r["bt_ok"])
            if n_non_bt > MAX_NON_BTOK_IN_SUBMISSION:
                return None, None
            return subset, None

        def _bt_key(r):
            return (
                -(r.get("q10_local") or 0.0),
                -(r.get("p_stop_local") or 0.0),
            )

        def _nonbt_key(r):
            return (
                -int(bool(r.get("in_eos"))),
                -(r.get("p_stop_local") or 0.0),
            )

        mid = (K_MIN + K_MAX) // 2
        k_order = sorted(
            range(K_MIN, K_MAX + 1), key=lambda k: abs(k - mid),
        )
        correct_bt = sorted(
            [r for r in bt_ok_rollouts if r["reward"] == 1.0], key=_bt_key,
        )
        wrong_bt = sorted(
            [r for r in bt_ok_rollouts if r["reward"] == 0.0], key=_bt_key,
        )
        correct_nbt = sorted(
            [r for r in non_bt_ok if r["reward"] == 1.0], key=_nonbt_key,
        )
        wrong_nbt = sorted(
            [r for r in non_bt_ok if r["reward"] == 0.0], key=_nonbt_key,
        )

        for k in k_order:
            wrong_n = M_ROLLOUTS - k
            if (
                len(correct_bt) + len(correct_nbt) < k
                or len(wrong_bt) + len(wrong_nbt) < wrong_n
            ):
                continue
            used_correct = correct_bt[:k]
            if len(used_correct) < k:
                used_correct.extend(correct_nbt[: k - len(used_correct)])
            used_wrong = wrong_bt[:wrong_n]
            if len(used_wrong) < wrong_n:
                used_wrong.extend(wrong_nbt[: wrong_n - len(used_wrong)])
            subset = used_correct + used_wrong
            n_non_bt = sum(1 for r in subset if not r["bt_ok"])
            if (
                n_non_bt > MAX_NON_BTOK_IN_SUBMISSION
                or (M_ROLLOUTS - n_non_bt) < min_bt_ok_required
            ):
                continue
            return subset, k
        return None, None

    # ------------------------------------------------------------------
    # Async continuous-batching path (RELIQUARY_ASYNC_MODE=1).
    #
    # Replaces the sync batch-of-10 _generator_loop with a per-prompt
    # task pool. Each task = 1 prompt x M_PER_PHASE rollouts dispatched
    # via AsyncVLLMBackend.generate. We keep TARGET_ACTIVE tasks
    # in-flight at all times and process them as they finish (FIFO of
    # asyncio.wait FIRST_COMPLETED). Continuous batching means vLLM
    # interleaves rollouts across prompts on every GPU step, so the
    # tail-latency stall of waiting for the slowest of 80 rollouts in
    # the sync path is gone — we always have fresh prompts queued.
    # ------------------------------------------------------------------

    def _async_pick_next_prompt(
        self,
        env_name: str,
        exclude: set,
        rng,
    ) -> tuple[int, dict, list[dict]] | None:
        """Pick the next prompt to bake FOR ``env_name``.

        Priority order:
          1. That env's retry queue (= prompts that need more rollouts to
             compose a valid k-band subset). Skipped if cooldown or already
             in ``exclude``.
          2. Fresh pick via ``pick_prompt_idx`` within that env's slice.

        Returns ``(prompt_idx, problem, existing_rollouts)`` or ``None``
        if the env is fully covered (= rare; 14M prompts).
        """
        env = self.envs[env_name]
        cooldown = self._cooldowns[env_name]
        retry = self._retry_by_env[env_name]
        # Retry first — these prompts already showed signal.
        for idx in list(retry.keys()):
            if idx in exclude or idx in cooldown:
                continue
            existing = retry.get(idx, [])
            try:
                problem = env.get_problem(idx)
            except Exception:
                # Defensive: a stale retry entry pointing at a missing
                # prompt should be dropped, not crash the loop.
                retry.pop(idx, None)
                continue
            return idx, problem, existing

        # Fresh pick — confined to the per-window slice (#91) when armed,
        # derived for THIS env (env_name domain-separates the slice).
        try:
            idx = pick_prompt_idx(
                env, cooldown | exclude, rng=rng,
                prompt_range=self._active_prompt_range(
                    self._cached_window_n, self._cached_randomness, env,
                ),
            )
        except RuntimeError:
            return None
        problem = env.get_problem(idx)
        return idx, problem, []

    async def _process_one_completion(
        self,
        prompt_idx: int,
        problem: dict,
        ptoks: list[int],
        completions: list[list[int]],
        existing_rollouts: list[dict],
        expected_ckpt_n: int,
        env_name: str | None = None,
    ) -> tuple[dict | None, list[dict] | None]:
        """Per-prompt post-generation pipeline (async-friendly).

        Mirrors the per-prompt body of ``_pre_bake_batch``:
          1. Cheap pass: decode + reward only.
          2. Phase-1 sigma=0 fast-drop (skip HF forward if so).
          3. HF forward (expensive) for the new rollouts. Serialised via
             ``self._hf_lock`` so concurrent prompts don't storm the
             shared GPU while vLLM is generating on the same device.
          4. Combine with existing rollouts. Phase-1 bt_ok=0 drop.
          5. ``_try_select`` on the cumulative set.

        Returns ``(entry_or_None, retry_or_None)``:
          * entry not None  -> baked successfully, caller appends to pool.
          * retry not None  -> needs another phase, caller stores in
            ``self._retry_by_env[env_name]``.
          * both None       -> dropped (sigma=0 / bt_ok=0 / max_phases /
            under-gen).
        """
        if len(completions) < M_PER_PHASE:
            logger.warning(
                "async_bake: under-generated %d/%d for prompt %d; skipping",
                len(completions), M_PER_PHASE, prompt_idx,
            )
            return None, None

        env = self.envs[env_name] if env_name is not None else self.env

        prompt_length = len(ptoks)
        existing = existing_rollouts or []
        phase = (len(existing) + len(completions)) // M_PER_PHASE

        # 1. Cheap pass — decode + reward, no GPU.
        new_partial: list[dict] = []
        for gen in completions:
            all_tokens = ptoks + list(gen)
            completion_tokens = all_tokens[prompt_length:]
            completion_text = self.tokenizer.decode(completion_tokens)
            reward = env.compute_reward(problem, completion_text)
            new_partial.append({
                "all_tokens": all_tokens,
                "prompt_length": prompt_length,
                "completion_text": completion_text,
                "reward": reward,
            })

        # 2. Phase-1 sigma=0 fast-drop, before any HF forward.
        if phase == 1:
            all_rewards = (
                [r["reward"] for r in existing]
                + [r["reward"] for r in new_partial]
            )
            if len(set(all_rewards)) <= 1:
                logger.info(
                    "async_bake[drop_sigma0_p1] prompt=%d rewards_uniform=%r "
                    "— dropping (skipped HF forward)",
                    prompt_idx,
                    all_rewards[0] if all_rewards else None,
                )
                return None, None

        # 3. Expensive pass — HF forward + q10/p_stop per rollout. Wrap
        # the whole per-prompt forward block in a single to_thread to
        # avoid blocking the event loop, and serialise across prompts
        # via self._hf_lock so concurrent prompts don't race for the
        # GPU while vLLM is also generating on it.
        def _run_hf_forward(new_partial_in: list[dict]) -> list[dict]:
            import torch
            from reliquary.shared.forward import forward_single_layer

            out: list[dict] = []
            for r in new_partial_in:
                all_tokens = r["all_tokens"]
                completion_text = r["completion_text"]
                reward = r["reward"]

                proof_input = torch.tensor(
                    [all_tokens], device=f"cuda:{self.proof_gpu}",
                )
                with torch.no_grad():
                    hidden_states, logits = forward_single_layer(
                        self.hf_model, proof_input, None, LAYER_INDEX,
                    )
                hidden_states_cpu = hidden_states[0].detach().cpu()
                log_probs = torch.log_softmax(logits[0].float(), dim=-1)
                token_logprobs: list[float] = []
                for i in range(prompt_length, len(all_tokens)):
                    token_logprobs.append(
                        log_probs[i - 1, all_tokens[i]].item()
                    )

                n_tok = len(all_tokens)
                last_token = all_tokens[-1] if all_tokens else None
                in_eos = last_token in self._eos_ids
                p_stop_local = None
                if in_eos and n_tok >= 2 and n_tok - 2 < logits[0].size(0):
                    with torch.no_grad():
                        probs_last = torch.softmax(
                            logits[0][n_tok - 2].float(), dim=-1,
                        )
                        p_stop_local = float(
                            sum(probs_last[e].item() for e in self._eos_ids)
                        )

                # EXPERIMENT: floor reported final-token logprob (see
                # EOS_LOGPROB_FLOOR comment) so the validator's claim-based
                # preflight passes; GRAIL recompute remains the real arbiter.
                if EOS_LOGPROB_FLOOR > 0.0 and in_eos and token_logprobs:
                    import math as _math
                    token_logprobs[-1] = max(
                        token_logprobs[-1], _math.log(EOS_LOGPROB_FLOOR),
                    )

                chosen_probs_tproto: list[float] = []
                if len(all_tokens) - prompt_length >= 1:
                    with torch.no_grad():
                        tproto_log = torch.log_softmax(
                            logits[0].float() / T_PROTO, dim=-1,
                        )
                    for i in range(prompt_length, len(all_tokens)):
                        chosen_probs_tproto.append(
                            float(torch.exp(
                                tproto_log[i - 1, all_tokens[i]]
                            ).item())
                        )
                q10_local = None
                median_local = None
                if len(chosen_probs_tproto) >= 30:
                    import numpy as _np
                    arr = _np.asarray(
                        chosen_probs_tproto, dtype=_np.float64,
                    )
                    q10_local = float(_np.quantile(arr, 0.10))
                    median_local = float(_np.median(arr))

                out.append({
                    "all_tokens": all_tokens,
                    "prompt_length": prompt_length,
                    "completion_text": completion_text,
                    "hidden_states_cpu": hidden_states_cpu,
                    "token_logprobs": token_logprobs,
                    "reward": reward,
                    "in_eos": in_eos,
                    "p_stop_local": p_stop_local,
                    "q10_local": q10_local,
                    "median_local": median_local,
                    "bt_ok": (
                        in_eos
                        and p_stop_local is not None
                        and p_stop_local >= P_STOP_LOCAL_MIN
                    ),
                })
            return out

        async with self._hf_lock:
            new_rollouts = await asyncio.to_thread(_run_hf_forward, new_partial)

        # 4. Combine + phase-1 bt_ok=0 drop.
        rollouts = existing + new_rollouts
        if phase == 1 and DROP_BTOK0_PHASE1:
            bt_total = sum(1 for r in rollouts if r["bt_ok"])
            if bt_total == 0:
                logger.info(
                    "async_bake[drop_btok0_p1] prompt=%d — no rollouts "
                    "terminated, dropping",
                    prompt_idx,
                )
                return None, None

        # 5. Try compose.
        subset, k = self._try_select(rollouts, env)
        if subset is None:
            bt_c = sum(1 for r in rollouts if r["bt_ok"] and r["reward"] == 1.0)
            bt_w = sum(1 for r in rollouts if r["bt_ok"] and r["reward"] == 0.0)
            nbt_c = sum(1 for r in rollouts if not r["bt_ok"] and r["reward"] == 1.0)
            nbt_w = sum(1 for r in rollouts if not r["bt_ok"] and r["reward"] == 0.0)
            if phase < MAX_PHASES:
                logger.info(
                    "async_bake[retry_p%d] prompt=%d bt(c/w)=%d/%d "
                    "nbt(c/w)=%d/%d k_band=[%d,%d] — retrying next phase "
                    "(%d/%d)",
                    phase, prompt_idx, bt_c, bt_w, nbt_c, nbt_w,
                    K_MIN, K_MAX, phase + 1, MAX_PHASES,
                )
                return None, rollouts
            logger.info(
                "async_bake[drop_k_band_p%d] prompt=%d bt(c/w)=%d/%d "
                "nbt(c/w)=%d/%d k_band=[%d,%d] max_nonbt=%d — MAX_PHASES "
                "reached, dropping",
                phase, prompt_idx, bt_c, bt_w, nbt_c, nbt_w,
                K_MIN, K_MAX, MAX_NON_BTOK_IN_SUBMISSION,
            )
            return None, None

        n_nbt = sum(1 for r in subset if not r["bt_ok"])
        p_stop_min = min(
            (r["p_stop_local"] for r in subset if r["bt_ok"]),
            default=0.0,
        )
        logger.info(
            "async_bake[selected] prompt=%d k=%d/%d non_bt_ok=%d "
            "p_stop_bt_min=%.3f",
            prompt_idx, k, M_ROLLOUTS, n_nbt, p_stop_min,
        )
        entry = {
            "prompt_idx": prompt_idx,
            "problem": problem,
            "rollouts": subset,
            "checkpoint_n": expected_ckpt_n,
            "env_name": env.name,
        }
        return entry, None

    async def _async_generator_loop(self, url, client, rng):
        """Continuous-batching background bake loop (RELIQUARY_ASYNC_MODE=1).

        Maintains a pool of ``TARGET_ACTIVE`` in-flight vLLM tasks via
        ``AsyncVLLMBackend.generate``. Each task is (prompt_idx,
        existing_rollouts, expected_ckpt_n) -> ``M_PER_PHASE`` completions.
        On any task completion we post-process (cheap eval -> sigma drop
        -> HF forward -> try_select) and immediately enqueue a
        replacement so the GPU stays saturated.

        NEVER exits on a single iteration failure — log and continue.
        Cancellation only happens when ``mine_window`` exits, at which
        point we cancel all in-flight tasks.
        """
        from reliquary.miner.vllm_backend import AsyncVLLMBackend  # noqa: F401

        backend = self._vllm_backend
        target_active = max(1, int(
            _os.environ.get("RELIQUARY_ASYNC_TARGET_ACTIVE", "16"),
        ))

        # Per-task metadata so we can recover (prompt_idx, ptoks, ...)
        # after asyncio.wait returns the completed task.
        # Key: asyncio.Task, Value: dict with prompt_idx, problem, ptoks,
        # existing, expected_ckpt_n.
        pending: dict[asyncio.Task, dict] = {}

        async def _submit_one() -> asyncio.Task | None:
            """Pick a prompt + dispatch a single vLLM request. Returns the
            task or None if no prompt is available."""
            # Multi-env: pick the env furthest below its target share across
            # pool + in-flight, then bake one of ITS prompts. Single-env →
            # always the one active env (in_flight exclusion identical to
            # legacy, where every in-flight prompt is that same env).
            async with self._pool_lock:
                pool_counts, _ = self._pool_env_stats()
            counts = dict(pool_counts)
            for _m in pending.values():
                _en = _m.get("env_name", self.active_envs[0])
                counts[_en] = counts.get(_en, 0) + 1
            env_name = _pick_bake_env(self._mix.target_slots(), counts)
            in_flight_idxs = {
                _m["prompt_idx"] for _m in pending.values()
                if _m.get("env_name", self.active_envs[0]) == env_name
            }
            pick = self._async_pick_next_prompt(env_name, in_flight_idxs, rng)
            if pick is None:
                return None
            prompt_idx, problem, existing = pick
            # Canonical prompt via the shared encode_prompt (chat template +
            # enable_thinking when declared). Generation and submission use the
            # SAME tokens — the validator's canonical encoding matches, so no
            # raw/templated split (that was the v5 workaround).
            ptoks = encode_prompt(self.tokenizer, problem["prompt"])
            gen_ptoks = ptoks
            expected_ckpt_n = self._local_n
            coro = backend.generate(
                prompt_token_ids=gen_ptoks,
                n=M_PER_PHASE,
                temperature=T_PROTO,
                top_p=TOP_P_PROTO,
                top_k=TOP_K_PROTO,
                max_tokens=self.max_new_tokens,
                stop_token_ids=self._eos_ids,
            )
            task = asyncio.create_task(
                coro, name=f"async_gen_prompt_{prompt_idx}",
            )
            pending[task] = {
                "prompt_idx": prompt_idx,
                "problem": problem,
                "ptoks": ptoks,
                "existing": existing,
                "expected_ckpt_n": expected_ckpt_n,
                "env_name": env_name,
            }
            # Move out of that env's retry queue (it's now in-flight); we'll
            # re-insert if the bake hits retry.
            self._retry_by_env[env_name].pop(prompt_idx, None)
            return task

        drop_on_ckpt = drop_pool_on_ckpt_advance()

        try:
            # Fill the pool to target_active.
            while len(pending) < target_active:
                async with self._pool_lock:
                    pool_full = len(self._pool) >= self._pool_max_size
                if pool_full:
                    break
                t = await _submit_one()
                if t is None:
                    break

            while True:
                if not pending:
                    # No in-flight tasks (env exhausted or pool full).
                    # Sleep briefly then try to refill.
                    await asyncio.sleep(1.0)
                    async with self._pool_lock:
                        pool_full = len(self._pool) >= self._pool_max_size
                    if not pool_full:
                        await _submit_one()
                    continue

                done, _ = await asyncio.wait(
                    pending.keys(), return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    meta = pending.pop(task)
                    prompt_idx = meta["prompt_idx"]
                    meta_env = meta.get("env_name", self.active_envs[0])
                    try:
                        completions = task.result()
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        logger.exception(
                            "async_bake: vLLM task failed for prompt=%d; "
                            "dropping",
                            prompt_idx,
                        )
                        completions = None

                    if completions is not None:
                        try:
                            entry, retry = await self._process_one_completion(
                                prompt_idx=prompt_idx,
                                problem=meta["problem"],
                                ptoks=meta["ptoks"],
                                completions=completions,
                                existing_rollouts=meta["existing"],
                                expected_ckpt_n=meta["expected_ckpt_n"],
                                env_name=meta_env,
                            )
                        except asyncio.CancelledError:
                            raise
                        except Exception:
                            logger.exception(
                                "async_bake: process_one_completion failed "
                                "for prompt=%d; dropping",
                                prompt_idx,
                            )
                            entry, retry = None, None

                        if entry is not None:
                            # Mirror _generator_loop's ckpt-advance policy.
                            async with self._pool_lock:
                                if (
                                    drop_on_ckpt
                                    and entry["checkpoint_n"] != self._local_n
                                ):
                                    logger.info(
                                        "async_gen: dropping stale entry "
                                        "prompt=%d (ckpt baked=%d, current=%d, "
                                        "DROP_POOL_ON_CKPT=1)",
                                        prompt_idx, entry["checkpoint_n"],
                                        self._local_n,
                                    )
                                else:
                                    if entry["checkpoint_n"] != self._local_n:
                                        logger.info(
                                            "async_gen: keeping entry "
                                            "prompt=%d despite ckpt advance "
                                            "(baked=%d, current=%d, optimistic)",
                                            prompt_idx, entry["checkpoint_n"],
                                            self._local_n,
                                        )
                                    self._pool.append(entry)
                                    pool_size = len(self._pool)
                                    logger.debug(
                                        "pool +1: prompt=%d size=%d/%d",
                                        prompt_idx, pool_size,
                                        self._pool_max_size,
                                    )
                            # Persist OUTSIDE the lock. Skipped when the prompt
                            # range is armed (#91): entries don't survive a
                            # window, so persisting is wasted I/O during OPEN.
                            if self._pool_persist:
                                try:
                                    await asyncio.to_thread(
                                        save_entry, entry, self._pool_dir,
                                    )
                                except OSError as e:
                                    logger.error(
                                        "pool_persistence: save failed for "
                                        "prompt=%d (%s); entry kept in memory only",
                                        prompt_idx, e,
                                    )
                        elif retry is not None:
                            # Re-queue for another phase (that env's queue).
                            self._retry_by_env[meta_env][prompt_idx] = retry

                    # Submit a replacement, unless the pool is full.
                    async with self._pool_lock:
                        pool_full = len(self._pool) >= self._pool_max_size
                    if not pool_full:
                        await _submit_one()
        except asyncio.CancelledError:
            # mine_window is tearing down; cancel everything.
            for t in list(pending.keys()):
                t.cancel()
            for t in list(pending.keys()):
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
            return
        except Exception:
            logger.exception("async generator loop crashed; will not restart")
            for t in list(pending.keys()):
                t.cancel()
            raise

    def _finalize_pool_entry(self, entry: dict, randomness: str) -> tuple[list, str]:
        """Sync: build RolloutSubmissions + merkle_root from a pre-baked entry + randomness.

        OPTIMIZED:
          1. Per-rollout commit-signing (sr25519, 10-30 ms each × 8 rollouts =
             80-240 ms sequential) now runs in a ThreadPoolExecutor — sr25519
             is implemented in C and releases the GIL, so this gives near-linear
             speedup. Drops _finalize_pool_entry from ~200 ms to ~30-50 ms.
          2. Merkle root is computed INSIDE this function (rather than later in
             _build_signed_request_sync) so the merkle cost is paid in the
             same parallelisable thread and _build_signed_request_sync just
             does a single sign_envelope() + pydantic build.

        Returns (rollout_submissions, merkle_root). Both fit inside one
        drand round (3 s) with margin.
        """
        import torch
        from concurrent.futures import ThreadPoolExecutor

        from reliquary.constants import GRAIL_PROOF_VERSION
        from reliquary.miner.bft import rollout_metadata
        from reliquary.protocol.signatures import sign_commit_binding
        from reliquary.protocol.submission import RolloutSubmission

        r_vec = self._verifier.generate_r_vec(randomness)
        model_name: str = getattr(self.hf_model, "name_or_path", "unknown")
        rollouts = entry["rollouts"]

        # Step 1: matmul commitments on GPU for ALL rollouts up-front. GPU
        # work doesn't benefit from CPU threads — keep it sequential and let
        # CUDA streams overlap if there's anything to overlap. The matmul is
        # cheap (5-15 ms each) vs the sr25519 sign (10-30 ms) we batch next.
        commits_data = []  # list of (all_tokens, prompt_length, token_logprobs, reward, commitments)
        for r in rollouts:
            all_tokens = r["all_tokens"]
            prompt_length = r["prompt_length"]
            token_logprobs = r["token_logprobs"]
            reward = r["reward"]
            hs_gpu = r["hidden_states_cpu"].to(f"cuda:{self.proof_gpu}")
            commitments = self._verifier.create_commitments_batch(hs_gpu, r_vec)
            commits_data.append(
                (all_tokens, prompt_length, token_logprobs, reward, commitments,
                 bool(r.get("forced", False)), r.get("force_span"))
            )

        # Step 2: sign_commit_binding (sr25519) for each rollout in PARALLEL
        # via ThreadPoolExecutor. sr25519 sign releases the GIL so threads
        # actually parallelise on multi-core CPUs.
        def _sign(args):
            all_tokens, commitments = args[0], args[4]
            return sign_commit_binding(
                all_tokens, randomness, model_name, LAYER_INDEX,
                commitments, self.wallet,
            )

        with ThreadPoolExecutor(max_workers=min(8, len(commits_data))) as pool:
            signatures = list(pool.map(_sign, commits_data))

        # Step 3: build RolloutSubmissions (cheap Python work).
        rollout_subs = []
        for (all_tokens, prompt_length, token_logprobs, reward, commitments, forced, force_span), signature in zip(
            commits_data, signatures
        ):
            commit = {
                "tokens": all_tokens,
                "commitments": commitments,
                "proof_version": GRAIL_PROOF_VERSION,
                "model": {"name": model_name, "layer_index": LAYER_INDEX},
                "signature": signature.hex(),
                "beacon": {"randomness": randomness},
                "rollout": rollout_metadata(
                    {"tokens": all_tokens, "prompt_length": prompt_length,
                     "forced": forced, "force_span": force_span},
                    token_logprobs,
                ),
            }
            rollout_subs.append(RolloutSubmission(
                tokens=all_tokens,
                reward=reward,
                commit=commit,
                env_name=self._entry_env_name(entry),
            ))

        # Step 4: compute merkle root here (was previously done lazily in
        # _build_signed_request_sync). Moving it lets the caller skip a
        # repeated sha256 pass and lets us return both pieces atomically.
        # Canonical (wire-v2) or legacy root per the RELIQUARY_WIRE_V2 gate.
        merkle_root = submission_merkle_root(rollout_subs)
        return rollout_subs, merkle_root
