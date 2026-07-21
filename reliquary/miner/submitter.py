"""HTTP client used by miners to push GRPO submissions to the validator.

V1 assumption: a single validator. Discovery returns the first axon advertised
by a hotkey holding `validator_permit`. Multi-validator routing is intentionally
out of scope here — see the GRPO refactor plan.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any
from urllib.parse import quote

import httpx

from reliquary.constants import VALIDATOR_HTTP_PORT
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    BatchSubmissionResponse,
    GrpoBatchState,
    RejectReason,
    SubmissionPrecommitRequest,
    SubmissionPrecommitResponse,
    VerdictsResponse,
)

logger = logging.getLogger(__name__)

# Retry configuration: 3 attempts, exponential backoff 1s / 2s / 4s.
_RETRY_DELAYS = (1.0, 2.0, 4.0)
# Default timeout is generous: the validator may need several seconds to verify
# a submission even in the async-queue path (the queue can back up under load).
# Miners running against slow links (Targon port-forward etc.) benefit further.
_DEFAULT_TIMEOUT = 60.0
# Header carrying the precommit receipt on the body reveal (upstream 8835a95).
_PRECOMMIT_HEADER = "X-Reliquary-Precommit"


class NoValidatorFoundError(RuntimeError):
    """No metagraph entry advertises a usable validator endpoint."""


class SubmissionError(RuntimeError):
    """All submission retries exhausted."""


def discover_validator_url(metagraph: Any, port: int = VALIDATOR_HTTP_PORT) -> str:
    """Return the HTTP URL of the first validator advertised on the metagraph.

    Picks the first uid with validator_permit=True and an axon IP that isn't
    the unset placeholder. Multi-validator coordination is out of scope; this
    deliberately picks ONE validator.
    """
    permits = getattr(metagraph, "validator_permit", None)
    axons = getattr(metagraph, "axons", None)
    if permits is None or axons is None:
        raise NoValidatorFoundError(
            "metagraph missing validator_permit or axons attributes"
        )
    for uid, (permit, axon) in enumerate(zip(permits, axons)):
        if not permit:
            continue
        ip = getattr(axon, "ip", None)
        if not ip or ip in ("0.0.0.0", ""):
            continue
        # Use the validator's own port if it's set; fall back to the protocol default.
        axon_port = getattr(axon, "port", None) or port
        return f"http://{ip}:{axon_port}"
    raise NoValidatorFoundError("no validator with permit and routable axon")


async def _post_with_retry(
    full_url: str,
    json_payload: dict,
    response_model: type,
    *,
    client: httpx.AsyncClient | None,
    timeout: float,
) -> Any:
    last_exc: Exception | None = None
    own_client = client is None
    cli = client or httpx.AsyncClient(timeout=timeout)
    try:
        for attempt, delay in enumerate(_RETRY_DELAYS, start=1):
            try:
                resp = await cli.post(full_url, json=json_payload, timeout=timeout)
            except (httpx.RequestError, httpx.TimeoutException) as e:
                last_exc = e
                logger.warning(
                    "submit attempt %d to %s failed: %r (type=%s)",
                    attempt, full_url, e, type(e).__name__,
                )
                if attempt < len(_RETRY_DELAYS):
                    await asyncio.sleep(delay)
                continue
            # 503 "no active window" is informational for BatchSubmissionResponse —
            # don't retry, surface as a structured reject.
            if resp.status_code == 503 and response_model is BatchSubmissionResponse:
                return BatchSubmissionResponse(
                    accepted=False, reason=RejectReason.WINDOW_NOT_ACTIVE
                )
            # 4xx means the request is malformed or the validator rejected it
            # for a deterministic reason — retrying is pointless. Parse and return.
            if 400 <= resp.status_code < 500:
                detail = _safe_detail(resp)
                if response_model is BatchSubmissionResponse:
                    if resp.status_code == 409:
                        reason = RejectReason.WINDOW_MISMATCH
                    elif resp.status_code == 422:
                        # Pydantic schema validation failed: a miner-side payload
                        # bug (missing/renamed field), NOT a normal validator
                        # reject. Surface it loudly — masking it as BAD_PROMPT_IDX
                        # is exactly what hid the v6 env_name break.
                        logger.error(
                            "submit to %s got HTTP 422 (schema mismatch — likely a "
                            "miner payload bug, NOT a real reject): %s",
                            full_url, detail,
                        )
                        reason = RejectReason.BAD_PROMPT_IDX
                    else:
                        logger.warning(
                            "submit to %s got HTTP %d, mapping to BAD_PROMPT_IDX: %s",
                            full_url, resp.status_code, detail,
                        )
                        reason = RejectReason.BAD_PROMPT_IDX
                    return BatchSubmissionResponse(accepted=False, reason=reason)
                raise SubmissionError(f"HTTP {resp.status_code}: {detail}")
            if resp.status_code >= 500:
                last_exc = SubmissionError(f"HTTP {resp.status_code}")
                if attempt < len(_RETRY_DELAYS):
                    await asyncio.sleep(delay)
                continue
            return response_model.model_validate(resp.json())
        raise SubmissionError(f"all retries failed: {last_exc}")
    finally:
        if own_client:
            await cli.aclose()


async def _get_with_retry(
    full_url: str,
    response_model: type,
    *,
    client: httpx.AsyncClient | None,
    timeout: float,
) -> Any:
    last_exc: Exception | None = None
    own_client = client is None
    cli = client or httpx.AsyncClient(timeout=timeout)
    try:
        for attempt, delay in enumerate(_RETRY_DELAYS, start=1):
            try:
                resp = await cli.get(full_url, timeout=timeout)
            except (httpx.RequestError, httpx.TimeoutException) as e:
                last_exc = e
                if attempt < len(_RETRY_DELAYS):
                    await asyncio.sleep(delay)
                continue
            if resp.status_code == 503:
                # No active window yet — caller's job to handle.
                raise SubmissionError(f"no active window at {full_url}")
            if resp.status_code == 404:
                raise SubmissionError(f"endpoint not found: {full_url}")
            if 400 <= resp.status_code < 500:
                raise SubmissionError(
                    f"HTTP {resp.status_code}: {_safe_detail(resp)}"
                )
            if resp.status_code >= 500:
                last_exc = SubmissionError(f"HTTP {resp.status_code}")
                if attempt < len(_RETRY_DELAYS):
                    await asyncio.sleep(delay)
                continue
            return response_model.model_validate(resp.json())
        raise SubmissionError(f"all retries failed: {last_exc}")
    finally:
        if own_client:
            await cli.aclose()


async def _post_bytes_with_retry(
    full_url: str,
    content: bytes,
    *,
    headers: dict,
    client: httpx.AsyncClient,
    timeout: float,
) -> BatchSubmissionResponse:
    """POST verbatim bytes, mirroring _post_with_retry's status mapping.

    Distinct from _post_with_retry because the precommitted body must go on the
    wire byte-for-byte: httpx's ``json=`` would re-serialize and could break the
    committed sha256.
    """
    last_exc: Exception | None = None
    for attempt, delay in enumerate(_RETRY_DELAYS, start=1):
        try:
            resp = await client.post(
                full_url, content=content, headers=headers, timeout=timeout,
            )
        except (httpx.RequestError, httpx.TimeoutException) as e:
            last_exc = e
            logger.warning(
                "reveal attempt %d to %s failed: %r", attempt, full_url, e,
            )
            if attempt < len(_RETRY_DELAYS):
                await asyncio.sleep(delay)
            continue
        if resp.status_code == 503:
            return BatchSubmissionResponse(
                accepted=False, reason=RejectReason.WINDOW_NOT_ACTIVE
            )
        if 400 <= resp.status_code < 500:
            detail = _safe_detail(resp)
            if resp.status_code == 409:
                reason = RejectReason.WINDOW_MISMATCH
            else:
                logger.error(
                    "reveal to %s got HTTP %d, mapping to BAD_PROMPT_IDX: %s",
                    full_url, resp.status_code, detail,
                )
                reason = RejectReason.BAD_PROMPT_IDX
            return BatchSubmissionResponse(accepted=False, reason=reason)
        if resp.status_code >= 500:
            last_exc = SubmissionError(f"HTTP {resp.status_code}")
            if attempt < len(_RETRY_DELAYS):
                await asyncio.sleep(delay)
            continue
        return BatchSubmissionResponse.model_validate(resp.json())
    raise SubmissionError(f"all reveal retries failed: {last_exc}")


def _safe_detail(resp: httpx.Response) -> str:
    try:
        body = resp.json()
        if isinstance(body, dict) and "detail" in body:
            return str(body["detail"])
        return str(body)[:200]
    except Exception:
        return resp.text[:200]


async def submit_batch_v2(
    url: str,
    request: BatchSubmissionRequest,
    *,
    client: httpx.AsyncClient | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
    wallet: Any = None,
    randomness: str | None = None,
) -> BatchSubmissionResponse:
    """POST a v2 batch submission. Retries network errors; 4xx is final.

    With ``wallet`` + ``randomness`` this performs the MANDATORY two-phase
    upload-precommit handshake (upstream 8835a95): a small signed commitment to
    the exact payload bytes goes to ``/submit/precommit`` first, and the body is
    revealed to ``/submit`` under the returned receipt. The validator rejects
    every bare ``/submit`` with ``PRECOMMIT_REQUIRED`` while the difficulty
    auction is enforced (it is, on both our envs).

    Omitting ``wallet`` keeps the legacy one-shot path — used by tests and by
    any validator predating the handshake.
    """
    if wallet is None or randomness is None:
        payload = request.model_dump(mode="json")
        return await _post_with_retry(
            f"{url}/submit", payload, BatchSubmissionResponse,
            client=client, timeout=timeout,
        )
    return await _submit_with_precommit(
        url, request, client=client, timeout=timeout,
        wallet=wallet, randomness=randomness,
    )


def _build_precommit(
    request: BatchSubmissionRequest, *, wallet: Any, randomness: str,
) -> tuple[bytes, SubmissionPrecommitRequest]:
    """Serialize the request ONCE and sign a commitment to those exact bytes.

    The payload must be posted verbatim afterwards: the validator re-hashes the
    revealed body and compares it to ``payload_sha256``. Re-serializing (e.g.
    via httpx ``json=``) risks different bytes and a PRECOMMIT_INVALID.
    """
    from reliquary.protocol.signatures import sign_precommit

    environments = {r.env_name for r in request.rollouts}
    if len(environments) != 1:
        raise SubmissionError(
            f"submission must contain exactly one environment, got {environments}"
        )

    payload = request.model_dump_json().encode("utf-8")
    fields = {
        "miner_hotkey": request.miner_hotkey,
        "window_start": request.window_start,
        "prompt_idx": request.prompt_idx,
        "merkle_root": request.merkle_root,
        "checkpoint_hash": request.checkpoint_hash,
        "environment": next(iter(environments)),
        "payload_bytes": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "drand_round": request.drand_round,
        "randomness": randomness,
        "protocol_version": request.protocol_version,
        "nonce": request.nonce,
    }
    signature = sign_precommit(wallet=wallet, **fields).hex()
    precommit = SubmissionPrecommitRequest(
        # randomness is signed but NOT transmitted — the validator substitutes
        # its own window randomness and forbids extra fields.
        **{k: v for k, v in fields.items() if k != "randomness"},
        precommit_signature=signature,
    )
    return payload, precommit


async def _submit_with_precommit(
    url: str,
    request: BatchSubmissionRequest,
    *,
    client: httpx.AsyncClient | None,
    timeout: float,
    wallet: Any,
    randomness: str,
) -> BatchSubmissionResponse:
    payload, precommit = _build_precommit(
        request, wallet=wallet, randomness=randomness,
    )
    own_client = client is None
    cli = client or httpx.AsyncClient(timeout=timeout)
    try:
        # --- phase 1: claim the arrival slot with the small signed commitment.
        # The validator stamps rank at PRECOMMIT arrival (server.py:3606), so
        # this POST — not the body upload — is what races the auction.
        pre_resp = await cli.post(
            f"{url}/submit/precommit",
            content=precommit.model_dump_json().encode("utf-8"),
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        if pre_resp.status_code == 404:
            logger.warning(
                "validator has no /submit/precommit; falling back to direct submit"
            )
            return await _post_with_retry(
                f"{url}/submit", request.model_dump(mode="json"),
                BatchSubmissionResponse, client=cli, timeout=timeout,
            )
        if pre_resp.status_code >= 400:
            detail = _safe_detail(pre_resp)
            logger.error(
                "precommit to %s got HTTP %d (miner-side payload bug, NOT a "
                "normal reject): %s", url, pre_resp.status_code, detail,
            )
            raise SubmissionError(
                f"precommit HTTP {pre_resp.status_code}: {detail}"
            )

        verdict = SubmissionPrecommitResponse.model_validate(pre_resp.json())
        if not verdict.accepted:
            # Refused before the body moved — don't burn window time uploading.
            logger.warning(
                "precommit rejected window=%d prompt=%d reason=%s",
                request.window_start, request.prompt_idx,
                verdict.reason.value if hasattr(verdict.reason, "value")
                else verdict.reason,
            )
            return BatchSubmissionResponse(
                accepted=False, reason=verdict.reason,
            )
        receipt_id = verdict.receipt_id
        if not receipt_id:
            raise SubmissionError("accepted precommit omitted receipt_id")

        # --- phase 2: reveal the byte-identical body under the receipt.
        return await _post_bytes_with_retry(
            f"{url}/submit", payload,
            headers={
                "Content-Type": "application/json",
                _PRECOMMIT_HEADER: receipt_id,
            },
            client=cli, timeout=timeout,
        )
    finally:
        if own_client:
            await cli.aclose()


def build_state_url(url: str, env: str | None = None) -> str:
    """``{url}/state``, with optional ``?env=`` (per-env cooldown, validator #88).

    Omitting ``env`` is byte-identical to the legacy call (first active env).
    A pre-#88 validator ignores the unknown query param, so this is safe both
    ways.
    """
    base = f"{url}/state"
    return f"{base}?env={quote(env, safe='')}" if env is not None else base


def build_verdicts_url(url: str, hotkey: str, since: float | None = None) -> str:
    """``{url}/verdicts/{hotkey}``, with optional ``?since=<ts>`` incremental cursor.

    Mirrors the validator's GET /verdicts/{hotkey} endpoint (PR #25). The
    hotkey is path-encoded; ss58 addresses are URL-safe but encode defensively.
    """
    base = f"{url}/verdicts/{quote(hotkey, safe='')}"
    return f"{base}?since={since}" if since is not None else base


async def fetch_verdicts(url, hotkey, *, client, since=None):
    """GET the recent verdicts ring for ``hotkey``. Returns a VerdictsResponse,
    or ``None`` on any transport/HTTP/parse failure (caller treats None as
    'no new signal this tick' — never raises, never kills the loop)."""
    try:
        r = await client.get(build_verdicts_url(url, hotkey, since), timeout=5.0)
        if r.status_code != 200:
            return None
        return VerdictsResponse.model_validate(r.json())
    except Exception:
        return None


async def get_window_state_v2(
    url: str,
    *,
    env: str | None = None,
    client: httpx.AsyncClient | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> GrpoBatchState:
    """GET the validator's current v2 GrpoBatchState (optionally per-env)."""
    return await _get_with_retry(
        build_state_url(url, env), GrpoBatchState,
        client=client, timeout=timeout,
    )


async def get_window_state_v2_with_resp(
    url: str,
    *,
    env: str | None = None,
    client: httpx.AsyncClient,
    timeout: float = _DEFAULT_TIMEOUT,
) -> tuple[GrpoBatchState, httpx.Response, float, float]:
    """GET /state and also expose the raw response + timing.

    Caller uses the response's ``Date`` header (the validator's NTP-synced
    wall clock at response generation) to recalibrate the miner's clock
    offset live, replacing the slower drand-network calibration loop. Returns
    ``(state, response, t_send, t_recv)`` so the caller can compute the
    half-RTT-corrected validator-vs-local-clock offset.

    No retry: this is called on a tight poll loop (~200 Hz); the caller
    already handles transient errors by skipping the iteration.
    """
    t_send = asyncio.get_running_loop().time()
    wall_send = __import__("time").time()
    resp = await client.get(build_state_url(url, env), timeout=timeout)
    wall_recv = __import__("time").time()
    if resp.status_code == 503:
        raise SubmissionError(f"no active window at {url}/state")
    if resp.status_code == 404:
        raise SubmissionError(f"endpoint not found: {url}/state")
    if 400 <= resp.status_code:
        raise SubmissionError(f"HTTP {resp.status_code}: {_safe_detail(resp)}")
    state = GrpoBatchState.model_validate(resp.json())
    return state, resp, wall_send, wall_recv


