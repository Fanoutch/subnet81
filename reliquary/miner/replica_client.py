"""Client du service « réplique validateur » (ops/replica_service.py).

Appels synchrones et bornés : le mineur les fait depuis le fil de preuve
(``_proof_rollouts``). Toute panne (service absent, timeout, réponse
invalide) rend ``None`` — à l'appelant de décider du repli.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import time

logger = logging.getLogger(__name__)


def replica_socket_path() -> str | None:
    """``RELIQUARY_REPLICA_SOCKET`` ; vide ou absent = réplique non utilisée."""
    path = os.environ.get("RELIQUARY_REPLICA_SOCKET", "").strip()
    return path or None


# file d'écoute du service momentanément pleine : on réessaie dans le délai
_RETRY_CONNECT = (BlockingIOError, ConnectionRefusedError, InterruptedError)


def _call(socket_path: str, req: dict, timeout: float) -> dict | None:
    deadline = time.monotonic() + float(timeout)
    pause = 0.02
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            while True:
                try:
                    s.connect(socket_path)
                    break
                except _RETRY_CONNECT:
                    if time.monotonic() + pause >= deadline:
                        raise
                    time.sleep(pause)
                    pause = min(pause * 2, 0.2)
            s.sendall((json.dumps(req) + "\n").encode())
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = s.recv(1 << 16)
                if not chunk:
                    break
                buf += chunk
        resp = json.loads(buf)
    except (OSError, ValueError) as exc:
        logger.warning("réplique indisponible (%s): %r", req.get("op"), exc)
        return None
    if not isinstance(resp, dict) or not resp.get("ok"):
        logger.warning("réplique: réponse en échec: %s",
                       (resp or {}).get("error") if isinstance(resp, dict) else resp)
        return None
    return resp


def ping(socket_path: str, *, timeout: float = 2.0) -> bool:
    return _call(socket_path, {"op": "ping"}, timeout) is not None


def load(socket_path: str, model_path: str, *, timeout: float = 120.0) -> bool:
    """Précharge un checkpoint dans la réplique (appelé au préchargement)."""
    return _call(socket_path, {"op": "load", "model_path": model_path},
                 timeout) is not None


def terminal_verdicts(
    socket_path: str, *, model_path: str, randomness: str,
    checkpoint_hash: str, prompt_idx: int, items: list[dict],
    timeout: float = 60.0,
) -> list[dict] | None:
    """Un verdict par item, dans l'ordre : ``{"ok", "pick", "cdf_miss"}``."""
    resp = _call(socket_path, {
        "op": "terminal", "model_path": model_path, "randomness": randomness,
        "checkpoint_hash": checkpoint_hash, "prompt_idx": int(prompt_idx),
        "items": items,
    }, timeout)
    if resp is None:
        return None
    results = resp.get("results")
    if not isinstance(results, list) or len(results) != len(items):
        logger.warning("réplique: nombre de verdicts incohérent")
        return None
    return results
