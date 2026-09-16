"""Vérification de l'EOS final par la réplique DÈS la sortie d'un rollout.

Mesuré le 14/09 (fen 45914-45917) : prêt → précommit reçu 14,0 s p50, dont
9,4 s de réparation EOS — le groupe attendait ses 16 rollouts puis les faisait
vérifier d'un bloc, derrière les autres groupes du bake. Ici chaque rollout
part en vérification à l'instant où vLLM le termine (rappel ``on_rollout`` du
bake en streaming) ; ``MiningEngine._repair_terminal_eos`` lit ces verdicts au
1er tour et n'interroge la réplique que pour les manquants.

Un verdict n'est réutilisé que pour EXACTEMENT les mêmes tokens, sous le même
contexte ``(randomness, checkpoint_hash, model_path)``.
"""
from __future__ import annotations

import collections
import hashlib
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable

logger = logging.getLogger(__name__)


def _digest(tokens) -> str:
    h = hashlib.blake2b(digest_size=16)
    for t in tokens:
        h.update(int(t).to_bytes(4, "little", signed=False))
    return h.hexdigest()


class EarlyTerminalVerifier:
    """``verify_fn(ctx, prompt_idx, items) -> list[dict] | None`` (réplique)."""

    def __init__(self, verify_fn: Callable, *, eos_ids, max_workers: int = 4,
                 keep_contexts: int = 2) -> None:
        self._verify_fn = verify_fn
        self._eos = {int(x) for x in eos_ids}
        self._pool = ThreadPoolExecutor(max_workers=max_workers,
                                        thread_name_prefix="early-eos")
        self._lock = threading.Lock()
        self._keep = max(1, int(keep_contexts))
        # ctx -> {(prompt_idx, rollout): (digest, Future)}
        self._by_ctx: "collections.OrderedDict[tuple, dict]" = collections.OrderedDict()

    def submit(self, ctx, prompt_idx, rollout, tokens, *, prompt_len) -> None:
        tokens = [int(x) for x in tokens]
        if len(tokens) < 2 or len(tokens) - 1 - int(prompt_len) < 0:
            return
        if tokens[-1] not in self._eos:
            return
        item = {"rollout": int(rollout), "prompt_len": int(prompt_len),
                "tokens": tokens}
        fut: Future = self._pool.submit(self._run, ctx, int(prompt_idx), item)
        with self._lock:
            slot = self._by_ctx.get(ctx)
            if slot is None:
                slot = self._by_ctx[ctx] = {}
                while len(self._by_ctx) > self._keep:
                    self._by_ctx.popitem(last=False)
            else:
                self._by_ctx.move_to_end(ctx)
            slot[(int(prompt_idx), int(rollout))] = (_digest(tokens), fut)

    def _run(self, ctx, prompt_idx, item):
        try:
            res = self._verify_fn(ctx, prompt_idx, [item])
        except Exception:
            logger.debug("vérification précoce en échec", exc_info=True)
            return None
        if not res or len(res) != 1:
            return None
        return res[0]

    def lookup(self, ctx, prompt_idx, rollout, tokens, *, timeout: float = 60.0):
        """Verdict ``{ok, pick, cdf_miss}`` si connu pour ces tokens, sinon None."""
        with self._lock:
            hit = (self._by_ctx.get(ctx) or {}).get((int(prompt_idx), int(rollout)))
        if hit is None or hit[0] != _digest(tokens):
            return None
        try:
            return hit[1].result(timeout=timeout)
        except Exception:
            return None

    def cancel_prompt(self, prompt_idx) -> int:
        """Annule les vérifications PAS ENCORE DÉMARRÉES d'un groupe jeté hors
        zone ; renvoie le nombre d'annulées (16/09).

        Chaque rollout part en vérification dès sa sortie du décodeur, donc
        avant le grading du groupe — or 46-54 % des groupes sont ensuite jetés
        ``out_of_zone``, et la réplique est une file FIFO à 2 workers : ces
        vérifications condamnées passent devant les utiles. Un ``Future`` déjà
        en cours n'est pas touché (``cancel()`` renvoie False) ; ``lookup()``
        sur un annulé renvoie None (``CancelledError`` est une ``Exception``).
        Toutes les ``ctx`` sont balayées : ``prompt_idx`` est unique dans une
        fenêtre et on ne garde que ``keep_contexts`` contextes."""
        n = 0
        pi = int(prompt_idx)
        with self._lock:
            for slot in self._by_ctx.values():
                for (p_idx, _r), (_dg, fut) in list(slot.items()):
                    if p_idx == pi and fut.cancel():
                        n += 1
        return n

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
