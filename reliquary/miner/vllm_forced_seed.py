"""Forced-seed logits processing for the vLLM generation path.

Under FORCED_SEED_ENFORCE every generated token must equal the public inverse-CDF
pick derived from ``u_at`` (v2, hotkey-free). On the HF path this is
``miner/forced_seed_sampler.ForcedSeedLogitsProcessor``. vLLM 0.24 exposes a
per-request logits processor as a ``Callable[[output_token_ids, logits], logits]``
wrapped by ``AdapterLogitsProcessor`` (which owns all batch bookkeeping), so we
only need:

* ``force_row`` — the pure per-position transform (torch-only, no vLLM), the
  single source of the forcing math, byte-parity with the HF processor for
  identical input logits.
* ``make_forced_seed_request_proc`` — build the per-request closure that derives
  the completion position ``t`` from the tokens generated so far and applies
  ``force_row`` with that request's ``rollout_index``.
* ``build_forced_seed_logitsproc_class`` — factory that imports vLLM lazily and
  returns the ``AdapterLogitsProcessor`` subclass (kept out of module import so
  the pure functions stay testable on a box without vLLM installed).

Per-request params travel on ``SamplingParams.extra_args["forced_seed"]`` as a
dict: ``{randomness, prompt_idx, checkpoint_hash, rollout_index, base_offset,
start_len}``.
"""
from __future__ import annotations

import torch

from reliquary.constants import T_PROTO, TOP_K_PROTO, TOP_P_PROTO
from reliquary.environment.forced_sampling import u_at, warp, pick

FORCED_SEED_EXTRA_KEY = "forced_seed"


def forced_seed_extra_args(*, randomness: str, prompt_idx: int,
                           checkpoint_hash: str, rollout_index: int,
                           base_offset: int, start_len: int) -> dict:
    """The per-request forced-seed payload the backend nests under
    ``SamplingParams.extra_args[FORCED_SEED_EXTRA_KEY]``. Its keys are exactly
    what ``make_forced_seed_request_proc`` / ``new_req_logits_processor`` read
    back — single source of the producer/consumer contract."""
    return {
        "randomness": randomness,
        "prompt_idx": prompt_idx,
        "checkpoint_hash": checkpoint_hash,
        "rollout_index": rollout_index,
        "base_offset": base_offset,
        "start_len": start_len,
    }


def force_row(logits_row: torch.Tensor, randomness: str, prompt_idx: int,
              checkpoint_hash: str, rollout_index: int, t: int) -> torch.Tensor:
    """Return ``logits_row`` masked to force the inverse-CDF pick at position ``t``.

    All entries become ``-inf`` except the forced token, set to ``0.0`` — exactly
    what ``ForcedSeedLogitsProcessor.__call__`` writes for one row, so a greedy
    sampler downstream emits the forced token.
    """
    u = u_at(randomness, prompt_idx, checkpoint_hash, rollout_index, t)
    # NOTE 2026-08-03: warp_fast (top_k-restricted sort) was wired here and
    # MEASURED SLOWER on GPU (1318 vs 1452 tok/s): its nonzero() forces a
    # CPU<->GPU sync per call, and the GPU sorts 151k cheaply anyway. The real
    # forced-seed cost is the per-row Python loop (128 processor calls/step) —
    # the fix is a BATCH-level processor (force_rows_batched style), not a
    # faster per-row warp. warp_fast stays available for CPU-side paths.
    probs = warp(logits_row, t=T_PROTO, top_k=TOP_K_PROTO, top_p=TOP_P_PROTO)
    tok = pick(probs, u)
    out = torch.full_like(logits_row, float("-inf"))
    out[tok] = 0.0
    return out


def make_forced_seed_request_proc(*, randomness: str, prompt_idx: int,
                                  checkpoint_hash: str, rollout_index: int,
                                  base_offset: int, start_len: int):
    """Build a per-request vLLM logits processor closure.

    Signature matches vLLM's 2-arg ``RequestLogitsProcessor``:
    ``(output_token_ids, logits) -> logits``. The completion position is
    ``t = base_offset + len(output_token_ids)`` — mirrors the HF processor's
    ``t = base_offsets[r] + (input_ids.shape[1] - start_len)`` where the number
    of already-generated tokens equals ``len(output_token_ids)``. ``start_len``
    is accepted for parity/validation with the HF sampler but the count comes
    from vLLM's output-token bookkeeping directly.
    """
    def _proc(output_token_ids, logits: torch.Tensor) -> torch.Tensor:
        t = base_offset + len(output_token_ids)
        return force_row(logits, randomness, prompt_idx, checkpoint_hash,
                         rollout_index, t)

    return _proc


def batched_force_mask(logits: torch.Tensor, entries) -> torch.Tensor:
    """ONE vectorized forced-seed pass over all forced rows of a batch.

    ``entries`` = ``[(row_idx, u), ...]``. Uses ``force_rows_batched`` (bit-exact
    vs per-row warp+pick — it already powers the HF path) to compute every forced
    token in a single tensor pass, then writes the same mask ``force_row`` wrote
    (-inf everywhere, 0.0 at the forced token) with two vectorized ops.

    Why: vLLM's AdapterLogitsProcessor.apply() loops Python-per-row (one
    ``force_row`` per sequence per step). Measured on the 4B/H100 that loop IS
    the forced-seed tax (fs_on 1452 vs fs_off 3487 tok/s); a faster per-row warp
    changed nothing (-9%). Batch the step, not the row.
    """
    if not entries:
        return logits
    # NOTE 2026-08-03: force_rows_batched_fast (top_k-restricted sort+cdf,
    # bit-exact, parity PASS) was wired here and MEASURED -5% (5568 vs 5859):
    # the GPU sorts 151k cheaply; the tie-guard syncs ate the savings. The
    # remaining forced/free gap (0.61) needs PROFILING, not more guesses.
    from reliquary.environment.forced_sampling import force_rows_batched

    rows = torch.tensor([int(r) for r, _ in entries],
                        device=logits.device, dtype=torch.long)
    us = [u for _, u in entries]
    toks = force_rows_batched(
        logits.index_select(0, rows), us,
        t=T_PROTO, top_k=TOP_K_PROTO, top_p=TOP_P_PROTO,
    )
    logits[rows] = float("-inf")
    logits[rows, toks] = 0.0
    return logits


# Chronométrage par section du processeur (diagnostic UNIQUEMENT — les
# torch.cuda.synchronize() du mode timing faussent le débit absolu ; ne
# jamais l'activer en production). RELIQUARY_FS_TIMING=1.
import os as _os_timing
_FS_TIMING = _os_timing.environ.get("RELIQUARY_FS_TIMING", "0") == "1"
_fs_timing_acc = {"steps": 0, "u_s": 0.0, "tensor_s": 0.0}

# Passe forced-seed capturée en CUDA GRAPH (chantier 2026-08-15).
# Profil mesuré : la passe tensorielle coûte 1,385 ms/step à n=8 (96 % du
# surcoût FS) alors que les tenseurs sont minuscules — c'est le LANCEMENT de
# ~16 kernels eager par step qui domine, le processeur vivant hors des CUDA
# graphs de vLLM. Un replay de graph = 1 lancement, mêmes kernels dans le même
# ordre → BIT-EXACT par construction. Off par défaut ; gate de parité + A/B
# tokens identiques exigés avant toute activation en production.
_FS_GRAPH = _os_timing.environ.get("RELIQUARY_FS_GRAPH", "0") == "1"
# Diagnostic UNIQUEMENT : processeur présent mais inerte — mesure le coût du
# MÉCANISME (mode piecewise de vLLM + crochet Python par step) sans notre
# calcul. Produirait des tokens non conformes : jamais en production.
_FS_NOOP = _os_timing.environ.get("RELIQUARY_FS_NOOP", "0") == "1"


class _FsGraphPass:
    """CUDA-graph replay de ``force_rows_batched`` + écriture du masque.

    Une capture par forme ``(n, vocab)`` : buffers statiques (logits des
    lignes forcées, u, sortie masquée), capture au premier apply de la forme,
    replay ensuite. La boucle u_at (CPU, protocole) reste dehors. Tout échec
    de capture ⇒ repli eager définitif (drapeau ``dead``) — jamais pire que
    l'état actuel.
    """

    def __init__(self) -> None:
        self._graphs: dict[tuple, tuple] = {}   # (n, vocab) -> (graph, in_lg, in_u, out_masked)
        self.dead = False

    def run(self, rows_logits: torch.Tensor, u_dev: torch.Tensor):
        """Retourne le bloc de logits masqué ([n, vocab], -inf sauf 0.0 au
        token forcé) — équivalent bit-exact du chemin eager, ou None si le
        mode graph est mort (l'appelant repasse en eager)."""
        if self.dead:
            return None
        from reliquary.environment.forced_sampling import force_rows_batched
        key = (rows_logits.shape[0], rows_logits.shape[1])
        try:
            entry = self._graphs.get(key)
            if entry is None:
                in_lg = torch.empty_like(rows_logits)
                in_u = torch.empty_like(u_dev)
                in_lg.copy_(rows_logits)
                in_u.copy_(u_dev)
                # Warmup hors capture (allocations/autotune) sur un stream dédié
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(2):
                        _fs_masked_block(in_lg, in_u, force_rows_batched)
                torch.cuda.current_stream().wait_stream(s)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    out_masked = _fs_masked_block(in_lg, in_u, force_rows_batched)
                self._graphs[key] = (g, in_lg, in_u, out_masked)
                entry = self._graphs[key]
            g, in_lg, in_u, out_masked = entry
            in_lg.copy_(rows_logits)
            in_u.copy_(u_dev)
            g.replay()
            return out_masked
        except Exception:
            import logging
            logging.getLogger(__name__).exception(
                "fs-graph: capture/replay échoué — repli eager définitif")
            self.dead = True
            self._graphs.clear()
            return None


def _fs_masked_block(lg: torch.Tensor, u: torch.Tensor, force_rows_batched):
    """Le corps capturable : mêmes appels que le chemin eager, sur buffers
    statiques, sortie = bloc masqué (-inf partout, 0.0 au token forcé)."""
    toks = force_rows_batched(lg, u, t=T_PROTO, top_k=TOP_K_PROTO,
                              top_p=TOP_P_PROTO)
    out = torch.full_like(lg, float("-inf"))
    out.scatter_(1, toks.unsqueeze(1), 0.0)
    return out


class ForcedRowsState:
    """Device-resident state for the batched processor (étude §3.2).

    Everything the per-step hook touches lives on the device and is REUSED:
      * ``_rows`` — persistent LongTensor of forced row indices, rebuilt ONLY
        when the batch composition changes (``rebuild``), not per step;
      * ``_u_stage`` — CPU staging buffer for the per-step ``u`` values,
        pinned when the target device is CUDA, written in place;
      * ``_u_dev`` — device buffer filled via ``copy_(non_blocking=True)`` so
        the H2D copy overlaps compute instead of stalling the stream;
      * grow-only capacity — no per-step allocation, ever.
    The SHA-256 ``u_at`` hashes stay on CPU (protocol-defined), but they are
    the only per-step Python work left.
    """

    def __init__(self) -> None:
        self._slots: list[tuple[dict, list]] = []
        self._rows = None
        self._u_stage = None
        self._u_dev = None
        self._cap = 0
        self._graph_pass = _FsGraphPass() if _FS_GRAPH else None

    def rebuild(self, req_info: dict, device) -> None:
        items = sorted(req_info.items())
        self._slots = [entry for _, entry in items]
        n = len(items)
        if n == 0:
            self._rows = None
            return
        dev = torch.device(device)
        self._rows = torch.tensor([idx for idx, _ in items],
                                  device=dev, dtype=torch.long)
        if self._u_stage is None or n > self._cap or (
                self._u_dev is not None and self._u_dev.device != dev):
            cap = max(n, self._cap)
            pin = dev.type == "cuda"
            self._u_stage = torch.empty(cap, dtype=torch.float32,
                                        pin_memory=pin)
            self._u_dev = torch.empty(cap, dtype=torch.float32, device=dev)
            self._cap = cap

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self._slots:
            return logits
        from reliquary.environment.forced_sampling import force_rows_batched

        if _FS_NOOP:
            return logits
        timing = _FS_TIMING
        if timing:
            import time as _time
            torch.cuda.synchronize()
            t0 = _time.perf_counter()
        n = len(self._slots)
        stage = self._u_stage
        for i, (fs, out_ids) in enumerate(self._slots):
            t = fs.get("base_offset", 0) + len(out_ids)
            stage[i] = u_at(fs["randomness"], fs["prompt_idx"],
                            fs["checkpoint_hash"], fs["rollout_index"], t)
        if timing:
            t1 = _time.perf_counter()
        u_dev = self._u_dev[:n]
        u_dev.copy_(stage[:n], non_blocking=True)
        rows = self._rows
        gp = self._graph_pass
        masked = None
        if gp is not None and logits.is_cuda:
            masked = gp.run(logits.index_select(0, rows), u_dev)
        if masked is not None:
            logits[rows] = masked
        else:
            toks = force_rows_batched(
                logits.index_select(0, rows), u_dev,
                t=T_PROTO, top_k=TOP_K_PROTO, top_p=TOP_P_PROTO,
            )
            logits[rows] = float("-inf")
            logits[rows, toks] = 0.0
        if timing:
            torch.cuda.synchronize()
            t2 = _time.perf_counter()
            _fs_timing_acc["steps"] += 1
            _fs_timing_acc["u_s"] += t1 - t0
            _fs_timing_acc["tensor_s"] += t2 - t1
            if _fs_timing_acc["steps"] % 200 == 0:
                import sys as _sys
                s = _fs_timing_acc
                print(f"[fs-timing] steps={s['steps']} n={n} "
                      f"u_loop={1e3*s['u_s']/s['steps']:.3f}ms/step "
                      f"tensor={1e3*s['tensor_s']/s['steps']:.3f}ms/step",
                      file=_sys.stderr, flush=True)
        return logits


def build_forced_seed_batched_logitsproc_class():
    """Native v1 batch-level LogitsProcessor (imports vLLM lazily).

    Same wire behaviour as ``build_forced_seed_logitsproc_class`` (the adapter),
    but ``apply`` runs ONE ``batched_force_mask`` pass per step instead of a
    Python loop over rows. Slot bookkeeping delegates to vLLM's own
    ``process_dict_updates`` (the exact helper the adapter uses), and ``t`` is
    derived from the live ``output_tok_ids`` reference vLLM guarantees.
    """
    from vllm.v1.sample.logits_processor import (
        LogitsProcessor, process_dict_updates,
    )

    class VLLMForcedSeedBatchedLogitsProcessor(LogitsProcessor):
        def __init__(self, vllm_config, device, is_pin_memory):
            self.req_info: dict[int, tuple] = {}
            self._device = device
            self._state = ForcedRowsState()

        def is_argmax_invariant(self) -> bool:
            return False  # forcing changes the argmax by design

        def update_state(self, batch_update):
            def _new_state(params, _prompt_tok_ids, output_tok_ids):
                extra = getattr(params, "extra_args", None) or {}
                fs = extra.get(FORCED_SEED_EXTRA_KEY)
                if not fs:
                    return None
                return (fs, output_tok_ids)  # live ref → len() = tokens done

            # Rebuild the persistent device state ONLY when the batch makeup
            # changed (§3.2: the row-index tensor is reused across steps).
            if process_dict_updates(self.req_info, batch_update, _new_state):
                self._state.rebuild(self.req_info, self._device)

        def apply(self, logits: torch.Tensor) -> torch.Tensor:
            return self._state.apply(logits)

    return VLLMForcedSeedBatchedLogitsProcessor


def build_forced_seed_logitsproc_class():
    """SELECTOR: the batched processor by default; the legacy per-row adapter
    with ``RELIQUARY_VLLM_BATCHED_FS=0`` (A/B measurement, or rollback if the
    batched path misbehaves on some vLLM version). Same wire behaviour either
    way — the batched core is parity-locked (test_batched_forced_processor)."""
    import os as _os
    if _os.environ.get("RELIQUARY_VLLM_BATCHED_FS", "1") == "1":
        return build_forced_seed_batched_logitsproc_class()
    return build_forced_seed_adapter_logitsproc_class()


def build_forced_seed_adapter_logitsproc_class():
    """Return the ``AdapterLogitsProcessor`` subclass (imports vLLM lazily)."""
    from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

    class VLLMForcedSeedLogitsProcessor(AdapterLogitsProcessor):
        """Batch adapter: one forced-seed closure per request that advertises a
        ``forced_seed`` extra_args payload; requests without it are untouched."""

        def is_argmax_invariant(self) -> bool:
            # Forcing changes which token wins → NOT argmax-invariant.
            return False

        def new_req_logits_processor(self, params):
            extra = getattr(params, "extra_args", None) or {}
            fs = extra.get(FORCED_SEED_EXTRA_KEY)
            if not fs:
                return None
            return make_forced_seed_request_proc(
                randomness=fs["randomness"], prompt_idx=fs["prompt_idx"],
                checkpoint_hash=fs["checkpoint_hash"],
                rollout_index=fs["rollout_index"],
                base_offset=fs.get("base_offset", 0),
                start_len=fs.get("start_len", 0),
            )

    return VLLMForcedSeedLogitsProcessor
